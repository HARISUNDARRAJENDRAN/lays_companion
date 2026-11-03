"""LAYS Companion: voice-driven Windows desktop agent.

The Computah command flow (github.com/musubipapi/computah), on Windows and fully local:

  hotkey -> microphone -> Hex /streams (Parakeet on a Vulkan GPU) -> final speech turn
  -> C6 harness (UIA snapshot + candidate controls, concurrent with inference)
  -> LAYS round-2 checkpoint selects the next action -> C6 guards check it
  -> agent-desktop sends input -> fresh observation -> repeat until done / fail / ask / limits
  -> overlay ("notch") shows transcript, each step and the result.

A new speech turn (pressing the hotkey while a task runs) revokes the running
command's permission to send more input, as Computah's StartOfTurn does. F8
remains the emergency stop. Input already sent cannot be undone.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path
from types import SimpleNamespace

HOME = Path.home()
ROOT = Path(__file__).resolve().parent.parent
os.environ["LAYS_DESKTOP_BIN"] = str(ROOT / "bin/agent-desktop.exe")
C6_AGENT = ROOT / "harness"
ROUND2 = ROOT / "models/round2/molab-round2-fast"
# Frozen C6 configuration (REPORT-FINAL section 7).
C6_FLAGS = {"S1_VERIFY_DONE": "1", "S1_C6_CONFIRM_GATE": "1", "S1_C6_CYCLE_HELPERS": "1", "S1_C6_TIMING": "1",
            "S1_C6_PRECOMMIT": "1", "S1_C6_TARGET_GROUNDING": "1", "S1_C6_NATIVE_SAFETY": "1",
            "S1_C6_NATIVE_GROUNDING": "0", "LAYS_PANEL_FORMAT": "P2", "LAYS_FLA_CONV": "0"}

user32 = ctypes.windll.user32


class Overlay:
    """Small always-on-top status strip at the top of the screen. It never takes focus
    and is excluded from screen capture, so the model never sees it."""

    COLORS = {"idle": "#3a3a3a", "listening": "#0b7a3e", "thinking": "#1f4fa3", "acting": "#8a5a00",
              "done": "#0b7a3e", "stopped": "#8a1f1f"}

    def __init__(self):
        import tkinter as tk
        self.tk = tk
        self.q = queue.Queue()
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True, "-alpha", 0.92)
        width = 640
        self.root.geometry(f"{width}x58+{(self.root.winfo_screenwidth() - width) // 2}+6")
        self.frame = tk.Frame(self.root, bg=self.COLORS["idle"])
        self.frame.pack(fill="both", expand=True)
        self.title = tk.Label(self.frame, fg="white", bg=self.COLORS["idle"], font=("Segoe UI Semibold", 11), anchor="w")
        self.detail = tk.Label(self.frame, fg="#e8e8e8", bg=self.COLORS["idle"], font=("Segoe UI", 9), anchor="w")
        self.title.pack(fill="x", padx=12, pady=(6, 0))
        self.detail.pack(fill="x", padx=12)
        self.root.update_idletasks()
        hwnd = user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()
        GWL_EXSTYLE, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, WS_EX_TRANSPARENT, WS_EX_LAYERED = -20, 0x08000000, 0x80, 0x20, 0x80000
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TRANSPARENT | WS_EX_LAYERED)
        user32.SetWindowDisplayAffinity(hwnd, 0x11)  # WDA_EXCLUDEFROMCAPTURE
        self.hwnd = hwnd
        self.set("idle", "LAYS ready", "Ctrl+Alt+Space: speak a command   |   F8: emergency stop")
        self.root.after(50, self._drain)

    def set(self, state, title, detail=""):
        self.q.put((state, title, detail))

    def _drain(self):
        try:
            while True:
                state, title, detail = self.q.get_nowait()
                color = self.COLORS.get(state, self.COLORS["idle"])
                for w in (self.frame, self.title, self.detail):
                    w.configure(bg=color)
                self.title.configure(text=title[:90])
                self.detail.configure(text=detail[:120])
        except queue.Empty:
            pass
        self.root.after(50, self._drain)

    def mainloop(self):
        self.root.mainloop()


def speak(text: str, enabled: bool):
    if enabled and text:
        safe = text.replace("'", "''")[:200]
        subprocess.Popen(["powershell", "-NoProfile", "-Command",
                          f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe}')"],
                         creationflags=0x08000000)


class StepObserver:
    """live_local.run observer: mirrors each step onto the overlay."""

    def __init__(self, ui, goal):
        self.ui, self.goal = ui, goal

    def start(self, run_dir, args, live):
        self.ui.set("thinking", f"'{self.goal}'", "Observing the screen...")

    def before_step(self, k, frame, live, history, previous):
        self.ui.set("thinking", f"'{self.goal}'", f"Step {k + 1}: reading controls and choosing an action...")

    def before_execute(self, action, frame):
        self.ui.set("acting", f"'{self.goal}'", f"Doing: {action.short()}")

    def after_step(self, row, frame, live):
        c6 = row.get("computah") or {}
        target = c6.get("snap_target")
        note = f" -> {target}" if target else ""
        self.ui.set("thinking", f"'{self.goal}'", f"Step {row['k'] + 1}: {row.get('line')}{note} [{row.get('status')}] "
                                                 f"{row.get('latency_ms', 0)} ms")

    def finish(self, summary, run_dir):
        pass


RESULT_TEXT = {"model_done": ("done", "Done"), "model_fail": ("stopped", "I couldn't do that"),
               "model_ask": ("stopped", "I need more information"), "max_steps": ("stopped", "Step limit reached"),
               "interrupted": ("stopped", "Stopped"),
               "repeated_action_without_screen_change": ("stopped", "Stopped: the screen stopped changing"),
               "delivery_unconfirmed": ("stopped", "Stopped: input delivery unconfirmed"),
               "observation_too_old": ("stopped", "Stopped: observation too old"), "error": ("stopped", "Error")}


class Companion:
    def __init__(self, a, ui):
        self.a, self.ui = a, ui
        self.cancel = threading.Event()   # revokes the running command's input permission
        self.busy = threading.Lock()
        self.listen_stop = threading.Event()
        self.listening = False
        self.commands = queue.Queue()
        self.api = None
        self.turn_lock = threading.RLock()
        self.generation = 0
        self.finished = threading.Event()
        self.shutdown = threading.Event()
        self.exit_code = 0

    def new_turn(self):
        with self.turn_lock:
            self.generation += 1
            self.cancel.set()
            return self.generation

    def submit(self, text, generation=None):
        with self.turn_lock:
            generation = self.new_turn() if generation is None else generation
            if generation == self.generation and not self.shutdown.is_set():
                self.commands.put((generation, text))

    def close(self):
        self.shutdown.set()
        self.new_turn()
        self.listen_stop.set()
        self.commands.put(None)

    # ---------------------------------------------------------------- startup
    def load(self):
        os.environ.update(C6_FLAGS)
        os.environ["LAYS_PIXELS"] = str(self.a.pixels)
        os.environ["LAYS_MODEL_DIR"] = str(ROOT / "models/base")
        os.environ["HF_HUB_OFFLINE"] = "1"
        sys.path.insert(0, str(C6_AGENT))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        if not self.a.no_voice:
            import hex_voice
            self.ui.set("thinking", "Starting voice (Hex)...", f"voice GPU: {self.a.voice_gpu}")
            self.api = hex_voice.ensure_hex(self.a.voice_gpu)
            self.voice_device = hex_voice.loaded_device()
        self.ui.set("thinking", "Loading LAYS round 2 + C6 harness...", str(self.a.ckpt.name))
        import torch
        import windows_input
        from agent import Lays
        from computah_r5.wrapper import ComputahR5
        from desktop_backend import DesktopBackend
        from live_local import LaysLive
        original = windows_input.emergency_stop
        windows_input.emergency_stop = lambda: original() or self.cancel.is_set()
        self.backend = DesktopBackend(None, self.a.screen)
        base = Lays(self.a.ckpt, device=self.a.device,
                    dtype={"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.a.dtype])
        self.work = ROOT / "data"
        self.work.mkdir(parents=True, exist_ok=True)
        self.live = LaysLive(self.a.ckpt, self.a.device, self.a.dtype, policy=ComputahR5(base, self.backend),
                             work_dir=self.work / "model-inputs")
        voice = f"voice on {getattr(self, 'voice_device', None) or 'off'}"
        self.ui.set("idle", "LAYS ready", f"Ctrl+Alt+Space: speak | F8: stop | {voice}")
        speak("LAYS is ready.", self.a.speak)

    # ---------------------------------------------------------------- voice turns
    def hotkey_pressed(self):
        with self.turn_lock:
            if self.listening:
                self.listen_stop.set()
                return
            generation = self.new_turn()
            self.listening = True
            self.listen_stop.clear()
        threading.Thread(target=self._listen, args=(generation,), daemon=True).start()

    def _listen(self, generation=None):
        import hex_voice
        if self.api is None:
            self.listening = False
            self.ui.set("stopped", "Voice is off", "Start with voice enabled, or type commands in the console")
            return
        self.ui.set("listening", "Listening...", "Speak your command; pause or press Ctrl+Alt+Space to finish")
        try:
            text = hex_voice.listen(self.api, self.listen_stop,
                                    lambda t: self.ui.set("listening", "Listening...", t))
        except Exception as exc:
            text = ""
            self.ui.set("stopped", "Voice error", str(exc))
            return
        finally:
            self.listening = False
        if not text:
            self.ui.set("idle", "Didn't catch that", "Ctrl+Alt+Space to try again")
            return
        # The new turn supersedes the old one; allow only this fresh command to proceed.
        self.submit(text, generation)

    # ---------------------------------------------------------------- execution
    def worker(self):
        while True:
            item = self.commands.get()
            if item is None:
                return
            generation, goal = item
            with self.busy:
                with self.turn_lock:
                    if generation != self.generation or self.shutdown.is_set():
                        continue
                    # Only restore input after the previous run and its watcher exit.
                    self.cancel.clear()
                try:
                    self.exit_code = self.execute(goal) or 0
                except Exception as exc:
                    self.exit_code = 1
                    self.ui.set("stopped", "Command error", str(exc))
                    print("Command failed:", exc, flush=True)
                finally:
                    if self.a.command:
                        self.finished.set()

    def execute(self, goal):
        import live_local
        args = SimpleNamespace(goal=goal, ckpt=self.a.ckpt, max_steps=self.a.max_steps, dry_run=not self.a.execute,
                               execute=self.a.execute, pixels=self.a.pixels, device=self.a.device, dtype=self.a.dtype,
                               screen=self.a.screen, desktop_bin=None, runs_dir=self.work / "runs", settle_ms=350,
                               max_observation_age=15, screen_change_limit=0.02, policy_wrapper="computah")
        summaries = []
        observer = StepObserver(self.ui, goal)
        observer.finish = lambda summary, run_dir: summaries.append(summary)
        started = time.time()
        live_local.run(args, backend=self.backend, live=self.live, observer=observer)
        summary = summaries[0] if summaries else {"stop_reason": "error", "error": "no summary"}
        state, text = RESULT_TEXT.get(summary["stop_reason"], ("stopped", summary["stop_reason"]))
        if not self.a.execute:
            state, text = "idle", f"Dry run finished ({summary['stop_reason']})"
        detail = f"{summary.get('steps', 0)} steps, {summary.get('executed', 0)} actions, {time.time() - started:.1f} s"
        if summary.get("error"):
            detail += f" | {summary['error']}"
        self.ui.set(state, f"{text}: '{goal}'", detail)
        speak(text, self.a.speak)
        with (self.work / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.time(), "goal": goal, "summary": summary}) + "\n")
        return summary.get("exit_code", 1)


def hotkey_loop(callback, stop: threading.Event):
    MOD_ALT, MOD_CONTROL, MOD_NOREPEAT, VK_SPACE, WM_HOTKEY = 0x1, 0x2, 0x4000, 0x20, 0x0312
    if not user32.RegisterHotKey(None, 1, MOD_ALT | MOD_CONTROL | MOD_NOREPEAT, VK_SPACE):
        print("Could not register Ctrl+Alt+Space (in use by another app)", flush=True)
        return
    msg = wintypes.MSG()
    try:
        while not stop.is_set():
            if user32.PeekMessageW(ctypes.byref(msg), None, WM_HOTKEY, WM_HOTKEY, 1) and msg.message == WM_HOTKEY:
                callback()
            time.sleep(0.02)
    finally:
        user32.UnregisterHotKey(None, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", type=Path, default=ROUND2)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Send real mouse/keyboard input")
    mode.add_argument("--dry-run", action="store_true", help="Default: predict and log only")
    p.add_argument("--voice-gpu", choices=("auto", "nvidia", "iris", "cpu"), default="nvidia",
                   help="Vulkan device for Hex speech recognition (iris = Intel Iris Xe)")
    p.add_argument("--no-voice", action="store_true", help="Typed commands only (console)")
    p.add_argument("--speak", action=argparse.BooleanOptionalAction, default=True, help="Spoken status replies")
    p.add_argument("--max-steps", type=int, default=25)
    p.add_argument("--pixels", type=int, default=500000)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--screen", type=int, default=0)
    p.add_argument("--stop-file", type=Path, help="Local graceful shutdown signal")
    p.add_argument("--command", help="Run one typed command, then exit (no overlay loop)")
    a = p.parse_args()
    if not (a.ckpt / "trainable.safetensors").is_file():
        p.error(f"Missing checkpoint: {a.ckpt}")
    user32.SetProcessDPIAware()

    ui = Overlay()
    companion = Companion(a, ui)
    stop = threading.Event()

    boot_done = threading.Event()
    worker_thread = None

    def boot():
        nonlocal worker_thread
        try:
            companion.load()
            if stop.is_set():
                return
            worker_thread = threading.Thread(target=companion.worker, daemon=True)
            worker_thread.start()
            if a.command:
                companion.submit(a.command)
                return
            threading.Thread(target=hotkey_loop, args=(companion.hotkey_pressed, stop), daemon=True).start()
            print(f"READY: checkpoint={a.ckpt}; voice={getattr(companion, 'voice_device', 'off')}; execute={a.execute}", flush=True)
            if sys.stdin is not None:
                for line in sys.stdin:
                    if line.strip():
                        companion.submit(line.strip())
        except Exception as exc:
            companion.exit_code = 1
            ui.set("stopped", "Startup failed", str(exc))
            print("Startup failed:", exc, flush=True)
            companion.finished.set()
        finally:
            boot_done.set()

    threading.Thread(target=boot, daemon=True).start()
    def poll():
        if (a.stop_file and a.stop_file.exists()) or companion.finished.is_set():
            companion.close()
            stop.set()
            ui.root.destroy()
            return
        # F8 also invalidates speech that has not finished transcribing yet.
        if user32.GetAsyncKeyState(0x77) & 0x8000:
            companion.new_turn()
            companion.listen_stop.set()
        ui.root.after(50, poll)
    ui.root.after(50, poll)
    try:
        ui.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        companion.close()
        if worker_thread:
            worker_thread.join(timeout=10)
    return companion.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
