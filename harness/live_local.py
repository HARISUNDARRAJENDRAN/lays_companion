"""Persistent local LAYS GUI agent. Dry-run by default; --execute is user-run only."""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

os.environ.setdefault("LAYS_PANEL_FORMAT", "P2")
os.environ.setdefault("LAYS_FLA_CONV", "0")

from desktop_backend import DesktopBackend, action_plan
from driver import Observation
from observe import visual_change
from schema import Action, Geometry


class LaysLive:
    def __init__(self, gen_dir, device="cuda", dtype="bf16", policy=None, work_dir=None):
        if policy is None:
            import torch
            from agent import Lays
            policy = Lays(gen_dir, device=device, dtype={"bf16": torch.bfloat16, "fp16": torch.float16,
                                                      "fp32": torch.float32}[dtype])
        self.pol, self.gen = policy, str(gen_dir)
        self._temp = tempfile.TemporaryDirectory(prefix="lays-live-") if work_dir is None else None
        self.directory = Path(work_dir or self._temp.name)
        self.directory.mkdir(parents=True, exist_ok=True)

    def close(self):
        if self._temp:
            self._temp.cleanup()

    def step(self, png: bytes, prev_png: bytes | None, goal: str, history: list, k: int, size: tuple):
        start = time.perf_counter()
        current = self.directory / f"o{k}.png"
        current.write_bytes(png)
        with Image.open(current) as im:
            if im.size != tuple(size):
                raise ValueError("size disagrees with PNG dimensions")
        prev, visual = None, "unknown"
        if prev_png:
            prev = str(self.directory / f"o{k - 1}.png")
            Path(prev).write_bytes(prev_png)
            with Image.open(prev) as p, Image.open(current) as c:
                visual = visual_change(p, c)
        if history:
            history[-1]["visual"] = visual
        observation = Observation(f"o{k}", str(current), Geometry(0, 0, *size), None, None)
        observation.os = "windows" if sys.platform == "win32" else "linux"
        action = self.pol(SimpleNamespace(goal=goal, id="live"), observation, history, prev, k)
        diagnostics = getattr(self.pol, "diagnostics", None)
        if not math.isfinite(self.pol.p_done):
            raise RuntimeError("Non-finite policy probability; check inference dtype")
        return {"action": action.to_json(), "line": action.short(), "visual_prev": visual,
                "p_done": round(self.pol.p_done, 3), "latency_ms": int(1000 * (time.perf_counter() - start)),
                "ckpt": self.gen, **({"computah": diagnostics} if diagnostics is not None else {})}


def change_fraction(before, after):
    arrays = []
    for path in (before, after):
        with Image.open(path) as im:
            arrays.append(np.asarray(im.convert("L").resize((320, 180)), dtype=np.int16))
    return float((np.abs(arrays[0] - arrays[1]) > 24).mean())


def annotate(source, target, action):
    with Image.open(source) as original:
        im = original.convert("RGB")
    draw = ImageDraw.Draw(im)
    for point, color in ((action.point, "red"), (action.dest, "cyan")):
        if point:
            x, y = min(int(point[0] * im.width), im.width - 1), min(int(point[1] * im.height), im.height - 1)
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), outline=color, width=3)
            draw.line((x - 16, y, x + 16, y), fill=color, width=2)
            draw.line((x, y - 16, x, y + 16), fill=color, width=2)
    draw.rectangle((0, 0, im.width, 28), fill="black")
    draw.text((8, 7), action.short()[:180], fill="white")
    im.save(target)


def run(args, backend=None, live=None, observer=None):
    run_dir = args.runs_dir / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True)
    stop = threading.Event()
    watcher_stop = threading.Event()
    monitor = None
    if sys.platform == "win32":
        def watch():
            import windows_input
            while not watcher_stop.wait(0.03):
                if windows_input.emergency_stop():
                    stop.set()
                    return
        monitor = threading.Thread(target=watch, daemon=True)
        monitor.start()
    history, previous, rows = [], None, []
    reason, error, exit_code = "max_steps", None, 0
    owns_live = live is None
    execution_lock = None
    started = time.perf_counter()
    try:
        if args.execute and sys.platform == "win32" and not getattr(observer, "owns_execution_lock", False):
            from windows_input import ExecutionLock
            execution_lock = ExecutionLock()
        backend = backend or DesktopBackend(args.desktop_bin, args.screen)
        version = backend.call("version")
        (run_dir / "config.json").write_text(json.dumps({**vars(args), "driver": version}, default=str, indent=2), encoding="utf-8")
        print(f"Loading LAYS; run logs: {run_dir}", flush=True)
        if live is None and getattr(args, "policy_wrapper", "none") == "computah":
            import torch
            from agent import Lays
            from computah_r5.wrapper import ComputahR5
            base = Lays(args.ckpt, device=args.device, dtype={"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype])
            live = LaysLive(args.ckpt, args.device, args.dtype, policy=ComputahR5(base, backend), work_dir=run_dir / "policy")
        live = live or LaysLive(args.ckpt, args.device, args.dtype, work_dir=run_dir / "policy")
        if observer:
            observer.start(run_dir, args, live)
        if args.execute:
            print("EXECUTE: physical mouse/keyboard input. Ctrl+C or F8 stops the run.", flush=True)
            for remaining in (3, 2, 1):
                print(f"Starting in {remaining}...", flush=True)
                if stop.wait(1):
                    raise KeyboardInterrupt
        repeats, last_line = 0, None
        with (run_dir / "steps.jsonl").open("a", encoding="utf-8", buffering=1) as log:
            for k in range(args.max_steps):
                if time.perf_counter() - started > getattr(args, "time_limit_s", float("inf")):
                    reason = "time_limit"
                    break
                if stop.is_set():
                    raise KeyboardInterrupt
                step_start = time.perf_counter()
                before_capture = getattr(observer, "before_capture", None)
                if before_capture:
                    before_capture(k)
                frame = backend.capture(run_dir / f"{k:03d}.png")
                png = frame.path.read_bytes()
                with Image.open(io.BytesIO(png)) as im:
                    size = im.size
                if observer:
                    observer.before_step(k, frame, live, history, previous)
                result = live.step(png, previous, args.goal, history, k, size)
                action = Action.from_json(result["action"])
                plan = action_plan(action, frame.geometry, f"o{k}")
                annotate(frame.path, run_dir / f"{k:03d}-action.png", action)
                row = {**result, "action": json.loads(result["action"]), "k": k, "mode": "execute" if args.execute else "dry-run",
                       "capture_ms": frame.capture_ms, "plan": plan, "delivery": "not_delivered", "effect": "unknown"}
                rows.append(row)
                try:
                    if not args.execute:
                        row["status"] = "proposed"
                    elif action.family in ("done", "fail", "ask"):
                        row["status"] = "model_stop"
                    elif time.perf_counter() - frame.captured_at > args.max_observation_age:
                        row["status"], reason = "stale_observation", "observation_too_old"
                    else:
                        check = backend.capture(run_dir / f"{k:03d}-guard.png")
                        fraction = change_fraction(frame.path, check.path)
                        row["guard_change_fraction"] = fraction
                        if check.display != frame.display or check.foreground != frame.foreground or fraction > args.screen_change_limit:
                            row["status"] = "screen_changed_reobserve"
                        else:
                            repeats = repeats + 1 if action.family != "wait" and result["line"] == last_line and result["visual_prev"] == "unchanged" else 0
                            if repeats >= 2:
                                row["status"], reason = "repeat_guard", "repeated_action_without_screen_change"
                            else:
                                row["delivery"] = "unknown"  # do not assume an exception implies no input
                                if observer:
                                    observer.before_execute(action, frame)
                                row["reports"] = backend.execute(action, frame, f"o{k}", stop.is_set)
                                deliveries = [r.get("disposition", {}).get("delivery", "unknown") for r in row["reports"]]
                                if plan and not any(d.startswith("delivered_") for d in deliveries):
                                    row["delivery"] = "not_delivered" if all(d == "not_delivered" for d in deliveries) else "unknown"
                                    row["status"], reason = "delivery_unconfirmed", "delivery_unconfirmed"
                                else:
                                    row["delivery"] = "delivered_verified" if deliveries and all(d == "delivered_verified" for d in deliveries) else "delivered_unverified" if plan else "not_delivered"
                                    row["status"] = "dispatched" if plan else "waited"
                                    history.append({"action": result["line"], "delivery": "confirmed" if plan else "unknown",
                                                    "effect": "unknown", "visual": "unknown"})
                                    history[:] = history[-4:]
                                    previous, last_line = png, result["line"]
                                if stop.wait(args.settle_ms / 1000):
                                    raise KeyboardInterrupt
                except BaseException as exc:
                    row["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "delivery_error"
                    row["error"] = str(exc)
                    raise
                finally:
                    row["step_ms"] = (time.perf_counter() - step_start) * 1000
                    row["history"] = list(history)
                    if observer:
                        observer.after_step(row, frame, live)
                    log.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                if action.family in ("done", "fail", "ask"):
                    reason = f"model_{action.family}"
                    break
                if reason != "max_steps":
                    break
    except KeyboardInterrupt:
        reason, exit_code = "interrupted", 130
    except Exception as exc:
        reason, error, exit_code = "error", str(exc), 1
    finally:
        watcher_stop.set()
        if monitor:
            monitor.join(timeout=1)
        if owns_live and live is not None:
            live.close()
        if execution_lock is not None:
            execution_lock.close()
        summary = {"stop_reason": reason, "steps": len(rows), "executed": sum(r.get("status") == "dispatched" for r in rows),
                   "elapsed_ms": (time.perf_counter() - started) * 1000, "error": error, "run_dir": str(run_dir),
                   "task_success": "unverified", "exit_code": exit_code}
        if observer:
            observer.finish(summary, run_dir)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"summary": summary}), flush=True)
    return exit_code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--goal", required=True)
    p.add_argument("--ckpt", type=Path, default=Path.home() / "lays-ckpt/rl3-r5")
    p.add_argument("--max-steps", type=int, default=15)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Default: predict/log only; never send input")
    mode.add_argument("--execute", action="store_true")
    p.add_argument("--pixels", type=int, default=500000)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--screen", type=int, default=0)
    p.add_argument("--desktop-bin")
    p.add_argument("--runs-dir", type=Path, default=Path.home() / "lays-run/runs")
    p.add_argument("--settle-ms", type=int, default=350, help="Minimum app pacing delay; fresh screenshots still verify change")
    p.add_argument("--max-observation-age", type=float, default=15)
    p.add_argument("--screen-change-limit", type=float, default=0.02)
    p.add_argument("--policy-wrapper", choices=("none", "computah"), default="none",
                   help="computah: ground r5's actions on UIA controls, copy exact goal text, scope/repeat/completion guards")
    args = p.parse_args()
    if args.max_steps < 1 or args.pixels < 4096 or args.screen < 0 or args.settle_ms < 0 or args.max_observation_age <= 0 or not 0 <= args.screen_change_limit <= 1:
        p.error("Invalid step, pixel, screen, timing, or change-limit value")
    if sys.platform != "win32":
        p.error("This agent-desktop harness currently supports Windows")
    if not (args.ckpt / "trainable.safetensors").is_file():
        p.error(f"Missing checkpoint: {args.ckpt}")
    os.environ["LAYS_PIXELS"] = str(args.pixels)
    os.environ["LAYS_PANEL_FORMAT"] = "P2"
    local_model = Path.home() / "lays-run/model"
    if local_model.exists():
        os.environ.setdefault("LAYS_MODEL_DIR", str(local_model))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
