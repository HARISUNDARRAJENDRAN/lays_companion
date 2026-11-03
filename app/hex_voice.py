"""Local speech-to-text through Hex's /streams API (docs/streams.md in anomalyco/hex).

Hex keeps its Parakeet model resident on a Vulkan GPU. This client plays the role
Deepgram Flux plays in Computah: push 16 kHz mono audio, receive partials and a
final transcript. Turn end is a local energy VAD (or an explicit stop).
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HEX_DIR = Path(os.environ.get("APPDATA", Path.home())) / "hex"
RATE = 16000


def hex_exe() -> Path:
    current = (HEX_DIR / "current-version").read_text().strip() if (HEX_DIR / "current-version").is_file() else None
    exe = HEX_DIR / "versions" / (current or sorted(p.name for p in (HEX_DIR / "versions").iterdir())[-1]) / "hex.exe"
    if not exe.is_file():
        raise FileNotFoundError("Hex is not installed; run the Hex installer first")
    return exe


def ensure_hex(voice_gpu: str = "auto", timeout: float = 45) -> dict:
    """Prefer the requested GPU; recover NVIDIA startup failures on Iris Xe."""
    try:
        return _ensure_hex(voice_gpu, timeout)
    except Exception as first:
        if voice_gpu != "nvidia":
            raise
        print(f"NVIDIA voice startup failed ({type(first).__name__}); trying Iris Xe", flush=True)
        return _ensure_hex("iris", timeout)


def _ensure_hex(voice_gpu: str = "auto", timeout: float = 45) -> dict:
    """Start Hex hidden if needed. voice_gpu: auto | nvidia | iris | cpu.

    ggml's Vulkan backend honours GGML_VK_VISIBLE_DEVICES; Hex enumerates the
    Iris Xe as Vulkan device 0 and the RTX 3050 as device 1 on this laptop.
    """
    exe = hex_exe()
    env = dict(os.environ)
    if voice_gpu != "auto":
        subprocess.run([str(exe), "quit", "--force"], capture_output=True, timeout=30)
        time.sleep(1.5)
        if voice_gpu == "cpu":
            env["GGML_VK_VISIBLE_DEVICES"] = ""
        else:
            env["GGML_VK_VISIBLE_DEVICES"] = _vulkan_index(exe, "NVIDIA" if voice_gpu == "nvidia" else "Iris")
    if not _api(quiet=True):
        subprocess.Popen([str(exe), "app", "--hidden"], env=env, creationflags=0x08000000,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + timeout
    while time.time() < deadline:
        api = _api(quiet=True)
        if api and _get(api, "/capabilities").get("streams"):
            try:
                Stream(api).cancel()  # 503 model-not-loaded until dictation holds a model
                return api
            except RuntimeError as exc:
                if "model-not-loaded" not in str(exc) and "engine-unavailable" not in str(exc):
                    raise
        time.sleep(1)
    raise TimeoutError("Hex did not become ready for /streams")


def loaded_device() -> str | None:
    log = HEX_DIR / "logs" / "process.log"
    if not log.is_file():
        return None
    import re
    text = re.sub(chr(27) + r"\[[0-9;]*m", "", log.read_text(encoding="utf-8", errors="replace"))
    for line in reversed(text.splitlines()):
        if "loaded local transcription model" in line and 'device="' in line:
            return line.split('device="', 1)[1].split('"', 1)[0]
    return None


def _vulkan_index(exe: Path, name: str) -> str:
    out = subprocess.run([str(exe), "model", "devices"], capture_output=True, text=True, timeout=60).stdout
    for row in out.splitlines():
        parts = row.split("\t")
        if len(parts) >= 4 and parts[1] == "vulkan" and name.lower() in parts[3].lower():
            return parts[0]
    raise RuntimeError(f"No Vulkan device matching {name!r}:\n{out}")


def _api(quiet=False):
    try:
        info = json.loads((HEX_DIR / "local-api.json").read_text(encoding="utf-8"))
        api = {"base": f"http://127.0.0.1:{info['port']}", "token": info["token"]}
        _get(api, "/capabilities")
        return api
    except Exception:
        if quiet:
            return None
        raise


def _request(api, method, path, body=None, headers=None, timeout=10):
    data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
    req = urllib.request.Request(api["base"] + path, data=data, method=method,
                                 headers={"Authorization": f"Bearer {api['token']}", **(headers or {})})
    if data is not None:
        req.add_header("Content-Length", str(len(data)))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{method} {path}: {exc.code} {exc.read().decode(errors='replace')}") from None


def _get(api, path):
    return _request(api, "GET", path)


class Stream:
    """One /streams session. on_partial(text) receives stable+tentative text."""

    def __init__(self, api, on_partial=None):
        self.api, self.on_partial = api, on_partial
        created = _request(api, "POST", "/streams", {"sampleFormat": "f32le", "sampleRate": RATE},
                           {"Content-Type": "application/json"})
        self.id, self.owner = created["id"], created["ownerToken"]
        self.partials = created.get("partials", False)
        self._segments, self._partial = {}, ""
        if on_partial:
            threading.Thread(target=self._events, daemon=True).start()

    def _h(self, extra=None):
        return {"X-Hex-Stream-Token": self.owner, **(extra or {})}

    def push(self, samples: np.ndarray):
        data = np.asarray(samples, dtype="<f4").clip(-1, 1).tobytes()
        for i in range(0, len(data), 256 * 1024 - 4):
            _request(self.api, "POST", f"/streams/{self.id}/audio", data[i:i + 256 * 1024 - 4],
                     self._h({"Content-Type": "application/octet-stream"}))

    def finish(self) -> str:
        return _request(self.api, "POST", f"/streams/{self.id}/finish", b"", self._h(), timeout=125).get("text", "").strip()

    def cancel(self):
        try:
            _request(self.api, "POST", f"/streams/{self.id}/cancel", b"", self._h())
        except RuntimeError:
            pass

    def _events(self):
        req = urllib.request.Request(f"{self.api['base']}/streams/{self.id}/events",
                                     headers={"Authorization": f"Bearer {self.api['token']}", **self._h()})
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                for raw in response:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[5:])
                    if event["type"] == "segment":
                        self._segments[event["segment"]] = event["text"]
                        self._partial = ""
                    elif event["type"] == "partial":
                        self._partial = event.get("stable", "") + event.get("tentative", "")
                    elif event["type"] in ("final", "error", "cancelled"):
                        return
                    text = " ".join([*(self._segments[k] for k in sorted(self._segments)), self._partial]).strip()
                    self.on_partial(text)
        except Exception:
            return


def listen(api, stop: threading.Event, on_partial=None, silence_s=1.2, max_s=30, device=None) -> str:
    """Record the microphone into a Hex stream until `stop` or end of turn (silence after speech)."""
    import sounddevice as sd

    stream = Stream(api, on_partial)
    started, last_voice, heard, floor = time.time(), None, False, None
    try:
        with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=RATE // 10, device=device) as mic:
            while not stop.is_set() and time.time() - started < max_s:
                block, _ = mic.read(RATE // 10)
                block = block[:, 0]
                stream.push(block)
                rms = float(np.sqrt(np.mean(block ** 2)))
                floor = rms if floor is None else min(floor * 1.01 + 1e-5, max(floor, rms) if not heard else floor)
                if rms > max(0.012, 3 * (floor or 0)):
                    heard, last_voice = True, time.time()
                if heard and time.time() - last_voice > silence_s:
                    break
        return stream.finish()
    except BaseException:
        stream.cancel()
        raise


def transcribe_wav(api, path) -> tuple[str, float]:
    import wave
    with wave.open(str(path)) as w:
        assert w.getframerate() == RATE and w.getnchannels() == 1 and w.getsampwidth() == 2
        samples = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    stream = Stream(api)
    stream.push(samples)
    t = time.perf_counter()
    text = stream.finish()
    return text, (time.perf_counter() - t) * 1000


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Hex voice check")
    p.add_argument("--voice-gpu", choices=("auto", "nvidia", "iris", "cpu"), default="auto")
    p.add_argument("--wav")
    p.add_argument("--mic", action="store_true", help="Speak one command")
    a = p.parse_args()
    api = ensure_hex(a.voice_gpu)
    print("hex device:", loaded_device())
    if a.wav:
        for _ in range(3):
            text, ms = transcribe_wav(api, a.wav)
            print(f"{ms:.0f} ms after finish: {text}")
    if a.mic:
        print("Speak now...")
        print("FINAL:", listen(api, threading.Event(), lambda t: print("  ..", t)))
