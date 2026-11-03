"""agent-desktop 1.0.2 adapter for LAYS screenshot-space actions. No shell execution."""
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from schema import ActionError, Geometry, MODS, to_screen

CAPABILITIES = {"click", "double_click", "right_click", "hover", "type", "key", "scroll", "drag",
                "wait", "done", "fail", "ask"}
KEY_NAMES = {"enter": "return", "delete": "forwarddelete", "pageup": "pageup", "pagedown": "pagedown",
             "minus": "-", "equal": "=", "bracketleft": "[", "bracketright": "]", "backslash": "\\",
             "semicolon": ";", "quote": "'", "grave": "`", "comma": ",", "period": ".", "slash": "/"}


@dataclass
class Frame:
    path: Path
    geometry: Geometry
    display: dict
    captured_at: float
    foreground: int | None
    capture_ms: float


def binary_path(explicit=None):
    path = explicit or os.environ.get("LAYS_DESKTOP_BIN")
    if not path:
        local = Path.home() / "lays-run/desktop/node_modules/agent-desktop/bin/agent-desktop-win32-x64.exe"
        path = str(local) if local.exists() else shutil.which("agent-desktop")
    if not path or not Path(path).is_file() or Path(path).suffix.lower() in (".cmd", ".ps1"):
        raise RuntimeError("Set --desktop-bin to the native agent-desktop executable (see HARNESS.md)")
    return str(Path(path).resolve())


def action_plan(action, geom, obs_id):
    action.validate(CAPABILITIES)
    if action.route == "element":
        raise ActionError("Pixel-only harness cannot resolve an element route")
    if action.route != "none" and action.obs_id != obs_id:
        raise ActionError("Action refers to a stale observation")
    pt = action.point or (0.5, 0.5)
    xy = to_screen(pt, geom, obs_id, obs_id)
    point = f"{xy[0]},{xy[1]}"
    f = action.family
    if f in ("click", "double_click", "right_click"):
        return [["mouse-click", f"--xy={point}", "--button", "right" if f == "right_click" else "left",
                 "--count", "2" if f == "double_click" else "1"]]
    if f == "hover":
        return [["mouse-move", f"--xy={point}"]]
    if f == "drag":
        dest = to_screen(action.dest, geom, obs_id, obs_id)
        return [["drag", f"--from-xy={point}", f"--to-xy={dest[0]},{dest[1]}", "--duration", "400"]]
    if f == "scroll":
        direction, amount = action.scroll
        dx = amount * (1 if direction == "right" else -1) if direction in ("left", "right") else 0
        dy = amount * (1 if direction == "up" else -1) if direction in ("up", "down") else 0
        return [["mouse-wheel", f"--x={xy[0]}", f"--y={xy[1]}", f"--dx={dx}", f"--dy={dy}"]]
    if f == "key":
        if action.key in ("insert", "printscreen", "capslock", "menu"):
            return [["win32-key", action.key, str(action.mods)]]
        parts = ["cmd" if m == "meta" else m for i, m in enumerate(MODS) if action.mods >> i & 1]
        if action.key != "none":
            parts.append(KEY_NAMES.get(action.key, action.key))
        return [["press", "+".join(parts)]]
    if f == "type":
        commands = [["mouse-click", f"--xy={point}"]] if action.route == "pixel" else []
        if action.type_mode == "replace":
            commands.append(["press", "ctrl+a"])
        return commands + [["unicode-text", action.text]]
    return []


class DesktopBackend:
    def __init__(self, binary=None, screen=0, timeout=10):
        self.binary = binary_path(binary)
        self.screen, self.timeout = screen, timeout
        self.env = dict(os.environ, AGENT_DESKTOP_HOME=str(Path.home() / "lays-run/desktop-state"))

    def call(self, *args):
        result = subprocess.run([self.binary, *map(str, args)], capture_output=True, text=True,
                                encoding="utf-8", timeout=self.timeout, env=self.env)
        try:
            envelope = json.loads(result.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(f"agent-desktop returned invalid JSON (exit {result.returncode})") from None
        if result.returncode or not envelope.get("ok"):
            raise RuntimeError(json.dumps(envelope.get("error", {"code": "CLI_ERROR"}), ensure_ascii=False))
        return envelope["data"]

    def uia_snapshot(self):
        """Fresh UIA controls belonging to the current foreground window."""
        # Physical-pixel boxes of the foreground window's interactive controls. agent-desktop reports no app pid for
        # Chrome windows, so ownership is the snapshot's own window; inside a browser, only the web document counts.
        # Inside a browser only descendants of the web document count: omnibox suggestions and context menus overlap
        # the page rectangle but belong to the browser, and activating them navigates away from the task.
        raw = self.call("snapshot", "-i", "--include-bounds")
        win = raw.get("window") or {}
        self.last_document_box = None
        self.last_window_box = None
        self.last_is_browser = False
        def boxed(node):
            box = node.get("bounds") or {}
            return all(k in box for k in ("x", "y", "width", "height")) and box["width"] > 1 and box["height"] > 1
        # agent-desktop versions expose the foreground bounds either on `window` or
        # the root Window node. Native apps have no web document to scope to.
        wb = win.get("bounds") or (raw.get("tree") or {}).get("bounds") or {}
        if all(k in wb for k in ("x", "y", "width", "height")) and wb["width"] > 1 and wb["height"] > 1:
            self.last_window_box = (wb["x"], wb["y"], wb["width"], wb["height"])
        if win.get("id") is None: return []
        documents = []
        def find_docs(node):
            if not isinstance(node, dict): return
            if node.get("role") == "document" and boxed(node):
                documents.append(node)
            for child in node.get("children") or []: find_docs(child)
        find_docs(raw.get("tree") or {})
        root = raw.get("tree") or {}
        identity = " ".join(str(win.get(k) or "") for k in
                             ("title", "name", "app", "application", "process", "process_name", "class_name"))
        browser_identity = re.search(r"\b(chrome|msedge|edge|firefox|brave|opera|vivaldi|browser)\b", identity, re.I)
        browser_chrome = False
        def find_browser_chrome(node):
            nonlocal browser_chrome
            if not isinstance(node, dict):
                return
            role = str(node.get("role") or "").lower()
            name = str(node.get("name") or "").lower()
            if role in {"omnibox", "browser address bar"} or name == "address and search bar":
                browser_chrome = True
            for child in node.get("children") or []:
                find_browser_chrome(child)
        find_browser_chrome(raw.get("tree") or {})
        self.last_is_browser = bool(documents and (browser_identity or browser_chrome))
        if self.last_is_browser:
            root = max(documents, key=lambda n: n["bounds"]["width"] * n["bounds"]["height"])
            b = root["bounds"]
            self.last_document_box = (b["x"], b["y"], b["width"], b["height"])
        out = []
        def walk(node):
            if not isinstance(node, dict): return
            role = node.get("role") or ""
            native_document = not self.last_is_browser and str(role).lower() == "document"
            if (node is not root or native_document) and node.get("ref_id") and (role != "document" or not self.last_is_browser) and boxed(node):
                box = node["bounds"]
                states = list(node.get("states") or [])
                out.append({"name": node.get("name") or "", "role": node.get("role") or "", "value": node.get("value") or "",
                            "states": states, "enabled": node.get("enabled", "disabled" not in states),
                            "editable": node.get("editable", node.get("is_editable")),
                            "box": (box["x"], box["y"], box["width"], box["height"]), "window_id": win.get("id")})
            for child in node.get("children") or []: walk(child)
        walk(root)
        return out

    def display(self):
        displays = self.call("list-displays")
        if not 0 <= self.screen < len(displays):
            raise RuntimeError(f"Display index {self.screen} is unavailable")
        return displays[self.screen]

    def capture(self, path):
        import windows_input as win
        start = time.perf_counter()
        display = self.display()
        before = win.foreground()
        meta = self.call("screenshot", "--screen", self.screen, str(path))
        bounds = display["bounds"]
        with Image.open(path) as im:
            if im.size != (meta["width"], meta["height"]):
                raise RuntimeError("Capture metadata disagrees with image size")
        # On Windows both the display bounds and mouse input are PHYSICAL pixels.
        if (meta["width"], meta["height"]) != (int(bounds["width"]), int(bounds["height"])):
            raise RuntimeError("Display geometry changed during capture; capture again")
        if win.foreground() != before or self.display() != display:
            raise RuntimeError("Foreground/display changed during capture; capture again")
        return Frame(path, Geometry(**{k: bounds[k] for k in ("x", "y", "width", "height")}),
                     display, time.perf_counter(), before, (time.perf_counter() - start) * 1000)

    def execute(self, action, frame, obs_id, stopped):
        import windows_input as win
        commands = action_plan(action, frame.geometry, obs_id)
        if self.display() != frame.display or win.foreground() != frame.foreground:
            raise RuntimeError("Display or foreground changed since observation; no action sent")
        reports = []
        expected_foreground = frame.foreground
        for command in commands:
            if stopped():
                raise KeyboardInterrupt
            if win.foreground() != expected_foreground:
                raise RuntimeError("Foreground changed during action; delivery may be partial")
            if command[0] == "unicode-text":
                report = win.type_unicode(command[1], stopped, expected_foreground)
            elif command[0] == "win32-key":
                report = win.press_special(command[1], int(command[2]), stopped)
            else:
                report = self.call("--headed", *command)
            reports.append(report)
            delivery = report.get("disposition", {}).get("delivery", "unknown")
            if delivery == "not_delivered":
                break
            if not delivery.startswith("delivered_"):
                raise RuntimeError("Action delivery is unknown; remaining commands were not sent")
            if command[0] == "mouse-click":
                expected_foreground = win.foreground()
        return reports
