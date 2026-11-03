"""Typed Linux desktop driver (lays-action/1) for live evaluation and the Run 2-3 RL environments.

One Desktop = one isolated X display (Xvfb + openbox + session D-Bus + AT-SPI) with its own HOME. observe() returns
everything policy.render needs; execute() validates an Action against the observation it was decided on (stale
bindings refused) and delivers it with xdotool. Screen content is untrusted: it only ever reaches the model as
observation, never as instructions to this process.

  python driver.py selftest          # starts a display, opens a window, exercises every family
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from observe import visual_change
from schema import Action, ActionError, Geometry, MODS, to_screen

HERE = Path(__file__).resolve().parent
CAPS = ["click", "double_click", "right_click", "hover", "type", "key", "scroll", "drag", "wait", "done", "fail", "ask",
        "launch_app", "focus_window"]
XKEY = {"enter": "Return", "tab": "Tab", "escape": "Escape", "backspace": "BackSpace", "delete": "Delete",
        "space": "space", "up": "Up", "down": "Down", "left": "Left", "right": "Right", "home": "Home", "end": "End",
        "pageup": "Prior", "pagedown": "Next", "insert": "Insert", "printscreen": "Print", "capslock": "Caps_Lock",
        "menu": "Menu", "minus": "minus", "equal": "equal", "bracketleft": "bracketleft", "bracketright": "bracketright",
        "backslash": "backslash", "semicolon": "semicolon", "quote": "apostrophe", "grave": "grave", "comma": "comma",
        "period": "period", "slash": "slash"}
XMOD = {"ctrl": "ctrl", "alt": "alt", "shift": "shift", "meta": "super"}
WHEEL = {"up": 4, "down": 5, "left": 6, "right": 7}


def type_runs(text):
    """xdotool argument lists that type `text`. xdotool binds a letter missing from the keymap to a spare keycode,
    which X reads as [lower, upper], and sends no Shift: an uppercase non-ASCII letter comes out lower case. Those
    go as Shift + the lower-case keysym instead."""
    out, buf = [], ""
    for c in text:
        lo = c.lower()
        if ord(c) > 127 and lo != c and len(lo) == 1:
            out += [["type", "--delay", 12, "--", buf]] if buf else []
            out.append(["key", "--clearmodifiers", f"shift+U{ord(lo):04X}"])
            buf = ""
        else:
            buf += c
    return out + [["type", "--delay", 12, "--", buf]] if buf else out


def _killpg(p, sig):
    try:
        os.killpg(p.pid, sig)
    except ProcessLookupError:
        pass


ATSPI_DUMP = r'''
import json, sys
import pyatspi
# keyed by role enum, emitted under canonical names: at-spi2 role *names* vary by version ("push button" vs "button")
ROLES = {getattr(pyatspi, "ROLE_" + n.upper().replace(" ", "_")): n for n in (
    "push button", "toggle button", "check box", "radio button", "menu item", "menu", "page tab", "link", "entry",
    "text", "password text", "combo box", "list item", "table cell", "tree item", "slider", "spin button",
    "icon", "tool bar", "check menu item", "radio menu item", "document web", "heading", "label", "image")}
if hasattr(pyatspi, "ROLE_STATIC"):
    ROLES[pyatspi.ROLE_STATIC] = "label"  # Chromium's static text
out = []
def walk(node, depth):
    if len(out) >= 300 or depth > 40:
        return
    try:
        st = node.getState()
        # depth 0 is the application node: GTK leaves its SHOWING/VISIBLE unset, which pruned every GTK app
        if depth and not (st.contains(pyatspi.STATE_SHOWING) and st.contains(pyatspi.STATE_VISIBLE)):
            return
        role = ROLES.get(node.getRole())
        if role:
            ext = node.queryComponent().getExtents(pyatspi.DESKTOP_COORDS)
            if ext.width > 1 and ext.height > 1:
                out.append({"role": role, "label": (node.name or "")[:100], "x": ext.x, "y": ext.y, "w": ext.width,
                            "h": ext.height, "flags": {"focused": st.contains(pyatspi.STATE_FOCUSED),
                                                       "editable": st.contains(pyatspi.STATE_EDITABLE),
                                                       "checked": st.contains(pyatspi.STATE_CHECKED),
                                                       "enabled": st.contains(pyatspi.STATE_ENABLED)}})
                if st.contains(pyatspi.STATE_FOCUSED) and st.contains(pyatspi.STATE_EDITABLE):
                    try:  # oracle-only (the prompt never shows it): what the focused field holds, e.g. the omnibox
                        out[-1]["value"] = node.queryText().getText(0, -1)[:200]
                    except Exception:
                        pass
        for i in range(min(node.childCount, 500)):
            walk(node.getChildAtIndex(i), depth + 1)
    except Exception:
        return
desk = pyatspi.Registry.getDesktop(0)
for i in range(desk.childCount):
    walk(desk.getChildAtIndex(i), 0)
print(json.dumps(out))
'''


@dataclass
class Observation:
    obs_id: str
    image: str
    geom: Geometry
    app: str | None
    window: str | None
    windows: list = field(default_factory=list)
    apps: list = field(default_factory=list)
    elements: list | None = None
    element_px: list | None = None  # screen boxes behind `elements`, same order


class Desktop:
    def __init__(self, display=":7", size=(1280, 800), home=None, apps=None):
        self.display, self.size = display, size
        self.home = Path(home or tempfile.mkdtemp(prefix="lays-home-"))
        self.apps = apps or []  # [{"id": "mousepad.desktop", "name": "Mousepad", "exec": "mousepad"}]
        self.procs, self.n, self.last, self.history = [], 0, None, []
        self.env = dict(os.environ, DISPLAY=display, HOME=str(self.home), XDG_RUNTIME_DIR=str(self.home / ".run"),
                        GTK_MODULES="gail:atk-bridge", NO_AT_BRIDGE="0", QT_ACCESSIBILITY="1", LANG="C.UTF-8",
                        LC_ALL="C.UTF-8",  # an inherited LC_ALL overrides LANG; xdotool then rejects non-ASCII text
                        SAL_USE_VCLPLUGIN="gtk3")  # LibreOffice's default "gen" backend has no AT-SPI
        for k in ("PYTHONPATH", "PYTHONHOME", "PYTHONSAFEPATH"):
            self.env.pop(k, None)

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        (self.home / ".run").mkdir(parents=True, exist_ok=True, mode=0o700)
        (self.home / "Desktop").mkdir(exist_ok=True)
        w, h = self.size
        self._spawn(["Xvfb", self.display, "-screen", "0", f"{w}x{h}x24", "-nolisten", "tcp"])
        for _ in range(50):
            if subprocess.run(["xdpyinfo"], env=self.env, capture_output=True).returncode == 0:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(f"Xvfb {self.display} did not start")
        out = subprocess.run(["dbus-launch", "--sh-syntax"], env=self.env, capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "=" in line and line.startswith("DBUS_SESSION_BUS"):
                k, v = line.split("=", 1)
                self.env[k] = v.strip().rstrip(";").strip("'")
        self._spawn(["/usr/libexec/at-spi-bus-launcher", "--launch-immediately"])
        self._spawn(["openbox"])
        time.sleep(1.0)
        return self

    def stop(self):
        # each spawn leads its own process group; signal the group, since apps fork (libreoffice -> soffice.bin) and a
        # surviving child carries its windows and dialogs into the next episode
        for p in reversed(self.procs):
            _killpg(p, signal.SIGTERM)
        for p in self.procs:
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                pass
            _killpg(p, signal.SIGKILL)
            p.wait()
        self.procs = []

    def _spawn(self, cmd, **kw):
        p = subprocess.Popen(cmd, env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True, **kw)
        self.procs.append(p)
        return p

    def sh(self, cmd, timeout=30, check=False):
        r = subprocess.run(cmd if isinstance(cmd, list) else ["bash", "-lc", cmd], env=self.env, capture_output=True,
                           text=True, timeout=timeout, cwd=self.home)
        if check and r.returncode:
            raise RuntimeError(f"{cmd}: {r.stderr[-500:]}")
        return r.stdout

    def launch(self, cmd):
        """Task setup: start an app detached from the driver (not an agent action)."""
        return self._spawn(shlex.split(cmd), cwd=self.home)

    # ------------------------------------------------------------------ observation
    def observe(self, out_dir, with_elements=True) -> Observation:
        self.n += 1
        oid = f"o{self.n}"
        img = str(Path(out_dir) / f"{oid}.png")
        self.sh(["import", "-window", "root", img], check=True)
        wins = []
        for line in self.sh(["wmctrl", "-lx"]).splitlines():
            parts = line.split(None, 4)
            if len(parts) == 5 and parts[1] != "-1":
                wins.append({"id": parts[0], "app": parts[2].split(".")[-1].lower(), "title": parts[4][:120]})
        active = self.sh(["xdotool", "getactivewindow"]).strip()
        aw = next((x for x in wins if active and int(x["id"], 16) == int(active)), None)
        els = px = None
        if with_elements:
            try:
                raw = json.loads(subprocess.run(["/usr/bin/python3", "-c", ATSPI_DUMP], env=self.env, capture_output=True,
                                                text=True, timeout=20).stdout or "[]")
            except (subprocess.TimeoutExpired, ValueError):
                raw = []
            W, H = self.size
            els, px = [], []
            for e in raw:
                x0, y0, x1, y1 = max(e["x"], 0), max(e["y"], 0), min(e["x"] + e["w"], W), min(e["y"] + e["h"], H)
                if x1 - x0 < 2 or y1 - y0 < 2:
                    continue
                els.append({"role": e["role"], "label": e["label"], "box": [x0 / W, y0 / H, x1 / W, y1 / H],
                            "flags": {k: v for k, v in e["flags"].items() if v}} | ({"value": e["value"]} if "value" in e else {}))
                px.append((x0, y0, x1, y1))
            els, px = (els[:64], px[:64]) if els else (None, None)
        o = Observation(oid, img, Geometry(0, 0, *self.size), aw["app"] if aw else None, aw["title"] if aw else None,
                        wins, [{"id": a["id"], "name": a["name"]} for a in self.apps], els, px)
        self.last = o
        return o

    # ------------------------------------------------------------------ actions
    def execute(self, a: Action, obs: Observation, out_dir=None) -> dict:
        """Deliver one action decided on `obs`. Returns the history entry (delivery/effect/visual)."""
        try:
            a.validate(set(CAPS))
            if self.last is None or obs.obs_id != self.last.obs_id:
                raise ActionError(f"stale observation {obs.obs_id}")
            self._deliver(a, obs)
            delivery = "delivered"
        except ActionError as e:
            delivery = f"error: {e}"
        entry = {"action": a.short(), "delivery": delivery, "effect": "unknown", "visual": "unknown"}
        if delivery == "delivered" and out_dir and a.family not in ("done", "fail", "ask"):
            time.sleep(0.6)
            nxt = self.observe(out_dir, with_elements=False)
            with Image.open(obs.image) as p, Image.open(nxt.image) as c:
                entry["visual"] = visual_change(p, c)
        self.history.append(entry)
        return entry

    def _point(self, a, obs, p=None):
        if a.route == "element":
            if not obs.element_px or a.element >= len(obs.element_px):
                raise ActionError(f"element {a.element} not in this observation")
            x0, y0, x1, y1 = obs.element_px[a.element]
            return int((x0 + x1) / 2), int((y0 + y1) / 2)
        return to_screen(p or a.point, obs.geom, a.obs_id, obs.obs_id)

    def _x(self, *args):
        r = subprocess.run(["xdotool", *map(str, args)], env=self.env, capture_output=True, text=True, timeout=30)
        if r.returncode:
            raise ActionError(f"xdotool {args[0]} failed: {r.stderr.strip()[:200]}")

    def _deliver(self, a: Action, obs: Observation):
        f = a.family
        if f in ("click", "double_click", "right_click", "hover"):
            x, y = self._point(a, obs)
            self._x("mousemove", "--sync", x, y)
            if f == "click":
                self._x("click", 1)
            elif f == "double_click":
                self._x("click", "--repeat", 2, "--delay", 80, 1)
            elif f == "right_click":
                self._x("click", 3)
        elif f == "drag":
            (x0, y0), (x1, y1) = self._point(a, obs), to_screen(a.dest, obs.geom, a.obs_id, obs.obs_id)
            self._x("mousemove", "--sync", x0, y0)
            self._x("mousedown", 1)
            for k in range(1, 11):
                self._x("mousemove", "--sync", int(x0 + (x1 - x0) * k / 10), int(y0 + (y1 - y0) * k / 10))
            self._x("mouseup", 1)
        elif f == "scroll":
            if a.route != "none":
                x, y = self._point(a, obs)
                self._x("mousemove", "--sync", x, y)
            self._x("click", "--repeat", int(a.scroll[1]), "--delay", 30, WHEEL[a.scroll[0]])
        elif f == "type":
            if a.route != "none":
                x, y = self._point(a, obs)
                self._x("mousemove", "--sync", x, y)
                self._x("click", 1)
            if a.type_mode == "replace":
                self._x("key", "--clearmodifiers", "ctrl+a")
            for run in type_runs(a.text):
                self._x(*run)
        elif f == "key":
            mods = [XMOD[m] for i, m in enumerate(MODS) if a.mods >> i & 1]
            key = [] if a.key == "none" else [XKEY.get(a.key, a.key)]
            self._x("key", "--clearmodifiers", "+".join(mods + key))
        elif f == "wait":
            time.sleep(1.0)
        elif f == "launch_app":
            app = next((x for x in self.apps if x["id"] == a.app_id), None)
            if app is None:
                raise ActionError(f"app {a.app_id!r} not offered in this observation")
            self.launch(app["exec"])
            time.sleep(1.5)
        elif f == "focus_window":
            if not any(w["id"] == a.window_id for w in obs.windows):
                raise ActionError(f"window {a.window_id!r} not in this observation")
            subprocess.run(["wmctrl", "-ia", a.window_id], env=self.env, timeout=10)
        # done / fail / ask: no side effect; the episode runner ends or escalates


def selftest():
    d = Desktop(display=":9").start()
    out = tempfile.mkdtemp(prefix="lays-obs-")
    try:
        d.apps = [{"id": "xterm.desktop", "name": "XTerm", "exec": "xterm"}]
        o = d.observe(out)
        r = d.execute(Action("launch_app", app_id="xterm.desktop"), o, out)
        assert r["delivery"] == "delivered", r
        time.sleep(1)
        o = d.observe(out)
        assert any(w["app"] == "xterm" for w in o.windows), o.windows
        r = d.execute(Action("click", route="pixel", point=(0.5, 0.5), obs_id=o.obs_id), o, out)
        assert r["delivery"] == "delivered", r
        o = d.observe(out)
        target = d.home / "made_by_typing"
        d.execute(Action("type", text=f"touch {target}", text_source="generate", type_mode="insert"), o, out)
        o = d.last
        d.execute(Action("key", key="enter"), o, out)
        time.sleep(0.5)
        assert target.exists(), "typed command did not run"
        stale = d.execute(Action("click", route="pixel", point=(0.1, 0.1), obs_id="o1"), o, out)
        assert stale["delivery"].startswith("error"), stale
        o = d.observe(out)
        for act in (Action("scroll", route="pixel", point=(0.5, 0.5), scroll=("down", 3), obs_id=o.obs_id),
                    Action("drag", route="pixel", point=(0.2, 0.2), dest=(0.3, 0.3), obs_id=o.obs_id),
                    Action("key", key="c", mods=1)):
            o = d.observe(out)
            act.obs_id = act.obs_id and o.obs_id
            assert d.execute(act, o, out)["delivery"] == "delivered", act
        print(json.dumps({"selftest": "ok", "windows": o.windows, "elements": len(o.elements or []),
                          "history": d.history[-3:]}))
    finally:
        d.stop()
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["selftest"]:
        selftest()
