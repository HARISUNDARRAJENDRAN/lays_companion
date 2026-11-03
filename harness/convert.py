"""Raw datasets -> canonical LAYS samples (one JSON line per training unit). Nothing is deleted: every raw step is
either a sample with a positive target, a context-only step (target null + reason), or counted under a reason.

Sample = the dict `policy.render` / `objective.terms` consume, plus bookkeeping:
  id, source, group (split key), split, provenance{...}, mask_reason (when target is null)
Units: AgentNet / VideoCUA / GUI-360 = one decision step; grounding sources = one (screenshot, element) query;
oasst1 = one assistant reply in its conversation context.

  python convert.py agentnet RAW_DIR IMAGE_ROOT OUT.jsonl [--exclude-tasks ids.json]
  python convert.py videocua RAW_DIR GROUNDCUA_DIR OUT.jsonl [--limit TASKS_PER_APP]
  python convert.py groundcua GROUNDCUA_DIR OUT.jsonl [VIDEOCUA_RAW_DIR]
  python convert.py jedi JEDI_DATASETS_DIR EXTRACTED_IMAGE_ROOT OUT.jsonl [--limit ROWS_PER_FILE]
  python convert.py gui360 TRAIN_DATA_DIR EXTRACTED_IMAGE_DIR OUT.jsonl [--limit FILES]
  python convert.py osatlas DOMAIN_SPLITED.json IMAGE_DIR OUT.jsonl linux|macos|windows
  python convert.py oasst1 RAW_DIR OUT.jsonl
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from observe import split_of, visual_change
from schema import Action, ActionError, key_names, scroll_bin

LINUX_CAPS = ["click", "double_click", "right_click", "hover", "type", "key", "scroll", "drag", "wait", "done",
              "fail", "ask"]  # recorded datasets: no launch_app / focus_window affordance was offered to the demonstrator
OS_MAP = {"windows": "windows", "macos": "macos", "mac": "macos", "darwin": "macos", "ubuntu": "linux", "linux": "linux"}
STATS = Counter()


# ------------------------------------------------------------------ AgentNet
def pyautogui_calls(code):
    """'pyautogui.click(x=0.1, y=0.2)\\npyautogui.write(message="a")' -> [(name, args, kwargs)] or None."""
    try:
        body = ast.parse(code).body
    except SyntaxError:
        return None
    calls = []
    for n in body:
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)):
            return None
        try:
            calls.append((n.value.func.attr, [ast.literal_eval(a) for a in n.value.args],
                          {k.arg: ast.literal_eval(k.value) for k in n.value.keywords}))
        except ValueError:
            return None
    return calls


def _xy(a, kw):
    x, y = kw.get("x", a[0] if len(a) > 0 else None), kw.get("y", a[1] if len(a) > 1 else None)
    return (x, y) if all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in (x, y)) else None


def agentnet_actions(code, goal):
    """One AgentNet step -> list of (target dict | None, history line, unsupported reason | None), in execution order.
    A step that packs two actions (write + press enter) yields two executed actions; the screenshot only shows the
    state before the first, so only the first can be a target (the caller masks the rest)."""
    calls = pyautogui_calls(code)
    if not calls:
        return [(None, "unparsed: " + code[:80], "unparsed_code")]
    out, cursor = [], None
    for name, a, kw in calls:
        t, why = None, None
        if name == "moveTo":
            cursor = _xy(a, kw)
            # a lone moveTo is a hover decision; one followed by a drag/scroll positions it (handled there)
            continue
        if name in ("click", "doubleClick", "rightClick"):
            p = _xy(a, kw)
            fam = {"click": "click", "doubleClick": "double_click", "rightClick": "right_click"}[name]
            clicks = kw.get("clicks", 1)
            if name == "click" and clicks == 2:
                fam = "double_click"
            if kw.get("button") == "right":
                fam = "right_click"
            if p is None or clicks not in (1, 2):
                why = "bad_click"
            else:
                t = {"family": fam, "route": "pixel", "point": p}
        elif name == "tripleClick":
            why = "triple_click_unsupported"
        elif name in ("write", "typewrite"):
            text = kw.get("message", a[0] if a else None)
            if not isinstance(text, str) or not text:
                why = "empty_text"
            else:
                i = goal.find(text)
                t = {"family": "type", "route": "none", "text": text, "type_mode": "insert",
                     "text_source": "copy_goal" if i >= 0 else "generate"}
                if i >= 0:
                    t["span"] = (i, i + len(text))
        elif name in ("press", "hotkey"):
            names = [k for x in (a or [kw.get("keys")]) for k in (x if isinstance(x, list) else [x])]
            presses = kw.get("presses", 1)
            try:
                if name == "press" and len(names) != 1:
                    raise ActionError("press of several keys")
                key, mods = key_names(names)
                t = {"family": "key", "route": "none", "key": key, "mods": mods}
                if presses != 1:
                    t, why = None, "repeated_press"  # n presses is n actions; no repeat field in lays-action/1
            except ActionError:
                why = "key_outside_closed_set"
        elif name in ("scroll", "hscroll"):
            n = kw.get("clicks", a[0] if a else None)
            if not isinstance(n, (int, float)) or n == 0:
                why = "bad_scroll"
            else:
                d = ("right" if n > 0 else "left") if name == "hscroll" else ("up" if n > 0 else "down")
                p = _xy(a[1:], kw) if (len(a) > 1 or "x" in kw) else cursor
                t = {"family": "scroll", "route": "pixel" if p else "none", "scroll_dir": d, "scroll_bin": scroll_bin(n)}
                if p:
                    t["point"] = p
        elif name == "dragTo":
            p = _xy(a, kw)
            if p is None or cursor is None:
                why = "drag_without_source"
            else:
                t = {"family": "drag", "route": "pixel", "point": cursor, "dest": p}
        elif name == "terminate":
            st = kw.get("status", a[0] if a else "success")
            t = {"family": "done" if st == "success" else "fail", "route": "none"}
        elif name == "wait":
            t = {"family": "wait", "route": "none"}
        else:
            why = f"unknown_call_{name}"
        out.append((t, history_line(t) if t else f"unsupported: {name}", why))
    if not out and cursor is not None:  # moveTo only
        t = {"family": "hover", "route": "pixel", "point": cursor}
        out.append((t, history_line(t), None))
    return out or [(None, "unparsed: " + code[:80], "no_action")]


def history_line(t):
    """Executed action -> the same short line the live harness writes (schema.Action.short)."""
    a = Action(t["family"], route=t.get("route", "none"), point=t.get("point"), dest=t.get("dest"),
               element=t.get("element"), key=t.get("key"), mods=t.get("mods", 0),
               scroll=(t["scroll_dir"], t["scroll_bin"]) if t.get("scroll_dir") else None, text=t.get("text"),
               type_mode=t.get("type_mode"), app_id=t.get("app_id"), window_id=t.get("window_id"))
    return a.short()


AN = {}  # meta/index/exclude, set before the Pool forks so workers inherit them instead of unpickling per task


def agentnet_episode(args):
    d, os_hint = args
    meta, index, exclude = AN["meta"], AN["index"], AN["exclude"]
    STATS.clear()
    goal = d["instruction"].strip()
    tid = d["task_id"]
    m = meta.get(tid, {})
    os_name = OS_MAP.get(str(m.get("system") or os_hint).lower(), "unknown")
    group = f"agentnet:{tid}"
    split = "excluded" if tid in exclude else split_of(group)
    completed = d.get("task_completed")
    traj, samples, history, prev = d["traj"], [], [], None
    for i, s in enumerate(traj):
        img = index.get(s["image"])
        if img is None:
            STATS["agentnet_missing_image_truncates_episode"] += 1
            break  # later screenshots cannot be tied to an unbroken history
        acts = agentnet_actions(s["value"].get("code", ""), goal)
        nxt = traj[i + 1]["value"] if i + 1 < len(traj) else {}
        judged_wrong = nxt.get("last_step_correct") is False or nxt.get("last_step_redundant") is True
        t, _, why = acts[0]
        if t is None:
            reason = why
        elif judged_wrong:
            reason = "machine_judged_incorrect_or_redundant"  # generated judgment may only EXCLUDE, never label
        elif t["family"] == "fail" and d.get("domain") == "infeasible":
            reason = None  # infeasible by construction: giving up is the correct final action
        elif t["family"] in ("done", "fail") and completed is not True:
            reason = "termination_outcome_unverified"
        else:
            reason = None
        sample = {"id": f"agentnet:{tid}:{i}", "source": "agentnet", "group": group, "split": split, "os": os_name,
                  "layout": None, "app": ", ".join(m.get("applications") or []) or None, "window": None,
                  "capabilities": LINUX_CAPS, "goal": goal, "goal_kind": "objective", "image": img, "prev_image": prev,
                  "history": history[-4:], "elements": None,
                  "target": None if reason else t, "mask_reason": reason,
                  "provenance": {"task_id": tid, "step": i, "code": s["value"].get("code", ""),
                                 "task_completed_machine": completed, "extra_actions_in_step": len(acts) - 1,
                                 "domain": d.get("domain") or m.get("domains"), "screen": [m.get("screen_width"), m.get("screen_height")]}}
        samples.append(sample)
        STATS["agentnet_steps"] += 1
        STATS[f"agentnet_mask:{reason}"] += reason is not None
        # history: every executed action of this step, with the visual evidence of what the step changed
        vis = "unknown"
        nimg = index.get(traj[i + 1]["image"]) if i + 1 < len(traj) else None
        if nimg:
            with Image.open(img) as a_, Image.open(nimg) as b_:
                vis = visual_change(a_, b_)
        for k, (_, line, _) in enumerate(acts):
            history.append({"action": line, "delivery": "delivered", "effect": "unknown",
                            "visual": vis if k == len(acts) - 1 else "unknown"})
        prev = img
    return samples, Counter(STATS)


def agentnet(raw_dir, image_root, out, exclude_path=None, workers=16, limit=None):
    raw = Path(raw_dir)
    meta = {}
    for line in open(raw / "meta_data_merged.jsonl"):
        m = json.loads(line)
        meta[m["task_id"]] = m
    index = {p.name: str(p) for p in Path(image_root).rglob("*.png")}
    exclude = set(json.load(open(exclude_path))) if exclude_path else set()
    jobs = []
    for fn in ("agentnet_ubuntu_5k.jsonl", "agentnet_win_mac_18k.jsonl"):
        for n, line in enumerate(open(raw / fn)):
            if limit and n >= limit:  # per file, so a smoke run sees both OS families
                break
            d = json.loads(line)
            jobs.append((d, "ubuntu" if "ubuntu" in fn else "unknown"))
    AN.update(meta=meta, index=index, exclude=exclude)
    _write(out, jobs, agentnet_episode, workers)


# ------------------------------------------------------------------ GroundCUA (locate) + VideoCUA (decisions)
def videocua_groups(raw_dir):
    """groundcua screenshot sha -> the VideoCUA task group that recorded it (so a screen shares its task's split)."""
    import zipfile
    out = {}
    for zp in sorted(Path(raw_dir).rglob("*.zip")):
        z = zipfile.ZipFile(zp)
        for n in z.namelist():
            if n.endswith("action_log.json"):
                d = json.loads(z.read(n))
                for a in d["action_log"]:
                    if a.get("groundcua_id"):
                        out.setdefault(a["groundcua_id"], f"videocua:{d['task_id']}")
    return out


def groundcua(gdir, out, videocua_dir=None):
    """One locate unit per annotated element with non-empty text. Category kept as role metadata; a text label is a
    referring expression, not an instruction to click. Screens recorded by a VideoCUA task take that task's group."""
    gdir = Path(gdir)
    vc = videocua_groups(videocua_dir) if videocua_dir else {}
    with open(out, "w") as f:
        for js in sorted((gdir / "data").rglob("*.json")):
            app = js.parent.name
            img = gdir / "images" / app / (js.stem + ".png")
            if not img.exists():
                STATS["groundcua_missing_image"] += 1
                continue
            with Image.open(img) as im:
                W, H = im.size
            for k, e in enumerate(json.load(open(js))):
                text = (e.get("text") or "").strip()
                x0, y0, x1, y1 = e["bbox"]
                box = [max(x0 / W, 0), max(y0 / H, 0), min(x1 / W, 1), min(y1 / H, 1)]
                if not text or box[2] <= box[0] or box[3] <= box[1]:
                    STATS["groundcua_unusable_element"] += 1
                    continue
                group = vc.get(js.stem, f"groundcua:{js.stem}")
                f.write(json.dumps({"id": f"groundcua:{js.stem}:{k}", "source": "groundcua", "group": group,
                                    "split": split_of(group), "os": "unknown", "app": app, "goal": text[:300],
                                    "goal_kind": "locate", "image": str(img), "target": {"box": box},
                                    "provenance": {"sha": js.stem, "element_id": e.get("id"),
                                                   "category": e.get("category") or None}}) + "\n")
                STATS["groundcua_units"] += 1


# VideoCUA: human demonstrations; each logged action names the GroundCUA screenshot taken at it (the decision frame),
# whose element annotations give the element list. Coordinates are native screen pixels of that frame.
VC = {}  # groundcua stem -> (image path, json path); set before the Pool forks
MAX_ELEMENTS = 64


def _button(p):
    t = str(p.get("text") or "left").strip().lower()
    return "right" if t.startswith("r") else "left" if t.startswith("l") or t in ("click", "na") else None


def _key_text(t):
    """'Down arrow key' / 'arrow down' / 'Page Down' -> closed-set names, else the lowered text."""
    t = str(t or "").strip().lower().replace(" key", "").replace("arrow", "").strip()
    return {"page down": "pagedown", "page up": "pageup", "back space": "backspace"}.get(t, t.replace(" ", ""))


def videocua_actions(log, goal):
    """VideoCUA action_log -> [(target | None in screen px, history line, unsupported reason | None, groundcua_id)].
    One entry per executed action; MOVE_TO only positions the cursor; MOUSE_DOWN..DRAG_TO..MOUSE_UP is one drag
    decided on the MOUSE_DOWN frame; actions while a KEY_DOWN is held are masked (modifier-held clicks are not
    representable), as is SCROLL (logged magnitudes carry no direction)."""
    out, cursor, held, i = [], None, set(), 0
    while i < len(log):
        a = log[i]
        k, p, gid = a["action_type"], a.get("action_params") or {}, a.get("groundcua_id")
        t, why = None, None
        xy = (p["x"], p["y"]) if isinstance(p.get("x"), (int, float)) and isinstance(p.get("y"), (int, float)) else None
        if k == "MOVE_TO":
            cursor = xy or cursor
            i += 1
            continue
        if k == "KEY_DOWN":
            held.add(_key_text(p.get("text")))
            out.append((None, f"unsupported: hold {_key_text(p.get('text'))}", "key_hold", gid))
            i += 1
            continue
        if k == "KEY_UP":
            held.discard(_key_text(p.get("text")))
            i += 1
            continue
        if k == "CLICK":
            n, b = p.get("numClicks", 1), _button(p)
            if xy is None or b is None or n not in (1, 2) or (b == "right" and n == 2):
                why = "click_unsupported"
            else:
                t = {"family": "double_click" if n == 2 else "right_click" if b == "right" else "click",
                     "route": "pixel", "point": xy}
            cursor = xy or cursor
        elif k == "MOUSE_DOWN":
            src, dest, j = xy or cursor, None, i + 1
            while j < len(log) and log[j]["action_type"] in ("DRAG_TO", "MOVE_TO"):
                q = log[j].get("action_params") or {}
                if log[j]["action_type"] == "DRAG_TO" and "x" in q:
                    dest = (q["x"], q["y"])
                j += 1
            if j < len(log) and log[j]["action_type"] == "MOUSE_UP":
                j += 1
            if _button(p) != "left" or src is None:
                why = "mouse_down_unsupported"
            elif dest is None or dest == src:
                t = {"family": "click", "route": "pixel", "point": src}  # press+release in place
            else:
                t = {"family": "drag", "route": "pixel", "point": src, "dest": dest}
            cursor = dest or src
            out.append((None if held else t, history_line(t) if t else "unsupported: mouse_down",
                        "under_held_key" if held and t else why, gid))
            i = j
            continue
        elif k in ("DRAG_TO", "MOUSE_UP"):
            why = "orphan_drag_event"
        elif k == "TYPING":
            text = p.get("text")
            if not isinstance(text, str) or not text:
                why = "empty_text"
            else:
                c = goal.find(text)
                t = {"family": "type", "route": "none", "text": text, "type_mode": "insert",
                     "text_source": "copy_goal" if c >= 0 else "generate"}
                if c >= 0:
                    t["span"] = (c, c + len(text))
        elif k in ("PRESS", "HOTKEY"):
            names = [_key_text(x) for x in str(p.get("text") or "").split("+")] if k == "HOTKEY" else [_key_text(p.get("text"))]
            try:
                key, mods = key_names(names)
                t = {"family": "key", "route": "none", "key": key, "mods": mods}
            except ActionError:
                why = "key_outside_closed_set"
        elif k == "SCROLL":
            why = "scroll_direction_unknown"
        elif k == "AFTER_LAST_ACTION":
            t = {"family": "done", "route": "none"}  # final frame of a completed human demonstration
        else:
            why = f"unknown_type_{k}"
        if t is not None and held:
            t, why = None, "under_held_key"
        out.append((t, history_line(t) if t else f"unsupported: {k.lower()}", why, gid))
        i += 1
    return out


def _elements(js, W, H):
    els = []
    for e in json.load(open(js)):
        x0, y0, x1, y1 = e["bbox"]
        box = [max(x0 / W, 0), max(y0 / H, 0), min(x1 / W, 1), min(y1 / H, 1)]
        if box[2] > box[0] and box[3] > box[1]:
            els.append({"role": (e.get("category") or "element").lower(), "label": (e.get("text") or "").strip()[:100],
                        "box": box})
    els.sort(key=lambda e: (round(e["box"][1], 2), e["box"][0]))  # reading order, as a live tree dump would be
    return els


def _route_element(t, els, prefer=None):
    """Pixel click inside annotated elements -> element route on the smallest containing box (point kept, so the
    trainer can fall back to pixel when it drops the element list). `prefer`: the element the dataset itself names as
    the target, used when it contains the point. Keeps at most MAX_ELEMENTS, target included."""
    x, y = t["point"]
    inside = [k for k, e in enumerate(els) if e["box"][0] <= x <= e["box"][2] and e["box"][1] <= y <= e["box"][3]]
    area = lambda k: (els[k]["box"][2] - els[k]["box"][0]) * (els[k]["box"][3] - els[k]["box"][1])  # noqa: E731
    best = (prefer if prefer in inside else min(inside, key=area)) if inside and t["family"] != "drag" else None
    if len(els) > MAX_ELEMENTS:
        keep = list(range(MAX_ELEMENTS)) if best is None or best < MAX_ELEMENTS else list(range(MAX_ELEMENTS - 1)) + [best]
        best = None if best is None else keep.index(best)
        els = [els[k] for k in keep]
    if best is None:
        return t, els, False
    return dict(t, route="element", element=best), els, True


def videocua_episode(args):
    zpath, member = args
    import zipfile
    STATS.clear()
    d = json.loads(zipfile.ZipFile(zpath).read(member))
    goal, tid = str(d.get("task_instruction") or "").strip(), d["task_id"]
    group = f"videocua:{tid}"
    split = split_of(group)
    acts = videocua_actions(d["action_log"], goal)
    frames = [VC.get(g) if g else None for *_, g in acts]
    samples, history, prev = [], [], None
    for i, (t, line, why, gid) in enumerate(acts):
        fr = frames[i]
        if fr is not None:
            img, js = fr
            with Image.open(img) as im:
                W, H = im.size
            els = _elements(js, W, H) if js else None
            tgt, reason, hit = None, why, False
            if t is not None:
                tgt = json.loads(json.dumps(t))
                for f in ("point", "dest"):
                    if f in tgt:
                        x, y = tgt[f]
                        tgt[f] = (x / W, y / H)
                if any(not (0 <= v <= 1) for f in ("point", "dest") if f in tgt for v in tgt[f]):
                    tgt, reason = None, "point_outside_frame"
                elif els and "point" in tgt:
                    tgt, els, hit = _route_element(tgt, els)
            STATS["videocua_decisions"] += 1
            STATS[f"videocua_mask:{reason}"] += reason is not None
            STATS["videocua_click_in_element"] += hit
            STATS["videocua_pointer_targets"] += bool(tgt and "point" in tgt)
            samples.append({"id": f"videocua:{tid}:{i}", "source": "videocua", "group": group, "split": split,
                            "os": "linux", "layout": None, "app": d.get("platform"), "window": None,
                            "capabilities": LINUX_CAPS, "goal": goal, "goal_kind": "objective", "image": img,
                            "prev_image": prev, "history": history[-4:], "elements": els or None,
                            "target": tgt, "mask_reason": reason,
                            "provenance": {"task_id": tid, "action": i, "groundcua_id": gid}})
            prev = img
        else:
            STATS["videocua_action_without_frame"] += 1
        nxt = next((frames[j] for j in range(i + 1, len(acts)) if frames[j]), None)
        vis = "unknown"
        if fr is not None and nxt is not None:
            with Image.open(fr[0]) as a_, Image.open(nxt[0]) as b_:
                vis = visual_change(a_, b_)
        if why != "orphan_drag_event":
            history.append({"action": line, "delivery": "delivered", "effect": "unknown", "visual": vis})
    return samples, Counter(STATS)


def videocua(raw_dir, groundcua_dir, out, workers=16, limit=None):
    import zipfile
    g = Path(groundcua_dir)
    VC.update({p.stem: (str(p), str(j) if (j := g / "data" / p.parent.name / (p.stem + ".json")).exists() else None)
               for p in (g / "images").rglob("*.png")})
    jobs = []
    for zp in sorted(Path(raw_dir).rglob("*.zip")):
        names = sorted(n for n in zipfile.ZipFile(zp).namelist() if n.endswith("action_log.json"))
        jobs += [(str(zp), n) for n in names[:limit]]
    _write(out, jobs, videocua_episode, workers)


# ------------------------------------------------------------------ GUI-360 (Windows Office decisions)
PWA_KEYS = {"enter": "enter", "return": "enter", "tab": "tab", "esc": "escape", "escape": "escape", "back": "backspace",
            "backspace": "backspace", "bs": "backspace", "bksp": "backspace", "delete": "delete", "del": "delete",
            "home": "home", "end": "end", "pgup": "pageup", "pgdn": "pagedown", "prior": "pageup", "next": "pagedown",
            "up": "up", "down": "down", "left": "left", "right": "right", "space": "space", "insert": "insert",
            "ins": "insert", "capslock": "capslock", "capital": "capslock", "prtsc": "printscreen",
            "snapshot": "printscreen", "apps": "menu"} | {f"f{i}": f"f{i}" for i in range(1, 13)}
PWA_MODS = {"^": "ctrl", "+": "shift", "%": "alt"}
VK_MODS = {"control": "ctrl", "lcontrol": "ctrl", "rcontrol": "ctrl", "shift": "shift", "lshift": "shift",
           "rshift": "shift", "menu": "alt", "lmenu": "alt", "rmenu": "alt", "lwin": "meta", "rwin": "meta"}


def pywinauto_keys(s):
    """pywinauto type_keys string -> [("type", text) | ("key", [names])] in execution order. Raises ValueError(reason)
    for what cannot be labelled faithfully: a bare {VK_CONTROL} is a press-and-release tap in pywinauto, so
    '{VK_CONTROL}c' types 'c' even though the agent meant ctrl+c; grouped modifiers '+(abc)' are not representable."""
    out, mods, held, text, i = [], [], [], "", 0

    def key(name):
        nonlocal text
        if text:
            out.append(("type", text))
            text = ""
        out.append(("key", held + mods + [name]))
        mods.clear()

    while i < len(s):
        c = s[i]
        if c in PWA_MODS:
            mods.append(PWA_MODS[c])
            i += 1
            if s[i:i + 1] == "(":
                raise ValueError("modifier_group")
            continue
        if c in "()":
            raise ValueError("modifier_group")
        if c == "~":
            key("enter")
            i += 1
            continue
        if c != "{":
            if mods or held:
                key(c.lower())
            else:
                text += c
            i += 1
            continue
        j = s.find("}", i + 2)  # from i+2 so "{}}" is the escaped literal brace
        if j < 0:
            raise ValueError("unclosed_brace")
        name, _, arg = s[i + 1:j].partition(" ")
        i = j + 1
        if len(name) == 1 and not arg:  # escaped literal: {+} {^} {%} {~} {(} {)} {{} {}}
            if mods or held:
                key(name.lower())
            else:
                text += name
            continue
        low = name.lower()
        if low.startswith("vk_"):
            low = low[3:]
            if low in VK_MODS:
                if arg == "down":
                    held.append(VK_MODS[low])
                elif arg == "up" and VK_MODS[low] in held:
                    held.remove(VK_MODS[low])
                else:
                    raise ValueError("vk_modifier_tap_ambiguous")
                continue
        if low not in PWA_KEYS or (arg and not arg.isdigit()):
            raise ValueError("key_outside_closed_set")
        for _ in range(int(arg or 1)):
            key(PWA_KEYS[low])
    if mods:
        raise ValueError("dangling_modifier")
    if text:
        out.append(("type", text))
    return out


def gui360_actions(a, goal, W, H):
    """One GUI-360 step's GUI action -> [(target | None, history line, reason | None)] in execution order, points
    normalized by the window screenshot (W, H). Like AgentNet, only the first entry can be the step's target."""
    fn, args = a.get("function") or "", a.get("args") or {}

    def pt(x, y):
        return (x / W, y / H) if isinstance(x, (int, float)) and isinstance(y, (int, float)) else None

    def text_target(text, mode):
        c = goal.find(text)
        t = {"family": "type", "route": "none", "text": text, "type_mode": mode,
             "text_source": "copy_goal" if c >= 0 else "generate"}
        if c >= 0:
            t["span"] = (c, c + len(text))
        return t

    t, why, extra = None, None, []
    if fn == "click":
        p, b = pt(a.get("coordinate_x"), a.get("coordinate_y")), args.get("button", "left")
        if args.get("pressed"):
            why = "click_with_held_key"
        elif p is None or b not in ("left", "right") or (b == "right" and args.get("double")):
            why = "click_unsupported"
        else:
            t = {"family": "right_click" if b == "right" else "double_click" if args.get("double") else "click",
                 "route": "pixel", "point": p}
    elif fn == "drag":
        p, d = pt(args.get("start_x"), args.get("start_y")), pt(args.get("end_x"), args.get("end_y"))
        if p is None or d is None or args.get("button", "left") != "left" or args.get("key_hold"):
            why = "drag_unsupported"
        else:
            t = {"family": "drag", "route": "pixel", "point": p, "dest": d}
    elif fn == "wheel_mouse_input":
        n, p = args.get("wheel_dist"), pt(a.get("coordinate_x"), a.get("coordinate_y"))
        if not isinstance(n, (int, float)) or n == 0:
            why = "bad_scroll"
        else:
            t = {"family": "scroll", "route": "pixel" if p else "none", "scroll_dir": "up" if n > 0 else "down",
                 "scroll_bin": scroll_bin(n)}
            if p:
                t["point"] = p
    elif fn == "type" and isinstance(args.get("keys"), str) and args["keys"]:
        try:
            seq = pywinauto_keys(args["keys"])
        except ValueError as e:
            seq, why = [], str(e)
        acts = []
        for kind, v in seq:
            if kind == "type":
                acts.append((text_target(v, "insert"), None))
            else:
                try:
                    k, m = key_names(v)
                    acts.append(({"family": "key", "route": "none", "key": k, "mods": m}, None))
                except ActionError:
                    acts.append((None, "key_outside_closed_set"))
        if acts:
            (t, why), extra = acts[0], acts[1:]
    elif fn == "type" and isinstance(args.get("text"), str) and args["text"]:
        t = text_target(args["text"], "replace" if args.get("clear_current_text") else "insert")
        p = pt(a.get("coordinate_x"), a.get("coordinate_y"))
        if p:  # typed into a named control: the click that focuses it is part of the action
            t.update(route="pixel", point=p)
    elif fn == "summary":
        return [(None, None, "summary_not_an_action")]  # describes the screen, executes nothing
    else:
        why = f"unsupported_{fn or 'empty'}"
    first = (t, history_line(t) if t else f"unsupported: {fn}", why)
    return [first] + [(x, history_line(x) if x else "unsupported: key", w) for x, w in extra]


def gui360_episode(path):
    """One GUI-360 trajectory file (one JSON line per step) -> samples. The screenshot is the app window, so element
    boxes (UIA controls, screen coords) are shifted by the window origin; the control GUI-360 names as the action's
    target is preferred for element routing."""
    STATS.clear()
    p = Path(path)
    app, cat = p.parts[-4], p.parts[-3]
    steps = [json.loads(line) for line in open(p)]
    root = Path(GUI360["image_root"]) / app / cat
    samples, history, prev = [], [], None
    for i, d in enumerate(steps):
        s, goal, tid = d["step"], d["request"].strip(), d["execution_id"]
        group = f"gui360:{tid}"
        img = root / s["screenshot_clean"]
        if not img.exists():
            STATS["gui360_missing_image_truncates_episode"] += 1
            break
        with Image.open(img) as im:
            W, H = im.size
        ci = s.get("control_infos") or {}
        ox, oy = ((ci.get("application_windows_info") or {}).get("control_rect") or [0, 0])[:2]
        els, prefer = [], None
        for c in ci.get("merged_controls_info") or []:
            r = c.get("control_rect") or []
            if len(r) != 4:
                continue
            box = [max((r[0] - ox) / W, 0), max((r[1] - oy) / H, 0), min((r[2] - ox) / W, 1), min((r[3] - oy) / H, 1)]
            if box[2] > box[0] and box[3] > box[1]:
                els.append({"role": str(c.get("control_type") or "element").lower(),
                            "label": str(c.get("control_text") or "").strip()[:100], "box": box,
                            "_tag": str(c.get("label"))})
        els.sort(key=lambda e: (round(e["box"][1], 2), e["box"][0]))  # reading order, as a live tree dump would be
        a = s.get("action") or {}
        prefer = next((k for k, e in enumerate(els) if e["_tag"] == str(a.get("control_label"))), None) \
            if a.get("control_label") not in (None, "") else None
        for e in els:
            del e["_tag"]
        status, fn = s.get("status"), a.get("function") or ""
        STATS[f"gui360_status:{status}:{'act' if fn else 'empty'}"] += 1
        if a.get("action_type") == "API" and fn:
            acts = [(None, f"unsupported: api {fn}", "api_call_unsupported")]
        elif not fn:
            done = status == "OVERALL_FINISH" and (d.get("evaluation") or {}).get("complete") == "yes"
            acts = [({"family": "done", "route": "none"}, history_line({"family": "done"}), None) if done
                    else (None, None, f"no_action_{status}")]
        else:
            acts = gui360_actions(a, goal, W, H)
        t, line, why = acts[0]
        hit = False
        if t is not None and any(not (0 <= v <= 1) for f in ("point", "dest") if f in t for v in t[f]):
            t, why = None, "point_outside_frame"
        elif t is not None and "point" in t and els:
            t, els, hit = _route_element(t, els, prefer)
        STATS["gui360_steps"] += 1
        STATS[f"gui360_mask:{why}"] += why is not None
        STATS["gui360_click_in_element"] += hit
        STATS["gui360_extra_actions"] += len(acts) - 1
        samples.append({"id": f"gui360:{tid}:{i}", "source": "gui360", "group": group, "split": split_of(group),
                        "os": "windows", "layout": None, "app": app, "window": None, "capabilities": LINUX_CAPS,
                        "goal": goal, "goal_kind": "objective", "image": str(img), "prev_image": prev,
                        "history": history[-4:], "elements": els[:MAX_ELEMENTS] or None,
                        "target": None if why else t, "mask_reason": why,
                        "provenance": {"execution_id": tid, "step": i, "category": cat, "status": status,
                                       "action": {k: a.get(k) for k in ("function", "args", "control_label", "control_test")},
                                       "extra_actions_in_step": len(acts) - 1}})
        nimg = root / steps[i + 1]["step"]["screenshot_clean"] if i + 1 < len(steps) else None
        vis = "unknown"
        if nimg is not None and nimg.exists():
            with Image.open(img) as a_, Image.open(nimg) as b_:
                vis = visual_change(a_, b_)
        lines = [ln for _, ln, _ in acts if ln]
        for k, ln in enumerate(lines):
            history.append({"action": ln, "delivery": "delivered", "effect": "unknown",
                            "visual": vis if k == len(lines) - 1 else "unknown"})
        prev = str(img)
    return samples, Counter(STATS)


GUI360 = {}  # image_root, set before the Pool forks


def gui360(data_dir, image_root, out, workers=16, limit=None):
    GUI360["image_root"] = image_root
    jobs = sorted(str(p) for p in Path(data_dir).rglob("*.jsonl"))
    _write(out, jobs[:limit] if limit else jobs, gui360_episode, workers)


def osatlas(json_path, image_root, out, os_name):
    """OS-Atlas desktop grounding: one locate unit per (screenshot, element instruction); boxes are already
    normalized xyxy. os_name is the domain file's OS (linux / macos / windows)."""
    index = {p.name: str(p) for p in Path(image_root).rglob("*.png")}
    with open(out, "w") as f:
        for r in json.load(open(json_path)):
            img = index.get(r["img_filename"])
            if img is None:
                STATS["osatlas_missing_image"] += 1
                continue
            group = f"osatlas:{os_name}:{r['img_filename']}"
            for k, e in enumerate(r.get("elements") or []):
                text, box = (e.get("instruction") or "").strip(), e.get("bbox") or []
                if not text or len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
                    STATS["osatlas_unusable_element"] += 1
                    continue
                f.write(json.dumps({"id": f"{group}:{k}", "source": "osatlas", "group": group, "split": split_of(group),
                                    "os": os_name, "app": None, "goal": text[:300], "goal_kind": "locate", "image": img,
                                    "target": {"box": box}, "provenance": {"data_type": e.get("data_type")}},
                                   ensure_ascii=False) + "\n")
                STATS[f"osatlas_units:{os_name}"] += 1


# Jedi: Qwen2.5-VL-style tool calls on synthetic/rendered screens. Coordinates live in the Qwen2.5-VL smart_resize
# frame (each side rounded to /28), not original pixels: 1920x1080 screens have targets up to (1921, 1083), i.e. a
# 1932x1092 frame. Larger screens were also downscaled by an unknown max_pixels; every value consistent with the data
# (2.10M-2.16M) leaves the frame height ambiguous (1064 vs 1092 for 3840x2160), so those rows are quarantined, not
# guessed. Jsonl -> extracted image subset:
JEDI = {"component/generated/component_v1_130k/component_v1_130k.jsonl": "component_v1_130k",
        "component/generated/component_final_1.5m_cleaned_split/component_final_1.5m_cleaned_split.jsonl": "final_1.5m",
        "component/generated/component_library_snap_icon_data_grounding/component_library_snap_icon_data_grounding_conversations.jsonl": "component_library_snap_icon_data",
        "icon_grounding/icon_v0222_grounding/icon_v0222_grounding_conversations.jsonl": "icons_v0222",
        "icon_grounding/training_data_icon_grounded_merged/training_data_icon_conversations-images_grounded_merged.jsonl": "icons_v0122",
        "component/rule-based/ethercalc_v1/ethercalc_v1.jsonl": "ethercalc_data",
        "component/rule-based/slide_v1_17k/slide_v1_17k.jsonl": "slide_v1_17k",
        "component/rule-based/doc_data_new/doc_data_new.jsonl": "doc_images",
        "component/rule-based/doc_scroll_data_new/doc_scroll_data_new.jsonl": "doc_scroll"}
JEDI_PREFIX = "Please generate the next move according to the UI screenshot and instruction.\n\nInstruction: "
POINT_ACTIONS = ("left_click", "double_click", "right_click", "mouse_move")


JEDI_MAX_UNSCALED = 2_099_606  # below the smallest consistent max_pixels: the frame is exactly round(side/28)*28


def _jedi_goal(text):
    g = text.replace("<image>", "").strip()
    for p in (JEDI_PREFIX, "Instruction: "):
        g = g[len(p):] if g.startswith(p) else g
    return g.strip()


def jedi_unit(args):
    """One Jedi row (one screenshot, one or more instruction -> tool-call turns) -> one unit per turn."""
    line, subset, root = args
    STATS.clear()
    d = json.loads(line)
    img = Path(root) / subset / d["image"]
    if not img.exists():
        img = Path(root) / subset / Path(d["image"]).name
    if not img.exists():
        STATS["jedi_missing_image"] += 1
        return [], Counter(STATS)
    with Image.open(img) as im:
        W, H = im.size
    if W * H > JEDI_MAX_UNSCALED:
        STATS["jedi_quarantine_resized_frame"] += 1
        return [], Counter(STATS)
    FW, FH = round(W / 28) * 28, round(H / 28) * 28
    conv = [m for m in d["conversations"] if m["from"] != "system"]
    out = []
    for turn, (hm, gm) in enumerate(zip(conv[0::2], conv[1::2])):
        if hm["from"] != "human" or gm["from"] != "gpt":
            STATS["jedi_unparsed"] += 1
            break
        goal, calls = _jedi_goal(hm["value"]), []
        for c in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", gm["value"], re.S):
            try:
                calls.append(json.loads(c)["arguments"])
            except (ValueError, KeyError):
                calls = None
                break
        if not calls or not goal:
            STATS["jedi_unparsed"] += 1
            continue
        pts = [(c["coordinate"][0] / FW, c["coordinate"][1] / FH) if isinstance(c.get("coordinate"), list) and len(c["coordinate"]) == 2
               else None for c in calls]
        if any(p is None or not (0 <= p[0] <= 1 and 0 <= p[1] <= 1) for p in pts):
            STATS["jedi_bad_coordinate"] += 1
            continue
        acts = [c.get("action") for c in calls]
        group = f"jedi:{subset}:{d['image']}"
        base = {"id": f"jedi:{subset}:{d.get('image_id', d['image'])}:{turn}", "source": "jedi", "group": group,
                "split": split_of(group), "os": "unknown", "app": None, "image": str(img), "goal": goal[:400],
                "provenance": {"subset": subset, "actions": acts, "frame": [FW, FH], "turn": turn}}
        if len(calls) == 1 and acts[0] in POINT_ACTIONS:
            out.append(dict(base, goal_kind="locate", target={"point": pts[0]}))  # referring expression -> where
        elif acts == ["left_click", "left_click_drag"]:
            out.append(dict(base, goal_kind="objective", capabilities=LINUX_CAPS, history=[], elements=None, window=None,
                            layout=None, target={"family": "drag", "route": "pixel", "point": pts[0], "dest": pts[1]}))
        else:
            STATS[f"jedi_unsupported:{'+'.join(map(str, acts))[:40]}"] += 1
            continue
        STATS[f"jedi_units:{subset}"] += 1
    return out, Counter(STATS)


def jedi(datasets_dir, image_root, out, workers=16, limit=None):
    jobs = []
    for rel, subset in JEDI.items():
        with open(Path(datasets_dir) / rel) as f:
            for n, line in enumerate(f):
                if limit and n >= limit:
                    break
                jobs.append((line, subset, image_root))
    _write(out, jobs, jedi_unit, workers)


# ------------------------------------------------------------------ oasst1 (instruction retention)
def oasst1(raw_dir, out, lang="en"):
    """Human-written assistant replies: best-ranked (rank 0), not deleted, review passed, not synthetic, as one
    retention unit with its full ancestor conversation."""
    import pyarrow.parquet as pq
    rows = []
    for p in sorted(Path(raw_dir).rglob("*.parquet")):
        rows += pq.read_table(p).to_pylist()
    by_id = {r["message_id"]: r for r in rows}
    with open(out, "w") as f:
        for r in rows:
            if r["role"] != "assistant" or r.get("rank") != 0 or r.get("deleted") or not r.get("review_result") \
                    or r.get("synthetic") or r.get("lang") != lang:
                continue
            chain, cur = [], r
            while cur is not None:
                chain.append({"role": "assistant" if cur["role"] == "assistant" else "user", "content": cur["text"]})
                cur = by_id.get(cur["parent_id"])
            chain.reverse()
            group = f"oasst1:{chain and r['message_tree_id']}"
            f.write(json.dumps({"id": f"oasst1:{r['message_id']}", "source": "oasst1", "group": group,
                                "split": split_of(group), "goal_kind": "chat", "chat": chain,
                                "provenance": {"message_id": r["message_id"], "tree": r["message_tree_id"]}}) + "\n")
            STATS["oasst1_units"] += 1


# ------------------------------------------------------------------ io
def _write(out, jobs, fn, workers):
    with open(out, "w") as f, Pool(workers) as pool:
        for samples, stats in pool.imap(fn, jobs, chunksize=8):
            STATS.update(stats)
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "agentnet":
        ex = a[a.index("--exclude-tasks") + 1] if "--exclude-tasks" in a else None
        lim = int(a[a.index("--limit") + 1]) if "--limit" in a else None
        agentnet(a[0], a[1], a[2], ex, limit=lim)
    elif cmd == "videocua":
        lim = int(a[a.index("--limit") + 1]) if "--limit" in a else None
        videocua(a[0], a[1], a[2], limit=lim)
    elif cmd == "jedi":
        lim = int(a[a.index("--limit") + 1]) if "--limit" in a else None
        jedi(a[0], a[1], a[2], limit=lim)
    elif cmd == "gui360":
        lim = int(a[a.index("--limit") + 1]) if "--limit" in a else None
        gui360(a[0], a[1], a[2], limit=lim)
    elif cmd == "osatlas":
        osatlas(a[0], a[1], a[2], a[3])
    elif cmd == "groundcua":
        groundcua(a[0], a[1], a[2] if len(a) > 2 else None)
    elif cmd == "oasst1":
        oasst1(a[0], a[1])
    print(json.dumps(STATS))
