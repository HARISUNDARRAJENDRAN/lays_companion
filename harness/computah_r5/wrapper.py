"""Computah-style harness around the frozen RL3-r5 policy (no weight changes).

r5 still decides what to do from the screenshot. This wrapper grounds and guards that decision with a fresh UIA
snapshot of the owned window:
  - pointer actions snap to the interactive control under / nearest the point (scope: owned window, web document);
  - typed text is replaced by the exact goal substring it was trying to reproduce, and typing into a field
    that already holds text uses replace mode, so the field ends up exactly equal to the copied value;
  - an identical action on an unchanged control is not dispatched twice in a row;
  - a harness completion signal (commit-type control activated + r5 p_done) ends the episode.
Nothing here reads the fixture verifier; evaluation stays independent.
"""
from __future__ import annotations

import difflib
import math
import re
import threading
import time

from schema import Action

POINTER = ("click", "double_click", "right_click", "drag", "scroll", "hover")
INTERACTIVE = {"button", "textfield", "checkbox", "combobox", "radiobutton", "listitem", "menuitem", "tab", "link",
               "slider", "spinbutton", "incrementor", "textarea", "treeitem", "cell", "switch", "option", "menubutton", "splitbutton", "row",
               "text", "image", "listbox", "searchfield", "togglebutton"}
NATIVE_INTERACTIVE = {"edit", "document", "text editor", "text field", "text box", "address bar", "addressbar",
                      "list", "list item", "list items", "item", "items view", "itemsview",
                      "titlebar", "title bar", "system menu", "pane", "menu", "menu bar", "header"}
COMMIT = re.compile(r"\b(save|send|confirm|apply|submit|done|ok|create|reset|export|archive|checkout|place order|"
                    r"restore|merge|invite|schedule|update|add|move|rename|copy|delete|close|finish)\b", re.I)
CONFIRM_INTENT = re.compile(r"\b(confirm|confirmation|place order|checkout|approve|finalize)\b", re.I)
CONFIRM_CONTROL = re.compile(r"\b(confirm|confirmation|place order|approve|finalize|complete order|yes|ok)\b", re.I)
SUCCESS_STATUS = re.compile(r"\b(saved|success(?:ful)?|completed|sent|downloaded|updated)\b", re.I)


def best_goal_span(goal: str, text: str, min_ratio: float = 0.8):
    """Exact substring of goal that best matches generated text, or None. Quoted literals are preferred."""
    if not text or not goal:
        return None
    if text in goal:
        return text
    pool = [a or b for a, b in re.findall(r"'([^']+)'|\"([^\"]+)\"", goal)]
    # Word-aligned windows of similar length as a fallback (e.g. body text, emails, dates).
    starts = [0] + [m.end() for m in re.finditer(r"[\s:,;(\[]", goal)]
    ends = [m.start() for m in re.finditer(r"[\s,;)\]]|\.(?=\s|$)", goal)] + [len(goal)]
    n = len(text)
    for s in starts:
        for e in ends:
            if e > s and 0.7 * n <= e - s <= 1.3 * n + 2:
                pool.append(goal[s:e])
    best, score = None, 0.0
    for cand in pool:
        r = difflib.SequenceMatcher(None, text.lower(), cand.lower()).ratio()
        if r > score or (r == score and best is not None and len(cand) > len(best)):
            best, score = cand, r
    return best if score >= min_ratio else None


ABSTAIN_CLAUSE = re.compile(r"\bif\b[^.]*\b(unavailable|not available|missing|does ?n[o']t exist|not found|cannot be found|can't be found)\b"
                            r"[^.]*\b(report|tell|say|explain|let me know)\b", re.I)
EDITABLE_ROLES = {"textfield", "searchfield", "combobox", "spinbutton", "spinner", "incrementor", "edit", "textarea",
                  "text editor", "document", "address bar", "addressbar", "text field", "text box"}
LIST_TARGET_ROLES = {"option", "listitem", "list item", "list items", "treeitem", "row", "item"}
FRAME_ROLES = {"titlebar", "title bar", "system menu"}
FRAME_NAME = re.compile(r"\b(close|close button|minimi[sz]e|maximi[sz]e|restore|system menu|title ?bar)\b", re.I)
FRAME_INTENT = re.compile(r"\b(close|exit|quit|minimi[sz]e|maximi[sz]e|restore)\b", re.I)
READ_ONLY_STATES = {"readonly", "read-only", "read only", "not editable", "noneditable", "non-editable"}
FIELD_SYNONYMS = {
    "to": {"to", "recipient", "email", "send"}, "recipient": {"to", "recipient", "email"},
    "email": {"email", "to", "address", "mail", "invite"}, "attendee": {"attendee", "invite", "with", "email"},
    "subject": {"subject"}, "title": {"title", "titled", "named", "save", "event", "card", "note", "task", "document", "create"},
    "message": {"body", "message"}, "body": {"body", "message", "exactly", "description"}, "description": {"description", "body"},
    "text": {"body", "text"}, "customer": {"for", "customer", "client"}, "name": {"name", "named", "contact", "called"},
    "company": {"company", "organisation", "organization"}, "reference": {"reference", "ref"},
    "units": {"units", "unit", "quantity", "qty", "least"}, "unit": set(), "quantity": {"quantity", "qty", "units"},
    "price": {"price", "cost"}, "formula": {"formula"}, "total": set(), "minimum": {"least", "minimum"},
    "label": {"label", "labelled", "labeled", "tag"}, "search": {"search", "find", "order", "record", "contact"},
    "date": {"on", "date", "to"}, "start": {"starting", "start", "from"}, "occurrences": {"occurrences"}, "time": {"at", "time"}, "amount": {"amount", "total"}, "find": {"find", "replace"},
    "replacement": {"with", "replacement"}, "filename": {"as", "named", "file", "export", "save"},
    "file": {"file", "as", "named", "save"},
    "copy": {"as"}, "new": {"to", "as"}, "assignee": {"assigned", "assignee", "to"}, "source": {"source", "card"},
}
EMAILISH = {"email", "to", "recipient", "attendee"}
NUMERIC = {"units", "quantity", "amount", "price", "minimum"}


def goal_slots(goal: str):
    """Literal values in the goal, each with nearby cue words: [(value, start, cues)]."""
    spans = []
    for m in re.finditer(r"(?:body|exactly|description|text|with|message)\s*:\s*(.+)$", goal, re.I):
        spans.append((m.start(1), m.end(1)))
    colon_bodies = {sp[0] for sp in spans}
    for m in re.finditer(r"'([^']+)'|\"([^\"]+)\"", goal):
        g = 1 if m.group(1) is not None else 2
        spans.append((m.start(g), m.end(g)))
    for m in re.finditer(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", goal):
        spans.append((m.start(), m.end()))
    for m in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}\b|(?<![\w.@-])\d+(?:\.\d+)?(?![\w.@:-]|\.\d)", goal):
        spans.append((m.start(), m.end()))
    spans.sort(key=lambda se: (se[0], -se[1]))
    kept = []
    for s, e in spans:  # drop literals nested inside a larger literal (e.g. a date inside the body)
        if any(ks <= s and e <= ke and (ks, ke) != (s, e) for ks, ke in kept):
            continue
        kept.append((s, e))
    slots, seen, prev_end = [], set(), 0
    for s, e in kept:
        value = goal[s:e].strip()
        if not value or (s, value) in seen:
            continue
        seen.add((s, value))
        before = re.findall(r"[a-z0-9]+", goal[max(prev_end, s - 30):s].lower())[-3:]
        cues = {w: 1 for w in before}
        if before:
            cues[before[-1]] = 2
        if re.fullmatch(r"\d+(?:\.\d+)?", value):
            for w in re.findall(r"[a-z]+", goal[e:e + 14].lower())[:1]:
                cues[w] = 2
        if s in colon_bodies:
            cues.update({"body": 2, "message": 2, "text": 1, "description": 1})
        prev_end = e
        slots.append((value, s, cues))
    return slots


def value_for_field(goal: str, label: str, with_strength: bool = False):
    """Exact goal literal whose cue words match the field label, if exactly one value matches best."""
    words = set(re.findall(r"[a-z0-9]+", (label or "").lower()))
    if not words:
        return (None, False) if with_strength else None
    wanted = set().union(*(FIELD_SYNONYMS.get(w, {w}) for w in words))
    scored = []
    for value, start, cues in goal_slots(goal):
        score = sum(cues.get(w, 0) for w in wanted)
        if score <= 0:
            continue
        is_email = "@" in value
        is_num = bool(re.fullmatch(r"\d+(?:\.\d+)?", value))
        has_ext = bool(re.search(r"\.[A-Za-z0-9]{2,4}$", value)) and " " not in value[-6:]
        if words & EMAILISH:
            score += 2 if is_email else -1
        elif is_email:
            score -= 2
        if words & NUMERIC:
            score += 2 if is_num else -2
        elif is_num and not words & {"date", "time", "formula"} and not any(re.fullmatch(r"[a-z]\d+", w) for w in words):
            score -= 1
        if words & {"filename", "file"} or words >= {"copy", "name"} or words >= {"new", "name"}:
            score += 2 if has_ext else -2
        elif words & {"title", "subject", "name", "customer", "company"} and has_ext:
            score -= 2
        if score > 0:
            strong = any(cues.get(w) == 2 for w in words)  # the label word itself sits right next to the value
            if words >= {"file", "name"} and any(cues.get(w) == 2 for w in {"as", "save", "named"}):
                strong = True
            scored.append((score, -start, value, strong))
    if not scored:
        return (None, False) if with_strength else None
    scored.sort(reverse=True)
    best = [(v, st) for sc, _, v, st in scored if sc == scored[0][0]]
    if len({v for v, _ in best}) != 1:
        return (None, False) if with_strength else None
    strong = any(st for _, st in best)
    return (best[0][0], strong) if with_strength else best[0][0]


def choose_text(goal: str, generated: str, label: str | None):
    """Exact goal text to type, and why. Field-label grounding first, but r5 keeps a deliberate different copy."""
    g = (generated or "").strip()
    gl = g.lower()
    alpha = lambda x: set(re.findall(r"[a-z]+", x.lower()))
    slots = [v for v, _, _ in goal_slots(goal)]
    mapped, strong = value_for_field(goal, label, with_strength=True) if label else (None, False)
    containing = {v for v in slots if len(gl) >= 3 and gl in v.lower()}
    own = g if g in slots else (next(iter(containing)) if len(containing) == 1 else None)
    if mapped is not None:
        if g == mapped:
            return mapped, "model"
        if strong:
            return mapped, "field_label_strong"
        if own is None or own == mapped or gl in mapped.lower() or (alpha(g) & alpha(mapped)):
            return mapped, "field_label"
    if own is not None:
        return own, "model" if own == g else "fragment"
    sim = best_goal_span(goal, g)
    return (sim, "similarity") if sim else (generated, "model")


def _role(control):
    return re.sub(r"[ _-]+", " ", str(control.get("role") or "").strip().lower())


def _editable(control):
    if not isinstance(control, dict):
        return False
    states = {str(s).strip().lower() for s in control.get("states", [])}
    return (_role(control) in EDITABLE_ROLES and not (states & READ_ONLY_STATES)
            and control.get("editable") is not False and control.get("enabled", True))


def _frame_control(control):
    role = _role(control)
    name = control.get("name") or ""
    return (role in FRAME_ROLES or bool(FRAME_NAME.search(name))
            or (role == "menuitem" and name.strip().casefold() == "system"))


def _frame_intended(goal):
    return bool(FRAME_INTENT.search(goal or ""))


def _is_list_target(control):
    return _role(control) in LIST_TARGET_ROLES


def _is_search(control):
    return _role(control) == "searchfield" or "search" in (control.get("name") or "").lower()


class ComputahR5:
    def __init__(self, policy, backend, snap_radius=40, scope_radius=120, done_threshold=0.30, log_features=None):
        import os
        self.log_features = bool(int(os.environ.get("S1_LOG_FEATURES", "0"))) if log_features is None else log_features
        self.probe, self.max_vetoes, self._vetoes = None, 3, 0
        self.verify_done = os.environ.get("S1_VERIFY_DONE", "1") != "0"
        self.c6_confirm_gate = os.environ.get("S1_C6_CONFIRM_GATE", "1") != "0"
        self.c6_timing = os.environ.get("S1_C6_TIMING", "1") != "0"
        self.c6_cycle_helpers = os.environ.get("S1_C6_CYCLE_HELPERS", "1") != "0"
        self.c6_precommit = os.environ.get("S1_C6_PRECOMMIT", "1") != "0"
        self.c6_target_grounding = os.environ.get("S1_C6_TARGET_GROUNDING", "1") != "0"
        self.c6_native_safety = os.environ.get("S1_C6_NATIVE_SAFETY", "1") != "0"
        self.c6_native_grounding = os.environ.get("S1_C6_NATIVE_GROUNDING", "0") != "0"
        self._done_vetoes = 0
        if os.environ.get("S1_DONE_PROBE"):
            import json as _json, torch
            raw = _json.loads(open(os.environ["S1_DONE_PROBE"], encoding="utf-8").read())
            self.probe = {k: torch.tensor(raw[k]) for k in ("mean", "std", "w", "b")}
            self.probe.update(tau_hi=float(raw["tau_hi"]), tau_lo=float(raw["tau_lo"]))
        self.policy, self.backend = policy, backend
        self.snap_radius, self.scope_radius, self.done_threshold = snap_radius, scope_radius, done_threshold
        self.p_done = 0.0
        self.diagnostics = {}
        self._last = None          # (family, control key, control fingerprint, text)
        self._committed = None     # control name of the last commit-type activation
        self._repeat_blocked = False
        self._field = None         # editable control most recently clicked (focus fallback)
        self._uia_snapshot_ms = 0.0
        self._readback_controls = []
        self._recent_fingerprints = []
        self._cycle_search_submitted = set()
        self._precommit_redirected = set()
        self._confirmation_clicks = 0
        self._confirmation_last = None
        self._confirmation_repeat_blocked = False
        self._snapshot_lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    def _snapshot(self):
        started = time.perf_counter()
        try:
            # Serialize snapshots even when a timed-out backend call is still unwinding.
            with self._snapshot_lock:
                raw = self.backend.uia_snapshot()
            roles = sorted({c["role"] for c in raw})
            readback = [c for c in raw if c["role"].lower() in {"status", "alert", "notification"}]
            native = (getattr(self.backend, "last_window_box", None) is not None
                      and getattr(self.backend, "last_document_box", None) is None)
            allowed = INTERACTIVE | (NATIVE_INTERACTIVE if native and (self.c6_native_safety or self.c6_native_grounding) else set())
            controls = [c for c in raw if c["role"].lower() in allowed or c["role"] == ""]
            return controls, None, roles, readback, (time.perf_counter() - started) * 1000
        except Exception as exc:  # grounding is best-effort; never crash the episode
            return [], repr(exc)[:200], [], [], (time.perf_counter() - started) * 1000

    def _snapshot_during_policy(self, task, observation, history, prev, k):
        """Overlap the single step snapshot with policy inference; timeout means no grounding."""
        result = []
        done = threading.Event()

        def take_snapshot():
            result.append(self._snapshot())
            done.set()

        worker = threading.Thread(target=take_snapshot, name="lays-uia-snapshot", daemon=True)
        worker.start()
        try:
            action = self.policy(task, observation, history, prev, k)
        finally:
            timeout = max(0.05, float(getattr(self.backend, "timeout", 10)) + 0.25)
            worker.join(timeout)
        if not done.is_set():
            if hasattr(self.backend, "last_document_box"):
                self.backend.last_document_box = None
            if hasattr(self.backend, "last_window_box"):
                self.backend.last_window_box = None
            return action, [], f"TimeoutError: UIA snapshot exceeded {timeout:.2f}s", [], [], timeout * 1000
        controls, err, roles, readback, elapsed = result[0]
        if err and hasattr(self.backend, "last_document_box"):
            self.backend.last_document_box = None
        if err and hasattr(self.backend, "last_window_box"):
            self.backend.last_window_box = None
        return action, controls, err, roles, readback, elapsed

    @staticmethod
    def _distance(box, p):
        x, y, w, h = box
        return math.hypot(max(x - p[0], 0, p[0] - (x + w)), max(y - p[1], 0, p[1] - (y + h)))

    def _nearest(self, controls, p, radius):
        ranked = sorted(((self._distance(c["box"], p), c["box"][2] * c["box"][3], c) for c in controls if c["enabled"]),
                        key=lambda z: (z[0], z[1]))
        if ranked and ranked[0][0] <= radius:
            return ranked[0][0], ranked[0][2]
        return None, None

    @staticmethod
    def _key(c):
        return (c["role"], c["name"], tuple(round(v / 8) for v in c["box"]))

    def _visible_control(self, control):
        states = {str(s).lower() for s in control.get("states", [])}
        if states & {"offscreen", "invisible", "hidden"}:
            return False
        page = getattr(self.backend, "last_document_box", None)
        if page is None and getattr(self.backend, "last_window_box", None) is not None:
            page = self.backend.last_window_box
        if page is None:
            return True
        x, y, w, h = control["box"]
        px, py, pw, ph = page
        return x + w > px and x < px + pw and y + h > py and y < py + ph

    # ------------------------------------------------------------------ policy interface
    def __call__(self, task, observation, history, prev, k):
        started = time.perf_counter()
        self._uia_snapshot_ms = 0.0
        try:
            return self._call(task, observation, history, prev, k)
        finally:
            if self.c6_timing and isinstance(self.diagnostics, dict):
                self.diagnostics["uia_snapshot_ms"] = round(self._uia_snapshot_ms, 3)
                self.diagnostics["wrapper_ms"] = round((time.perf_counter() - started) * 1000, 3)

    def _call(self, task, observation, history, prev, k):
        if k == 0:  # one wrapper serves a whole campaign: never carry state into a new episode
            self._last, self._committed, self._repeat_blocked, self._field, self._vetoes = None, None, False, None, 0
            self._done_vetoes = 0
            self._recent_fingerprints = []
            self._cycle_search_submitted = set()
            self._precommit_redirected = set()
            self._confirmation_clicks = 0
            self._confirmation_last = None
            self._confirmation_repeat_blocked = False
        action, controls, err, roles, readback, self._uia_snapshot_ms = self._snapshot_during_policy(
            task, observation, history, prev, k)
        self.p_done = float(self.policy.p_done)
        goal = getattr(task, "goal", "") or ""
        self._readback_controls = readback
        d = {"route": "pixel_snap", "controls": len(controls), "snapped": False, "snap_px": None, "snap_target": None,
             "text_substituted": False, "type_mode_forced": False, "repeat_blocked": False, "completion_signal": False}
        self.diagnostics = d
        if err:
            d["uia_error"] = err
        d["roles_seen"] = roles
        h = getattr(self.policy, "last_decision_h", None)
        if h is not None and self.log_features:
            d["h"] = [round(float(v), 4) for v in h.tolist()]
        d["r5_family"] = action.family
        d["r5_p_done"] = round(self.p_done, 4)
        W, H = observation.geom.width, observation.geom.height
        checked, mismatches = self._verify_fields(goal, controls)
        mismatched_fields = self._strong_field_mismatches(goal, controls)
        d["verify"] = {"checked": checked, "mismatches": mismatches[:4]}
        confirm_target = self._confirmation_target(goal, controls) if self.c6_confirm_gate else None
        if self._confirmation_repeat_blocked and action.family == "done":
            d.update(confirmation_repeat_blocked=True,
                     completion_reason="confirmation remains visible without a new control state")
            return Action(family="wait", route="none", obs_id=observation.obs_id)
        # If r5 stops while an explicitly requested confirmation is still on screen,
        # activate that control. This uses only the goal, current UIA controls and r5 action.
        if confirm_target is not None and action.family == "done" and not mismatches:
            return self._click_confirmation(confirm_target, observation, W, H, d)
        if self.probe is not None and h is not None:
            import torch
            pr = self.probe
            pc = float(torch.sigmoid(((h - pr["mean"]) / pr["std"]) @ pr["w"] + pr["b"]))
            d["p_complete"] = round(pc, 4)
            if action.family == "done" and pc < pr["tau_lo"] and self._vetoes < self.max_vetoes:
                self._vetoes += 1
                d.update(done_vetoed=True, completion_reason=f"probe p_complete={pc:.2f} < {pr['tau_lo']:.2f}")
                return Action(family="wait", route="none", obs_id=observation.obs_id)
            if action.family not in ("done", "fail", "ask") and k > 0 and pc >= pr["tau_hi"]:
                if confirm_target is not None and self._committed and not mismatches:
                    return self._click_confirmation(confirm_target, observation, W, H, d)
                if self._confirmation_repeat_blocked:
                    d.update(confirmation_repeat_blocked=True,
                             completion_reason="confirmation remains visible without a new control state")
                    return Action(family="wait", route="none", obs_id=observation.obs_id)
                d.update(completion_signal=True, completion_reason=f"probe p_complete={pc:.2f} >= {pr['tau_hi']:.2f}")
                return self._remember(Action(family="done", route="none", obs_id=observation.obs_id), None, d)

        prev_commit = self._committed
        # Harness completion after a commit needs positive evidence: visible goal fields all hold their goal values.
        if (self._committed and action.family not in ("done", "fail", "ask") and self.p_done >= self.done_threshold
                and (not self.verify_done or (checked > 0 and not mismatches))):
            if confirm_target is not None:
                return self._click_confirmation(confirm_target, observation, W, H, d)
            if self._confirmation_repeat_blocked:
                d.update(confirmation_repeat_blocked=True,
                         completion_reason="confirmation remains visible without a new control state")
                return Action(family="wait", route="none", obs_id=observation.obs_id)
            d.update(completion_signal=True, completion_reason=f"after '{self._committed}' p_done={self.p_done:.2f}; verified {checked} fields")
            self._committed = None
            return Action(family="done", route="none", obs_id=observation.obs_id)
        # r5's own "done" is vetoed when a visible field contradicts the goal; never claim success we can see is false.
        if action.family == "done" and self.verify_done and mismatches:
            self._done_vetoes += 1
            if self._done_vetoes <= 2:
                d.update(done_vetoed=True, completion_reason=f"done vetoed: {mismatches[0]}")
                return Action(family="wait", route="none", obs_id=observation.obs_id)
            d.update(done_vetoed=True, completion_reason=f"could not verify completion: {mismatches[0]}")
            return Action(family="ask", route="none", obs_id=observation.obs_id)

        target = None
        native = (getattr(self.backend, "last_window_box", None) is not None
                  and getattr(self.backend, "last_document_box", None) is None)
        if action.family in POINTER and action.point and native and self.c6_native_safety:
            window = self.backend.last_window_box
            points = [(action.point[0] * W, action.point[1] * H)]
            if action.family == "drag" and action.dest:
                points.append((action.dest[0] * W, action.dest[1] * H))
            if any(self._distance(window, point) > 0 for point in points):
                d["blocked_outside_window"] = True
                return Action(family="wait", route="none", obs_id=observation.obs_id)
            title_band = max(32.0, min(56.0, window[3] * 0.06))
            if (not controls and not _frame_intended(goal)
                    and window[1] <= points[0][1] < window[1] + title_band):
                d["blocked_frame_control"] = True
                return Action(family="wait", route="none", obs_id=observation.obs_id)
        if action.family in POINTER and action.point and controls:
            p = (action.point[0] * W, action.point[1] * H)
            page = getattr(self.backend, "last_document_box", None)
            window = getattr(self.backend, "last_window_box", None)
            native = window is not None and page is None
            if native and self.c6_native_safety:
                endpoints = [p]
                if action.family == "drag" and action.dest:
                    endpoints.append((action.dest[0] * W, action.dest[1] * H))
                if any(self._distance(window, point) > 0 for point in endpoints):
                    d["blocked_outside_window"] = True
                    return Action(family="wait", route="none", obs_id=observation.obs_id)
                explicit_frame_action = _frame_intended(goal)
                top_band = max(32.0, min(56.0, window[3] * 0.06))
                in_title_band = window[1] <= p[1] < window[1] + top_band
                frame_hits = [c for c in controls if _frame_control(c) and self._distance(c["box"], p) == 0]
                if not explicit_frame_action and (frame_hits or in_title_band):
                    task_controls = [c for c in controls if not _frame_control(c) and self._visible_control(c)]
                    dist, target = self._nearest(task_controls, p, self.snap_radius)
                    if target is None:
                        d["blocked_frame_control"] = True
                        return Action(family="wait", route="none", obs_id=observation.obs_id)
                    d["native_frame_rescue"] = True
                else:
                    dist, target = self._nearest(controls, p, self.snap_radius)
            elif page is not None and self._distance(page, p) > 0:
                # Browser chrome (address bar, tabs, toolbar) is never part of the task: rescue or reobserve.
                dist, target = self._nearest(controls, p, self.scope_radius)
                if target is None:
                    d["blocked_outside_page"] = True
                    return Action(family="wait", route="none", obs_id=observation.obs_id)
                d["rescued_into_page"] = True
            elif not native or not self.c6_native_safety:
                dist, target = self._nearest(controls, p, self.snap_radius)
            if target is None:
                # Outside every control: only rescue clearly out-of-scope points (B0 would hit a scope violation).
                in_scope = any(self._distance(c["box"], p) == 0 for c in controls)
                if not in_scope:
                    dist, target = self._nearest(controls, p, self.scope_radius)
            if (native and self.c6_native_safety and target is not None and _frame_control(target)
                    and not _frame_intended(goal)):
                task_controls = [c for c in controls if not _frame_control(c) and self._visible_control(c)]
                dist, target = self._nearest(task_controls, p, self.snap_radius)
                if target is None:
                    d["blocked_frame_control"] = True
                    return Action(family="wait", route="none", obs_id=observation.obs_id)
                d["native_frame_rescue"] = True
            if target is not None:
                x, y, w, h = target["box"]
                action.point = ((x + w / 2) / W, (y + h / 2) / H)
                d.update(snapped=True, snap_px=round(dist, 1), snap_target=f"{target['role']}:{target['name'][:60]}")
            if action.family == "drag" and action.dest:
                q = (action.dest[0] * W, action.dest[1] * H)
                dest_candidates = controls
                if native and self.c6_native_safety and not _frame_intended(goal):
                    top_band = max(32.0, min(56.0, window[3] * 0.06))
                    hits_frame = (window[1] <= q[1] < window[1] + top_band
                                  or any(_frame_control(c) and self._distance(c["box"], q) == 0 for c in controls))
                    if hits_frame:
                        dest_candidates = [c for c in controls if not _frame_control(c) and self._visible_control(c)]
                        d["native_frame_rescue"] = True
                _, dest = self._nearest(dest_candidates, q, self.snap_radius)
                if dest is not None:
                    x, y, w, h = dest["box"]
                    action.dest = ((x + w / 2) / W, (y + h / 2) / H)
                elif dest_candidates is not controls:
                    d["blocked_frame_control"] = True
                    return Action(family="wait", route="none", obs_id=observation.obs_id)

        # If r5 selected a different visible record while exactly one goal literal is
        # present in the same list, route directly to that matching option.
        if (((self.c6_native_grounding if native else self.c6_target_grounding)) and target is not None
                and _is_list_target(target)
                and action.family in ("click", "double_click")):
            expected = {v.casefold() for v, _, _ in goal_slots(goal) if v.strip()}
            matches = [c for c in controls if c["enabled"] and self._visible_control(c)
                       and _is_list_target(c)
                       and (c.get("name") or "").strip().casefold() in expected]
            if len(matches) == 1 and (target.get("name") or "").strip().casefold() not in expected:
                target = matches[0]
                x, y, w, h = target["box"]
                action.point = ((x + w / 2) / W, (y + h / 2) / H)
                d.update(target_grounded=True, target_grounded_from=d.get("snap_target"),
                         target_grounded_to=f"{target['role']}:{target['name'][:60]}")

        # Apply the same duplicate/cap guard to r5-proposed confirmation clicks,
        # including generic Yes/OK controls that match the confirmation intent.
        if (self.c6_confirm_gate and CONFIRM_INTENT.search(goal) and target is not None
                and action.family in ("click", "double_click")
                and target["role"].lower() in {"button", "menuitem", "splitbutton", "menubutton"}
                and CONFIRM_CONTROL.search(target.get("name") or "")):
            fingerprint = (self._key(target), target.get("value"), target.get("enabled"),
                           tuple(sorted(str(s).lower() for s in target.get("states", []))))
            if (action.family == "double_click" or self._confirmation_clicks >= 2
                    or self._confirmation_last == fingerprint):
                d.update(confirmation_repeat_blocked=True,
                         completion_reason="confirmation control already activated without a new control state")
                return Action(family="wait", route="none", obs_id=observation.obs_id)
            self._confirmation_clicks += 1
            self._confirmation_last = fingerprint

        # Do not commit while a strongly mapped visible goal field is empty or wrong.
        # Each field gets one redirect per episode to avoid a redirect loop.
        if (self.c6_precommit and target is not None and action.family in ("click", "double_click")
                and COMMIT.search(target.get("name") or "") and mismatched_fields):
            field = next((c for c in mismatched_fields if self._key(c) not in self._precommit_redirected), None)
            if field is not None:
                self._precommit_redirected.add(self._key(field))
                x, y, w, h = field["box"]
                redirect = Action(family="click", route="pixel", point=((x + w / 2) / W, (y + h / 2) / H),
                                  obs_id=observation.obs_id)
                d.update(precommit_redirect=f"{field['role']}:{field['name'][:60]}",
                         precommit_blocked=target.get("name", "")[:60])
                return self._remember(redirect, field, d,
                                      ("click", self._key(field), (field.get("value"), field.get("enabled")), None))

        if action.family == "type" and action.text:
            focused = next((c for c in controls if "focused" in [s.lower() for s in c["states"]]), None)
            if focused is None and self._field is not None:
                # UIA did not report focus: fall back to the editable control r5 clicked most recently.
                focused = next((c for c in controls if self._key(c) == self._key(self._field)), self._field)
            if focused is not None:
                d["focused"] = f"{focused['role']}:{focused['name'][:60]}"
            exact, how = choose_text(goal, action.text, focused["name"] if focused is not None else None)
            d["text_choice"] = how
            if exact and exact != action.text:
                d.update(text_substituted=True, substitution=how, generated_text=action.text[:120], copied_text=exact[:120])
                action.text = exact
                if action.text_source == "generate":
                    action.text_source = "copy_goal"
            if action.text in goal:
                start = goal.find(action.text)
                action.span = (start, start + len(action.text))
            if focused is None or not _editable(focused):
                if prev_commit and action.text not in goal and not (self.verify_done and mismatches):
                    if confirm_target is not None:
                        return self._click_confirmation(confirm_target, observation, W, H, d)
                    # r5 just committed and now invents text with nowhere to put it: the form it filled is gone.
                    d.update(completion_signal=True, completion_reason=f"lost after '{prev_commit}': non-goal text, no field")
                    self._committed = None
                    return Action(family="done", route="none", obs_id=observation.obs_id)
                fields = [c for c in controls if _editable(c)
                          and (c.get("value") or "") != action.text and value_for_field(goal, c["name"]) == action.text]
                if len(fields) == 1:
                    # Typing a goal value with no field focused: click the one visible field that this value belongs to.
                    c = fields[0]
                    x, y, w, h = c["box"]
                    d["routed_to_field"] = f"{c['role']}:{c['name'][:60]}"
                    click = Action(family="click", route="pixel", point=((x + w / 2) / W, (y + h / 2) / H), obs_id=observation.obs_id)
                    return self._remember(click, c, d, ("click", self._key(c), (c["value"], c["enabled"]), None))
            if native and self.c6_native_safety and not _editable(focused):
                d["blocked_noneditable_type"] = True
                return Action(family="wait", route="none", obs_id=observation.obs_id)
            if focused is not None:
                current = focused.get("value") or ""
                if current == action.text:
                    d["repeat_blocked"] = True  # field already holds exactly this text: typing again duplicates it
                    return self._remember(Action(family="key", route="none", key="tab", obs_id=observation.obs_id), None, d)
                if current and action.type_mode != "replace" and _editable(focused):
                    action.type_mode = "replace"
                    d["type_mode_forced"] = True

        # Repeat guard: same action on the same unchanged control twice in a row -> reobserve once.
        fingerprint = (action.family, self._key(target) if target else None,
                       (target["value"], target["enabled"]) if target else None, action.text)
        point_fingerprint = (tuple(round(float(v), 3) for v in action.point)
                             if target is None and action.point is not None else None)
        cycle_fingerprint = (*fingerprint, point_fingerprint)
        self._recent_fingerprints.append(cycle_fingerprint)
        self._recent_fingerprints = self._recent_fingerprints[-4:]
        unchanged_tail = len(history) >= 2 and all(
            str(item.get("visual", "")).lower() in {"unchanged", "same"} for item in history[-2:])
        same_action_cycle = len(self._recent_fingerprints) >= 3 and all(
            item == self._recent_fingerprints[-1] for item in self._recent_fingerprints[-3:])
        if self.c6_cycle_helpers and same_action_cycle and unchanged_tail:
            alternative, alternative_target, reason = self._cycle_alternative(goal, controls, observation)
            if alternative is not None:
                d.update(cycle_recovery=reason, repeat_blocked=True)
                if alternative_target is not None:
                    d["cycle_target"] = f"{alternative_target['role']}:{alternative_target['name'][:60]}"
                return self._remember(alternative, alternative_target, d)
        if self._last is not None and fingerprint == self._last and action.family not in ("wait", "done", "fail", "ask"):
            if not self._repeat_blocked:
                self._repeat_blocked = True
                d["repeat_blocked"] = True
                return Action(family="wait", route="none", obs_id=observation.obs_id)
            # Still stuck after a reobserve. Stop only with evidence: every visible goal field already holds its
            # goal value (done), or the user explicitly allowed reporting an unavailable prerequisite (abstain).
            if ABSTAIN_CLAUSE.search(goal):
                d.update(abstained=True, completion_reason="stuck; goal permits reporting an unavailable prerequisite")
                return self._remember(Action(family="fail", route="none", obs_id=observation.obs_id), None, d)
            mapped = [(c, value_for_field(goal, c["name"])) for c in controls
                      if _editable(c) and not _is_search(c)]
            mapped = [(c, v) for c, v in mapped if v is not None]
            if len(mapped) >= 2 and all((c.get("value") or "") == v for c, v in mapped):
                d.update(completion_signal=True, completion_reason=f"stuck; {len(mapped)} goal fields already hold their values")
                return self._remember(Action(family="done", route="none", obs_id=observation.obs_id), None, d)
        else:
            self._repeat_blocked = False
        return self._remember(action, target, d, fingerprint)

    def _cycle_alternative(self, goal, controls, observation):
        """Submit an already-correct search query after its submit action repeats without effect."""
        focused = next((c for c in controls if "focused" in [s.lower() for s in c.get("states", [])]), None)
        candidate = focused if focused and _editable(focused) else None
        if candidate is None and self._field is not None:
            candidate = next((c for c in controls if self._key(c) == self._key(self._field)), None)
        search = candidate if candidate and _is_search(candidate) else None
        if search is not None:
            want = value_for_field(goal, search.get("name", ""))
            submitted = (self._key(search), want)
            if want and submitted not in self._cycle_search_submitted and (search.get("value") or "").strip() == want:
                self._cycle_search_submitted.add(submitted)
                return Action(family="key", route="none", key="enter", obs_id=observation.obs_id), None, "search_enter"
        return None, None, None

    def _confirmation_target(self, goal, controls):
        self._confirmation_repeat_blocked = False
        if not CONFIRM_INTENT.search(goal):
            return None
        statuses = " ".join(str(c.get("name") or "") for c in self._readback_controls)
        if SUCCESS_STATUS.search(statuses):
            return None
        candidates = [c for c in controls if c.get("enabled")
                      and c["role"].lower() in {"button", "menuitem", "splitbutton", "menubutton"}
                      and CONFIRM_CONTROL.search(c.get("name") or "")]
        if len(candidates) != 1:
            return None
        candidate = candidates[0]
        fingerprint = (self._key(candidate), candidate.get("value"), candidate.get("enabled"),
                       tuple(sorted(str(s).lower() for s in candidate.get("states", []))))
        if self._confirmation_clicks >= 2 or self._confirmation_last == fingerprint:
            self._confirmation_repeat_blocked = True
            return None
        return candidate

    def _click_confirmation(self, control, observation, width, height, diagnostics):
        fingerprint = (self._key(control), control.get("value"), control.get("enabled"),
                       tuple(sorted(str(s).lower() for s in control.get("states", []))))
        self._confirmation_clicks += 1
        self._confirmation_last = fingerprint
        x, y, w, h = control["box"]
        click = Action(family="click", route="pixel", point=((x + w / 2) / width, (y + h / 2) / height),
                       obs_id=observation.obs_id)
        diagnostics.update(confirmation_routed=f"{control['role']}:{control['name'][:60]}",
                           completion_signal=False, completion_reason="requested confirmation remains visible")
        return self._remember(click, control, diagnostics,
                              ("click", self._key(control), (control.get("value"), control.get("enabled")), None))

    def _verify_fields(self, goal, controls):
        """Visible editable fields whose label strongly maps to a goal literal: (n checked, [mismatch descriptions])."""
        checked, bad = 0, []
        for c in controls:
            if not _editable(c) or _is_search(c):
                continue
            want, strong = value_for_field(goal, c["name"], with_strength=True)
            if want is None or not strong:
                continue
            checked += 1
            have = c.get("value") or ""
            if have != want:
                bad.append(f"{c['name'][:30]}={have[:30]!r} expected {want[:30]!r}")
        return checked, bad

    def _strong_field_mismatches(self, goal, controls):
        mismatches = []
        for c in controls:
            if (not _editable(c) or _is_search(c) or not self._visible_control(c)):
                continue
            want, strong = value_for_field(goal, c["name"], with_strength=True)
            if want is not None and strong and (c.get("value") or "") != want:
                mismatches.append(c)
        return mismatches

    def _remember(self, action, target, d, fingerprint=None):
        self._last = fingerprint
        if target is not None and action.family in ("click", "double_click"):
            self._field = target if _editable(target) else None
        self._committed = target["name"] if (target is not None and action.family in ("click", "double_click")
                                              and COMMIT.search(target["name"] or "")) else None
        d["commit_armed"] = self._committed
        return action
