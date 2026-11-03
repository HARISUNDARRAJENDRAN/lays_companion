"""LAYS action/observation contract, version lays-action/1. Shared by data converters, the policy heads, the driver
and the evaluators: one mapping, so training, serving and scoring cannot drift apart.

Points are normalized (x, y) in [0, 1] over the screenshot of the observation that produced them; the driver turns
them into screen pixels with that observation's capture geometry (`to_screen`) and refuses a stale binding.
`meta` is the physical Super/Windows/Command key; the driver never swaps ctrl <-> meta (no logical remapping).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

SCHEMA = "lays-action/1"
FAMILIES = ("click", "double_click", "right_click", "hover", "type", "key", "scroll", "drag", "wait", "done", "fail",
            "ask", "launch_app", "focus_window")
POINTED = {"click", "double_click", "right_click", "hover", "drag"}
CONTROL = {"wait", "done", "fail", "ask"}
ROUTES = ("none", "pixel", "element")
LEGAL_ROUTES = {f: ("pixel", "element") for f in ("click", "double_click", "right_click", "hover")}
LEGAL_ROUTES |= {"type": ROUTES, "scroll": ROUTES, "drag": ("pixel",), "key": ("none",)}
LEGAL_ROUTES |= {f: ("none",) for f in ("wait", "done", "fail", "ask", "launch_app", "focus_window")}
MODS = ("ctrl", "alt", "shift", "meta")  # bit i of the 16-class mask
KEYS = (("none",)  # a bare modifier tap (e.g. Super alone opens a launcher): key none + non-zero mask
        + tuple("abcdefghijklmnopqrstuvwxyz") + tuple("0123456789") + tuple(f"f{i}" for i in range(1, 13))
        + ("enter", "tab", "escape", "backspace", "delete", "space", "up", "down", "left", "right", "home", "end",
           "pageup", "pagedown", "insert", "printscreen", "capslock", "menu", "minus", "equal", "bracketleft",
           "bracketright", "backslash", "semicolon", "quote", "grave", "comma", "period", "slash"))
SCROLL_DIRS = ("up", "down", "left", "right")
SCROLL_BINS = (1, 2, 3, 5, 10, 20)  # wheel clicks; data rounds to the nearest bin, the raw value stays in provenance
TEXT_SOURCES = ("copy_goal", "copy_observed", "generate", "ask")
TYPE_MODES = ("insert", "replace")
EFFECTS = ("confirmed", "confirmed_no_effect", "partial", "unknown", "error")
OSES = ("linux", "windows", "macos", "unknown")
KEY_ALIAS = {"return": "enter", "esc": "escape", "del": "delete", "pgdn": "pagedown", "pgup": "pageup", " ": "space",
             "arrowup": "up", "arrowdown": "down", "arrowleft": "left", "arrowright": "right", "prtsc": "printscreen",
             "-": "minus", "=": "equal", "[": "bracketleft", "]": "bracketright", "\\": "backslash", ";": "semicolon",
             "'": "quote", "`": "grave", ",": "comma", ".": "period", "/": "slash", "backquote": "grave"}
MOD_ALIAS = {"ctrl": "ctrl", "control": "ctrl", "ctrlleft": "ctrl", "ctrlright": "ctrl", "alt": "alt", "option": "alt",
             "altleft": "alt", "altright": "alt", "shift": "shift", "shiftleft": "shift", "shiftright": "shift",
             "command": "meta", "cmd": "meta", "win": "meta", "super": "meta", "winleft": "meta", "winright": "meta",
             "meta": "meta"}


class ActionError(ValueError):
    """Invalid or unsupported action: reported to the harness as an error, never turned into an implicit wait."""


@dataclass
class Action:
    family: str
    route: str = "none"
    point: tuple[float, float] | None = None       # pixel route target (drag: source)
    dest: tuple[float, float] | None = None        # drag destination
    element: int | None = None                     # element route: index into the observation's element list
    key: str | None = None
    mods: int = 0                                  # 4-bit mask over MODS
    scroll: tuple[str, int] | None = None          # (direction, clicks)
    text: str | None = None                        # exact payload (type)
    text_source: str | None = None
    span: tuple[int, int] | None = None            # [start, end) char offsets into goal / observed text
    type_mode: str | None = None                   # insert | replace
    app_id: str | None = None
    window_id: str | None = None
    obs_id: str | None = None                      # observation the targets are bound to
    schema: str = field(default=SCHEMA)

    def validate(self, capabilities: set[str] | None = None) -> "Action":
        f = self.family
        if self.schema != SCHEMA:
            raise ActionError(f"schema {self.schema!r} != {SCHEMA}")
        if f not in FAMILIES:
            raise ActionError(f"unknown family {f!r}")
        if capabilities is not None and f not in capabilities:
            raise ActionError(f"{f} not in this driver's capabilities")
        if self.route not in LEGAL_ROUTES[f]:
            raise ActionError(f"route {self.route!r} illegal for {f}")
        if self.route == "pixel" and not _unit(self.point):
            raise ActionError("pixel route needs a point in [0,1]^2")
        if self.route == "element" and (not isinstance(self.element, int) or self.element < 0):
            raise ActionError("element route needs an element index")
        if self.route != "none" and self.obs_id is None:
            raise ActionError("targeted action is not bound to an observation")
        if f == "drag" and not _unit(self.dest):
            raise ActionError("drag needs a destination point")
        if f == "key":
            if self.key not in KEYS or not 0 <= self.mods < 16 or (self.key == "none" and not self.mods):
                raise ActionError(f"bad key {self.key!r} mods {self.mods}")
        if f == "scroll" and (not self.scroll or self.scroll[0] not in SCROLL_DIRS or int(self.scroll[1]) < 1):
            raise ActionError(f"bad scroll {self.scroll!r}")
        if f == "type":
            if not isinstance(self.text, str) or self.text == "":
                raise ActionError("type needs non-empty text (no silent empty payload)")
            if self.text_source not in TEXT_SOURCES[:3] or self.type_mode not in TYPE_MODES:
                raise ActionError(f"bad text source/mode {self.text_source}/{self.type_mode}")
        if f == "launch_app" and not self.app_id:
            raise ActionError("launch_app needs an observed app id")
        if f == "focus_window" and not self.window_id:
            raise ActionError("focus_window needs an observed window id")
        return self

    def to_json(self) -> str:
        return json.dumps({k: v for k, v in asdict(self).items() if v is not None and not (k == "mods" and v == 0)},
                          ensure_ascii=False, sort_keys=True)

    @staticmethod
    def from_json(s: str) -> "Action":
        try:
            d = json.loads(s)
            for k in ("point", "dest", "scroll", "span"):
                if d.get(k) is not None:
                    d[k] = tuple(d[k])
            return Action(**d).validate()
        except (TypeError, json.JSONDecodeError) as e:
            raise ActionError(f"unparseable action: {e}") from e

    def short(self) -> str:
        """History line for the prompt: what was executed, never a free-form description."""
        f, bits = self.family, [self.family]
        if self.route == "pixel":
            bits.append(f"({self.point[0]:.3f},{self.point[1]:.3f})")
        elif self.route == "element":
            bits.append(f"element#{self.element}")
        if f == "drag":
            bits.append(f"->({self.dest[0]:.3f},{self.dest[1]:.3f})")
        if f == "key":
            bits.append("+".join([m for i, m in enumerate(MODS) if self.mods >> i & 1] + ([] if self.key == "none" else [self.key])))
        if f == "scroll":
            bits.append(f"{self.scroll[0]} {self.scroll[1]}")
        if f == "type":
            bits.append(f"{self.type_mode} {json.dumps(self.text[:80], ensure_ascii=False)}")
        if f == "launch_app":
            bits.append(self.app_id)
        if f == "focus_window":
            bits.append(self.window_id)
        return " ".join(bits)


def _unit(p) -> bool:
    return (isinstance(p, (tuple, list)) and len(p) == 2
            and all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in p))


def key_names(names) -> tuple[str, int]:
    """['ctrl', 'Shift', 't'] -> ('t', mask). Raises ActionError outside the closed key set."""
    names = [str(n).lower() for n in names]
    mods = 0
    for n in names:
        if n in MOD_ALIAS:
            mods |= 1 << MODS.index(MOD_ALIAS[n])
    rest = [KEY_ALIAS.get(n, n) for n in names if n not in MOD_ALIAS]
    if not rest and mods:
        return "none", mods
    if len(rest) != 1 or rest[0] not in KEYS:
        raise ActionError(f"key chord {names} outside the closed key set")
    return rest[0], mods


def scroll_bin(clicks: int) -> int:
    return min(SCROLL_BINS, key=lambda b: (abs(b - abs(int(clicks))), b))


@dataclass
class Geometry:
    """Where an observation's screenshot sits on the logical screen: origin + size in logical px, plus the scale
    from logical to physical pixels (HiDPI). The screenshot may have been resized; normalized points do not care."""
    x: float
    y: float
    width: float
    height: float
    scale: float = 1.0


def to_screen(point, geom: Geometry, obs_id: str, current_obs_id: str) -> tuple[int, int]:
    """Normalized point of observation `obs_id` -> integer logical screen coordinates. A point from an older
    observation is refused: the screen may have moved under it."""
    if obs_id != current_obs_id:
        raise ActionError(f"stale target: bound to {obs_id}, current observation is {current_obs_id}")
    if not _unit(point):
        raise ActionError(f"point {point} outside the screenshot")
    x = geom.x + min(point[0] * geom.width, geom.width - 1)
    y = geom.y + min(point[1] * geom.height, geom.height - 1)
    return int(x), int(y)
