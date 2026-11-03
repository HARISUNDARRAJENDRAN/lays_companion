"""Windows foreground identity, emergency stop, and clipboard-free Unicode input."""
import ctypes
import sys
from functools import lru_cache
from ctypes import wintypes as W


@lru_cache(maxsize=1)
def user32():
    if sys.platform != "win32":
        raise RuntimeError("This input backend requires Windows")
    return ctypes.WinDLL("user32", use_last_error=True)


def foreground():
    api = user32()
    api.GetForegroundWindow.restype = W.HWND
    return api.GetForegroundWindow()


def emergency_stop():
    api = user32()
    return bool(api.GetAsyncKeyState(0x77) & 0x8000)  # F8


class ExecutionLock:
    """One physical-input loop per interactive Windows session."""
    def __init__(self):
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.CreateMutexW.argtypes = [ctypes.c_void_p, W.BOOL, W.LPCWSTR]
        api.CreateMutexW.restype = W.HANDLE
        api.CloseHandle.argtypes = [W.HANDLE]
        api.ReleaseMutex.argtypes = [W.HANDLE]
        handle = api.CreateMutexW(None, True, "Local\\LAYS-desktop-execution")
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if ctypes.get_last_error() == 183:
            api.CloseHandle(handle)
            raise RuntimeError("Another LAYS execute loop is running in this Windows session")
        self.api, self.handle = api, handle

    def close(self):
        if self.handle:
            self.api.ReleaseMutex(self.handle)
            self.api.CloseHandle(self.handle)
            self.handle = None


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", W.LONG), ("dy", W.LONG), ("mouseData", W.DWORD),
                ("dwFlags", W.DWORD), ("time", W.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", W.WORD), ("wScan", W.WORD), ("dwFlags", W.DWORD),
                ("time", W.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", W.DWORD), ("wParamL", W.WORD), ("wParamH", W.WORD)]


class INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", W.DWORD), ("data", INPUTUNION)]


SPECIAL_KEYS = {"insert": 0x2D, "printscreen": 0x2C, "capslock": 0x14, "menu": 0x5D}


def special_key_events(key, mods):
    modifiers = [vk for i, vk in enumerate((0x11, 0x12, 0x10, 0x5B)) if mods >> i & 1]
    sequence = [(vk, 0) for vk in modifiers] + [(SPECIAL_KEYS[key], 0), (SPECIAL_KEYS[key], 2)]
    sequence += [(vk, 2) for vk in reversed(modifiers)]
    return [INPUT(1, INPUTUNION(ki=KEYBDINPUT(vk, 0, flags, 0, 0))) for vk, flags in sequence]


def press_special(key, mods, stopped):
    if stopped():
        raise KeyboardInterrupt
    api = user32()
    api.SendInput.argtypes = [W.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    api.SendInput.restype = W.UINT
    events = special_key_events(key, mods)
    block = (INPUT * len(events))(*events)
    try:
        if api.SendInput(len(events), block, ctypes.sizeof(INPUT)) != len(events):
            raise RuntimeError("Special key delivery may be partial")
    finally:
        # A partial batch must not leave a modifier or the primary key held.
        releases = [event for event in events if event.data.ki.dwFlags & 2]
        for event in releases:
            if api.GetAsyncKeyState(event.data.ki.wVk) & 0x8000:
                api.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT))
    return {"disposition": {"delivery": "delivered_unverified", "retry": "unsafe"}, "mechanism": "win32_key"}


def text_events(text):
    """Paired down/up events, including UTF-16 surrogate pairs and literal newlines."""
    out = []
    for c in text.replace("\r\n", "\n").replace("\r", "\n"):
        if c in ("\n", "\t"):
            vk = 0x0D if c == "\n" else 0x09
            out.extend([INPUT(1, INPUTUNION(ki=KEYBDINPUT(vk, 0, flags, 0, 0))) for flags in (0, 2)])
        else:
            data = c.encode("utf-16-le")
            for i in range(0, len(data), 2):
                unit = int.from_bytes(data[i:i + 2], "little")
                out.extend([INPUT(1, INPUTUNION(ki=KEYBDINPUT(0, unit, flags, 0, 0))) for flags in (4, 6)])
    return out


def type_unicode(text, stopped, expected_window):
    api = user32()
    api.SendInput.argtypes = [W.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    api.SendInput.restype = W.UINT
    if any(api.GetAsyncKeyState(vk) & 0x8000 for vk in (0x10, 0x11, 0x12, 0x5B, 0x5C)):
        raise RuntimeError("Release held modifiers before typing; no text sent")
    # Send one character per call so focus and the stop switch are checked frequently.
    for c in text.replace("\r\n", "\n").replace("\r", "\n"):
        if stopped() or foreground() != expected_window:
            raise RuntimeError("Typing interrupted: stop requested or foreground window changed; delivery may be partial")
        events = text_events(c)
        block = (INPUT * len(events))(*events)
        if api.SendInput(len(events), block, ctypes.sizeof(INPUT)) != len(events):
            raise RuntimeError("SendInput did not accept all text events; delivery may be partial (check elevation)")
    return {"disposition": {"delivery": "delivered_unverified", "retry": "unsafe"},
            "mechanism": "win32_unicode", "characters": len(text)}
