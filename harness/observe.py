"""Observation helpers shared by data conversion and the live harness (so offline and live evidence agree)."""
import hashlib

import numpy as np
from PIL import Image


def visual_change(prev: Image.Image, cur: Image.Image, size=(640, 360), level=12, count=20) -> str:
    """'changed' when > count downscaled grey pixels moved by > level, else 'unchanged'. Evidence only: a blink is
    not progress and a still screen is not proof of failure (run-7 lays_h_data effect_code, same constants)."""
    a, b = (np.asarray(im.convert("L").resize(size, Image.BILINEAR), dtype=np.int16) for im in (prev, cur))
    return "changed" if int((np.abs(a - b) > level).sum()) > count else "unchanged"


def split_of(group: str, val=0.02, test=0.02) -> str:
    """Deterministic split by GROUP key (episode/task/recording/document), never by row."""
    h = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test else "val" if h < test + val else "train"


def image_sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
