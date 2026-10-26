"""Debug visualisation helpers."""

from __future__ import annotations

import cv2
import numpy as np

from pulsecheck.regions import CONTROL

# BGR, one distinct color per region; the control region is grey and dashed.
_COLORS: dict[str, tuple[int, int, int]] = {
    "forehead_left": (255, 128, 0),
    "forehead_center": (0, 200, 0),
    "forehead_right": (0, 215, 255),
    "cheek_left": (255, 0, 255),
    "cheek_right": (0, 0, 255),
    "nose": (255, 255, 0),
    "chin": (0, 128, 255),
    CONTROL: (160, 160, 160),
}


def _dashed_polygon(img: np.ndarray, poly: np.ndarray, color, dash: int = 6) -> None:
    n = len(poly)
    for i in range(n):
        p, q = poly[i].astype(float), poly[(i + 1) % n].astype(float)
        length = float(np.hypot(*(q - p)))
        steps = max(int(length // dash), 1)
        for s in range(0, steps, 2):
            a = p + (q - p) * (s / steps)
            b = p + (q - p) * (min(s + 1, steps) / steps)
            cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), color, 1)


def draw_regions(
    frame_bgr: np.ndarray, polygons: dict[str, np.ndarray], labels: bool = True
) -> np.ndarray:
    """Return a copy of the frame with every region outlined (control dashed grey)."""
    img = frame_bgr.copy()
    for name, poly in polygons.items():
        if len(poly) < 3:
            continue
        color = _COLORS.get(name, (255, 255, 255))
        if name == CONTROL:
            _dashed_polygon(img, poly, color)
        else:
            cv2.polylines(img, [poly.reshape(-1, 1, 2)], True, color, 1, cv2.LINE_AA)
        if labels:
            x, y = poly.min(axis=0)
            cv2.putText(img, name, (int(x), max(int(y) - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        color, 1, cv2.LINE_AA)
    return img
