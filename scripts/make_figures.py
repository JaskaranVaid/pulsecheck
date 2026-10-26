"""Regenerate the figures used by docs/METHOD.md into docs/figures/.

    python scripts/make_figures.py

* ``regions.png``: the 7 skin regions and the control region drawn on the public-domain NASA
  portrait in ``tests/fixtures`` (the only face image that may be published; see its README).
* ``flicker_methods.png``: spectra of GREEN and POS on a synthetic 72 bpm pulse corrupted by
  a 108 bpm lighting flicker. Pure OpenCV drawing (matplotlib is not a dependency).

Needs no webcam, no model download and no dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))  # synth.py is a plain module there

from pulsecheck.dsp import bandpass, power_spectrum
from pulsecheck.regions import region_polygons
from pulsecheck.rppg import METHODS
from pulsecheck.viz import _COLORS, draw_regions
from synth import synth_rgb

FIXTURES = ROOT / "tests" / "fixtures"
OUT = ROOT / "docs" / "figures"
SCALE = 3
_LABEL_SHIFT = {"forehead_center": (0, -22), "forehead_right": (40, 0)}
PAD = 0.35  # extra margin around the landmark bounding box, as a fraction of its size


def make_regions_figure() -> Path:
    img = cv2.imread(str(FIXTURES / "face.jpg"))
    landmarks = np.load(FIXTURES / "landmarks_face.npy")
    if img is None:
        raise SystemExit(f"cannot read {FIXTURES / 'face.jpg'}")
    h, w = img.shape[:2]

    # Crop around the head first, then upscale, then draw: lines and labels stay crisp.
    x0, y0 = landmarks.min(axis=0)
    x1, y1 = landmarks.max(axis=0)
    mx, my = PAD * (x1 - x0), PAD * (y1 - y0)
    cx0, cy0 = max(int(x0 - mx), 0), max(int(y0 - my), 0)
    cx1, cy1 = min(int(x1 + mx), w), min(int(y1 + my), h)
    crop = cv2.resize(
        img[cy0:cy1, cx0:cx1], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC
    )
    shifted = (landmarks - (cx0, cy0)) * SCALE
    polys = region_polygons(shifted.astype(np.float32), crop.shape[:2])
    out = draw_regions(crop, polys, labels=False)
    for name, poly in polys.items():  # own labels: bigger, with a dark outline to stay readable
        if len(poly) < 3:
            continue
        x, y = poly.min(axis=0)
        dx, dy = _LABEL_SHIFT.get(name, (0, 0))  # keep neighbouring forehead labels apart
        org = (int(x) + dx, max(int(y) - 6 + dy, 12))
        for thick, colour in ((3, (0, 0, 0)), (1, _COLORS.get(name, (255, 255, 255)))):
            cv2.putText(out, name, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, thick, cv2.LINE_AA)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "regions.png"
    cv2.imwrite(str(path), out)
    return path


# ------------------------------------------------------------------ flicker figure (OpenCV)

_BG, _FG, _GRID = (255, 255, 255), (40, 40, 40), (225, 225, 225)
_BLUE, _RED, _GREY = (180, 100, 20), (50, 50, 200), (120, 120, 120)


def _panel(title: str, freqs: np.ndarray, power: np.ndarray, w: int, h: int) -> np.ndarray:
    """One spectrum panel, 0.5-4 Hz on x, power normalised to its own maximum."""
    img = np.full((h, w, 3), _BG, np.uint8)
    left, right, top, bottom = 50, 20, 40, 45
    pw, ph = w - left - right, h - top - bottom
    lo, hi = 0.5, 4.0
    band = (freqs >= lo) & (freqs <= hi)
    f, p = freqs[band], power[band] / power[band].max()

    def px(hz: float) -> int:
        return int(left + (hz - lo) / (hi - lo) * pw)

    for hz in np.arange(1.0, 4.01, 1.0):
        cv2.line(img, (px(hz), top), (px(hz), top + ph), _GRID, 1)
        cv2.putText(img, f"{hz:.0f} Hz", (px(hz) - 18, top + ph + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, _FG, 1, cv2.LINE_AA)
    cv2.rectangle(img, (left, top), (left + pw, top + ph), _FG, 1)
    pts = np.stack([[px(a) for a in f], [int(top + ph - b * ph) for b in p]], axis=1)
    cv2.polylines(img, [pts.astype(np.int32).reshape(-1, 1, 2)], False, _BLUE, 2, cv2.LINE_AA)
    for hz, colour in ((1.2, _BLUE), (1.8, _RED)):
        cv2.line(img, (px(hz), top), (px(hz), top + ph), colour, 1, cv2.LINE_AA)
    for dy, label, colour in ((0, "true pulse: 72 bpm", _BLUE), (18, "lighting flicker: 108 bpm", _RED)):
        cv2.line(img, (px(2.55), top + 12 + dy), (px(2.55) + 18, top + 12 + dy), colour, 2)
        cv2.putText(img, label, (px(2.55) + 24, top + 16 + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    colour, 1, cv2.LINE_AA)
    cv2.putText(img, title, (left, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, _FG, 2, cv2.LINE_AA)
    cv2.putText(img, "relative power", (4, top + ph // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, _GREY,
                1, cv2.LINE_AA)
    return img


def make_flicker_figure() -> Path:
    fs = 30.0
    _, rgb = synth_rgb(72, 20, fs, intensity_noise=(1.8, 0.01))  # same setup as tests/test_rppg.py
    panels = []
    for name in ("green", "pos"):
        pulse = bandpass(METHODS[name](rgb, fs), fs)
        freqs, power = power_spectrum(pulse, fs)
        peak = freqs[(freqs >= 0.7) & (freqs <= 3.3)][
            np.argmax(power[(freqs >= 0.7) & (freqs <= 3.3)])
        ]
        panels.append(_panel(f"{name.upper()}: peak at {peak * 60:.0f} bpm", freqs, power, 520, 300))
    out = np.hstack(panels)
    path = OUT / "flicker_methods.png"
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out)
    return path


def main() -> None:
    print(f"wrote {make_regions_figure()}")
    print(f"wrote {make_flicker_figure()}")


if __name__ == "__main__":
    main()
