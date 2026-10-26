"""Pulse signals from per-frame mean skin colour: GREEN, CHROM and POS.

Each method takes ``rgb`` of shape ``(N, 3)`` (RGB order, uniformly sampled at ``fs``, no NaN)
and returns an ``(N,)`` pulse signal. GREEN and POS return a zero-mean signal that the caller
bandpasses; CHROM bandpasses internally. Pure NumPy/SciPy; no video or face-tracking imports.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from pulsecheck.config import SETTINGS
from pulsecheck.dsp import bandpass

EPS = 1e-9  # guards std ratios against division by zero


def _check_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must have shape (N, 3), got {rgb.shape}")
    return rgb


def green(rgb: np.ndarray, fs: float) -> np.ndarray:
    """Green channel, normalised to zero mean. Sensitive to lighting flicker."""
    g = _check_rgb(rgb)[:, 1]
    return g / g.mean() - 1


def chrom(rgb: np.ndarray, fs: float) -> np.ndarray:
    """CHROM (de Haan & Jeanne 2013), global version. Bandpasses internally."""
    rgb = _check_rgb(rgb)
    cn = rgb / rgb.mean(axis=0)
    r, g, b = cn[:, 0], cn[:, 1], cn[:, 2]
    x = 3 * r - 2 * g
    y = 1.5 * r + g - 1.5 * b
    xf = bandpass(x, fs)
    yf = bandpass(y, fs)
    alpha = xf.std() / (yf.std() + EPS)
    return xf - alpha * yf


def pos(rgb: np.ndarray, fs: float, window_s: float = SETTINGS.pos_window_s) -> np.ndarray:
    """POS (Wang et al. 2017): projection onto the plane orthogonal to skin tone, overlap-added.

    Returns a zero-mean signal that is not bandpassed. Raises ``ValueError`` if the signal is
    shorter than one window.
    """
    rgb = _check_rgb(rgb)
    n = len(rgb)
    win_len = round(window_s * fs)
    if n < win_len:
        raise ValueError("signal shorter than POS window")
    proj = np.array([[0, 1, -1], [-2, 1, 1]], dtype=float)
    h_out = np.zeros(n)
    for end in range(win_len, n + 1):
        win = rgb[end - win_len : end]
        c = (win / win.mean(axis=0)).T  # 3 x win_len, temporally normalised
        s = proj @ c  # 2 x win_len
        h = s[0] + (s[0].std() / (s[1].std() + EPS)) * s[1]
        h_out[end - win_len : end] += h - h.mean()
    return h_out - h_out.mean()


METHODS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "green": green,
    "chrom": chrom,
    "pos": pos,
}
