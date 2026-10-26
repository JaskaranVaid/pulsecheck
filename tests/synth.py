"""Synthetic RGB traces with a known pulse, shared by the rPPG/DSP tests.

Import as a plain module (``tests/`` has no ``__init__.py``)::

    from synth import synth_rgb
"""

from __future__ import annotations

import numpy as np

DC = np.array([180.0, 130.0, 110.0])  # mean skin colour (R, G, B)
BLOOD_VOLUME = np.array([0.33, 0.77, 0.53])  # relative pulse amplitude per channel


def synth_rgb(
    bpm: float,
    seconds: float,
    fs: float = 30.0,
    pulse_amp: float = 0.002,
    intensity_noise: tuple[float, float] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(t, rgb)`` with ``rgb`` of shape ``(T, 3)``.

    ``rgb = DC * (1 + pulse_amp * v * w(t)) * (1 + a_n * sin(2 pi f_n t))`` plus Gaussian sensor
    noise (sigma 0.05). ``w(t) = sin(2 pi f0 t) + 0.3 sin(4 pi f0 t)``. ``intensity_noise =
    (f_n, a_n)`` adds a lighting flicker that scales all channels equally.
    """
    t = np.arange(round(seconds * fs)) / fs
    f0 = bpm / 60.0
    w = np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(4 * np.pi * f0 * t)
    rgb = DC * (1 + pulse_amp * np.outer(w, BLOOD_VOLUME))
    if intensity_noise is not None:
        f_n, a_n = intensity_noise
        rgb = rgb * (1 + a_n * np.sin(2 * np.pi * f_n * t))[:, None]
    rgb = rgb + np.random.default_rng(seed).normal(0.0, 0.05, rgb.shape)
    return t, rgb
