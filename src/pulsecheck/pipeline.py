"""Analysis pipeline: RegionTraces -> resample -> rPPG method -> bandpass -> heart rate.

Every region (and the ``face`` aggregate) is analysed on one shared uniform time grid, so their
pulse signals and HR windows line up sample for sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pulsecheck.config import SETTINGS
from pulsecheck.dsp import HRResult, HRSeries, bandpass, estimate_hr, hr_series, resample_uniform
from pulsecheck.extract import RegionTraces
from pulsecheck.regions import SKIN_REGIONS
from pulsecheck.rppg import METHODS


@dataclass
class RegionResult:
    name: str
    pulse: np.ndarray  # (N,) bandpassed pulse on the shared grid; all NaN if unusable
    hr: HRResult  # whole-clip estimate
    series: HRSeries  # sliding-window estimates


@dataclass
class Analysis:
    method: str
    fs: float
    t: np.ndarray  # (N,) seconds from the start of the grid; matches ``series.t_center``
    valid: np.ndarray  # (N,) bool; False inside gaps longer than ``max_gap_s`` (face estimate)
    regions: dict[str, RegionResult]
    face: RegionResult  # computed on the nan-mean colour of the 7 skin regions
    quality: dict[str, float]


def _interp_onto(
    t: np.ndarray, x: np.ndarray, t_u: np.ndarray, fs: float, max_gap_s: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Linear interpolation of ``x`` (T, 3) onto the shared grid ``t_u``.

    Rows with a non-finite value are dropped first. Grid points outside the span of the valid
    rows, or inside a gap longer than ``max_gap_s``, are flagged invalid. ``None`` if fewer than
    two rows are valid. Mirrors ``dsp.resample_uniform`` but targets a given grid.
    """
    ok = np.isfinite(t) & np.isfinite(x).all(axis=1)
    if ok.sum() < 2:
        return None
    tv, xv = t[ok], x[ok]
    x_u = np.stack([np.interp(t_u, tv, xv[:, c]) for c in range(3)], axis=1)
    left = np.clip(np.searchsorted(tv, t_u, side="right") - 1, 0, len(tv) - 2)
    valid = ((tv[left + 1] - tv[left]) <= max_gap_s) & (t_u >= tv[0] - 0.5 / fs)
    valid &= t_u <= tv[-1] + 0.5 / fs
    return x_u, valid


def _unusable(name: str, n: int, fs: float) -> RegionResult:
    """All-NaN result whose series sits on the same window grid as a usable region's."""
    pulse = np.full(n, np.nan)
    series = hr_series(pulse, fs, valid=np.zeros(n, bool))  # every window invalid -> NaN
    return RegionResult(name, pulse, HRResult(np.nan, np.nan, np.nan), series)


def _analyze_rgb(
    name: str, rgb: np.ndarray | None, valid: np.ndarray, method: str, fs: float
) -> RegionResult:
    n = len(valid)
    if rgb is None or not np.all(rgb.mean(axis=0) > 0):
        return _unusable(name, n, fs)
    with np.errstate(all="ignore"):
        pulse = bandpass(METHODS[method](rgb, fs), fs)
    if not np.isfinite(pulse).all():
        return _unusable(name, n, fs)
    return RegionResult(
        name=name,
        pulse=pulse,
        hr=estimate_hr(pulse, fs),
        series=hr_series(pulse, fs, valid=valid),
    )


def analyze(traces: RegionTraces, method: str = "pos", fs: float = SETTINGS.fs) -> Analysis:
    """Heart-rate analysis of every region plus an overall ``face`` estimate.

    The ``face`` signal is the per-frame nan-mean colour of the skin regions. It defines the
    shared uniform time grid; each region is interpolated onto that grid separately, so a region
    with missing frames cannot shorten the others. A region that is unusable (fewer than two
    valid samples, or a non-positive channel mean) yields NaN results instead of an error.

    Raises ``KeyError`` for an unknown method, and ``ValueError`` if the face signal is unusable
    or the clip is shorter than ``SETTINGS.min_seconds``.
    """
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; choose from {sorted(METHODS)}")

    names = traces.region_names
    skin_idx = [names.index(n) for n in SKIN_REGIONS if n in names]
    skin = traces.rgb[:, skin_idx, :].astype(float)
    finite = np.isfinite(skin)
    count = finite.sum(axis=1)  # (T, 3) skin regions contributing per frame and channel
    total = np.where(finite, skin, 0.0).sum(axis=1)
    face_rgb = np.where(count > 0, total / np.maximum(count, 1), np.nan)  # nan-mean, no warnings

    t_u, face_u, face_valid = resample_uniform(  # ValueError("not enough valid samples")
        traces.t, face_rgb, fs=fs, max_gap_s=SETTINGS.max_gap_s
    )
    if len(t_u) < SETTINGS.min_seconds * fs:
        raise ValueError("clip too short")
    if not np.all(face_u.mean(axis=0) > 0):
        raise ValueError("face signal is unusable (non-positive mean colour)")

    regions: dict[str, RegionResult] = {}
    for j, name in enumerate(names):
        res = _interp_onto(
            traces.t, traces.rgb[:, j, :].astype(float), t_u, fs, SETTINGS.max_gap_s
        )
        if res is None:
            regions[name] = _unusable(name, len(t_u), fs)
        else:
            regions[name] = _analyze_rgb(name, res[0], res[1], method, fs)

    return Analysis(
        method=method,
        fs=fs,
        t=t_u - t_u[0],
        valid=face_valid,
        regions=regions,
        face=_analyze_rgb("face", face_u, face_valid, method, fs),
        quality=traces.quality_summary(),
    )
