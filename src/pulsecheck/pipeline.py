"""Analysis pipeline: RegionTraces -> resample -> rPPG method -> bandpass -> heart rate.

Every region (and the ``face`` aggregate) is analysed on one shared uniform time grid, so their
pulse signals and HR windows line up sample for sample.
"""

from __future__ import annotations

from collections.abc import Sequence
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


def _unusable(name: str, n: int, fs: float) -> RegionResult:
    """All-NaN result whose series sits on the same window grid as a usable region's."""
    pulse = np.full(n, np.nan)
    series = hr_series(pulse, fs, valid=np.zeros(n, bool))  # every window invalid -> NaN
    return RegionResult(name, pulse, HRResult(np.nan, np.nan, np.nan), series)


def face_signal(rgb: np.ndarray, names: Sequence[str] = SKIN_REGIONS) -> np.ndarray:
    """Per-frame nan-mean colour of the skin regions: ``(T, R, 3)`` -> ``(T, 3)``.

    ``names`` labels the ``R`` regions of ``rgb``; only those in ``SKIN_REGIONS`` contribute (so
    the ``control`` region is ignored). A frame where no skin region has data is NaN.
    """
    skin_idx = [i for i, n in enumerate(names) if n in SKIN_REGIONS]
    skin = np.asarray(rgb, float)[:, skin_idx, :]
    finite = np.isfinite(skin)
    count = finite.sum(axis=1)  # (T, 3) skin regions contributing per frame and channel
    total = np.where(finite, skin, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)  # nan-mean, no warnings


def pulse_from_rgb(rgb: np.ndarray, method: str, fs: float) -> np.ndarray | None:
    """Band-passed pulse from a uniform ``(N, 3)`` RGB signal, or ``None`` if it is unusable.

    Unusable means a non-positive channel mean or a non-finite result (e.g. a degenerate
    signal). May raise ``ValueError`` if the signal is too short for the method or filter.
    """
    if not np.all(rgb.mean(axis=0) > 0):
        return None
    with np.errstate(all="ignore"):
        pulse = bandpass(METHODS[method](rgb, fs), fs)
    return pulse if np.isfinite(pulse).all() else None


def _analyze_rgb(
    name: str, rgb: np.ndarray | None, valid: np.ndarray, method: str, fs: float
) -> RegionResult:
    n = len(valid)
    pulse = None if rgb is None else pulse_from_rgb(rgb, method, fs)
    if pulse is None:
        return _unusable(name, n, fs)
    return RegionResult(
        name=name,
        pulse=pulse,
        hr=estimate_hr(pulse, fs),
        series=hr_series(pulse, fs, valid=valid),
    )


def analyze(traces: RegionTraces, method: str = "pos", fs: float = SETTINGS.fs) -> Analysis:
    """Heart-rate analysis of every region plus an overall ``face`` estimate.

    The ``face`` signal is the per-frame nan-mean colour of the skin regions. All signals share
    one uniform grid that starts at ``traces.t[0]`` (the video start) and ends at ``traces.t[-1]``;
    ``Analysis.t`` and every ``series.t_center`` are seconds from ``traces.t[0]``, so windows line
    up with ground truth timed from the start of the video. Samples where a signal has no data
    (leading or trailing frames without a face, or gaps over ``max_gap_s``) are ``valid=False``
    and windows with too many of them are NaN. A region that is unusable (fewer than two valid
    samples, or a non-positive channel mean) yields NaN results instead of an error.

    Raises ``KeyError`` for an unknown method, and ``ValueError`` if the face signal is unusable
    or the clip is shorter than ``SETTINGS.min_seconds``.
    """
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; choose from {sorted(METHODS)}")

    names = traces.region_names
    face_rgb = face_signal(traces.rgb, names)

    # One grid for everything: from the first to the last video frame, at 1/fs. Leading or
    # trailing samples where the face is missing are flagged invalid (not trimmed), so window
    # centres are always seconds from the video start.
    n = int(np.floor((traces.t[-1] - traces.t[0]) * fs + 1e-9)) + 1
    if n < SETTINGS.min_seconds * fs:
        raise ValueError("clip too short")
    grid = traces.t[0] + np.arange(n) / fs

    # ValueError("not enough valid samples") if the face is never usable
    t_u, face_u, face_valid = resample_uniform(
        traces.t, face_rgb, max_gap_s=SETTINGS.max_gap_s, t_out=grid
    )
    if not np.all(face_u.mean(axis=0) > 0):
        raise ValueError("face signal is unusable (non-positive mean colour)")

    regions: dict[str, RegionResult] = {}
    for j, name in enumerate(names):
        try:
            _, x_u, valid = resample_uniform(
                traces.t, traces.rgb[:, j, :], max_gap_s=SETTINGS.max_gap_s, t_out=grid
            )
        except ValueError:  # fewer than two valid samples
            regions[name] = _unusable(name, n, fs)
        else:
            regions[name] = _analyze_rgb(name, x_u, valid, method, fs)

    return Analysis(
        method=method,
        fs=fs,
        t=t_u - t_u[0],  # == arange(n) / fs, seconds from traces.t[0]
        valid=face_valid,
        regions=regions,
        face=_analyze_rgb("face", face_u, face_valid, method, fs),
        quality=traces.quality_summary(),
    )
