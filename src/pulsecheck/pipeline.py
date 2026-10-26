"""Analysis pipeline: RegionTraces -> resample -> rPPG method -> band-pass -> heart rate.

Every region (and the ``face`` aggregate) is analysed on one shared uniform time grid, so their
pulse signals and HR windows line up sample for sample.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pulsecheck.config import SETTINGS
from pulsecheck.dsp import (
    MAX_INVALID_FRACTION,
    HRResult,
    HRSeries,
    bandpass,
    estimate_hr,
    hr_series,
    resample_uniform,
)
from pulsecheck.extract import RegionTraces
from pulsecheck.regions import SKIN_REGIONS
from pulsecheck.rppg import METHODS


@dataclass
class RegionResult:
    name: str
    pulse: np.ndarray  # (N,) band-passed pulse on the shared grid; all NaN if unusable
    hr: HRResult  # whole-clip estimate; NaN if over MAX_INVALID_FRACTION of ``valid`` is False
    series: HRSeries  # sliding-window estimates
    valid: np.ndarray  # (N,) bool on the shared grid; all False if unusable


@dataclass
class Analysis:
    method: str
    fs: float
    t: np.ndarray  # (N,) seconds from the start of the grid; matches ``series.t_center``
    valid: np.ndarray  # (N,) bool; False inside gaps longer than ``max_gap_s`` (face estimate)
    regions: dict[str, RegionResult]
    face: RegionResult  # computed on the normalised mean colour of the 7 skin regions
    quality: dict[str, float]


_NAN_HR = HRResult(np.nan, np.nan, np.nan)


def _unusable(name: str, n: int, fs: float) -> RegionResult:
    """All-NaN result whose series sits on the same window grid as a usable region's."""
    pulse = np.full(n, np.nan)
    valid = np.zeros(n, bool)
    series = hr_series(pulse, fs, valid=valid)  # every window invalid -> NaN
    return RegionResult(name, pulse, _NAN_HR, series, valid)


def _nanmean(x: np.ndarray, axis: int) -> np.ndarray:
    """``np.nanmean`` without the all-NaN RuntimeWarning (all-NaN slices give NaN)."""
    finite = np.isfinite(x)
    count = finite.sum(axis=axis)
    total = np.where(finite, x, 0.0).sum(axis=axis)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def face_signal(rgb: np.ndarray, names: Sequence[str] = SKIN_REGIONS) -> np.ndarray:
    """Combined colour of the skin regions: ``(T, R, 3)`` -> ``(T, 3)``.

    ``names`` labels the ``R`` regions of ``rgb``; only those in ``SKIN_REGIONS`` contribute (so
    the ``control`` region is ignored). Each region's channels are first divided by their own
    temporal mean, so every region sits around 1.0; the per-frame mean over the regions with data
    is then scaled back by the mean of the region means to keep an RGB-like level. Without this,
    a bright and a dark region dropping in and out (one frame lost, the next found) would step the
    average by far more than the pulse. A region channel with no data or a non-positive mean is
    left out. A frame where no skin region has data is NaN.
    """
    skin_idx = [i for i, n in enumerate(names) if n in SKIN_REGIONS]
    skin = np.asarray(rgb, float)[:, skin_idx, :]
    level = _nanmean(skin, axis=0)  # (R, 3) temporal mean per region and channel
    level = np.where(level > 0, level, np.nan)  # NaN (no data) or non-positive: left out
    with np.errstate(invalid="ignore"):
        relative = skin / level  # NaN for left-out region channels
    return _nanmean(relative, axis=1) * _nanmean(level, axis=0)


def pulse_from_rgb(rgb: np.ndarray, method: str, fs: float) -> np.ndarray | None:
    """Band-passed pulse from a uniform ``(N, 3)`` RGB signal, or ``None`` if it is unusable.

    Unusable means a non-positive channel mean or a non-finite result (e.g. a degenerate
    signal). May raise ``ValueError`` if the signal is too short for the method or filter.
    """
    if not np.all(rgb.mean(axis=0) > 0):
        return None
    with np.errstate(all="ignore"):
        pulse = METHODS[method](rgb, fs)
        # CHROM band-passes X and Y itself (its alpha is defined on the filtered signals), so it
        # is already band-limited; filtering it again would only steepen the band edges.
        if method != "chrom":
            pulse = bandpass(pulse, fs)
    return pulse if np.isfinite(pulse).all() else None


def _analyze_rgb(
    name: str, rgb: np.ndarray | None, valid: np.ndarray, method: str, fs: float
) -> RegionResult:
    n = len(valid)
    pulse = None if rgb is None else pulse_from_rgb(rgb, method, fs)
    if pulse is None:
        return _unusable(name, n, fs)
    # The whole-clip estimate follows the same rule as a single window: no answer when too much
    # of the grid was interpolated across long gaps or padded where the region was missing.
    mostly_valid = (~valid).mean() <= MAX_INVALID_FRACTION
    return RegionResult(
        name=name,
        pulse=pulse,
        hr=estimate_hr(pulse, fs) if mostly_valid else _NAN_HR,
        series=hr_series(pulse, fs, valid=valid),
        valid=valid,
    )


def analyze(traces: RegionTraces, method: str = "pos", fs: float = SETTINGS.fs) -> Analysis:
    """Heart-rate analysis of every region plus an overall ``face`` estimate.

    The ``face`` signal is the mean of the skin regions, each normalised by its own mean colour
    (``face_signal``). All signals share
    one uniform grid that starts at ``traces.t[0]`` (the video start) and ends at ``traces.t[-1]``;
    ``Analysis.t`` and every ``series.t_center`` are seconds from ``traces.t[0]``, so windows line
    up with ground truth timed from the start of the video. Samples where a signal has no data
    (leading or trailing frames without a face, or gaps over ``max_gap_s``) are ``valid=False``
    and windows with too many of them are NaN; so is the whole-clip ``hr`` of a signal with more
    than ``MAX_INVALID_FRACTION`` of its grid invalid. A region that is unusable (fewer than two
    valid samples, or a non-positive channel mean) yields NaN results and an all-False ``valid``
    instead of an error.

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
