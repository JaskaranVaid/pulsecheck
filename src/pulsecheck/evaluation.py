"""Heart-rate evaluation: error metrics and the ground-truth reference series."""

from __future__ import annotations

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

ALIGN_TOL_S = 0.5  # largest tolerated gap between matched window centres
_PLAUSIBLE_DT_S = (0.001, 1.0)  # median ground-truth sample spacing that looks like seconds


def hr_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """``mae``, ``rmse``, ``pearson`` and ``n`` over the pairs where both values are finite.

    ``mae``/``rmse`` are NaN with no pairs; ``pearson`` is NaN with fewer than 2 pairs or when
    either side is constant.
    """
    pred = np.asarray(pred, float)
    ref = np.asarray(ref, float)
    if pred.shape != ref.shape:
        raise ValueError(f"pred and ref differ in shape: {pred.shape} vs {ref.shape}")
    ok = np.isfinite(pred) & np.isfinite(ref)
    p, r = pred[ok], ref[ok]
    n = int(ok.sum())
    if n == 0:
        return {"mae": np.nan, "rmse": np.nan, "pearson": np.nan, "n": 0}
    err = p - r
    pearson = np.nan
    if n >= 2 and p.std() > 0 and r.std() > 0:
        pearson = float(np.corrcoef(p, r)[0, 1])
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "pearson": pearson,
        "n": n,
    }


def _check_seconds(t_s: np.ndarray) -> None:
    """Raise ``ValueError`` unless ``t_s`` looks like strictly increasing seconds."""
    dt = np.diff(t_s)
    if len(t_s) < 2 or not np.all(dt > 0):
        raise ValueError("ground-truth timestamps must be strictly increasing")
    if not _PLAUSIBLE_DT_S[0] <= np.median(dt) <= _PLAUSIBLE_DT_S[1]:
        raise ValueError(
            f"ground-truth timestamps do not look like seconds (median spacing {np.median(dt):g})"
        )


def _reference_pulse(
    t_s: np.ndarray, ppg: np.ndarray, n: int, fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """PPG resampled onto ``n`` samples at ``fs`` from ``t_s[0]`` and band-passed: ``(x, valid)``."""
    grid = t_s[0] + np.arange(n) / fs
    _, x, valid = resample_uniform(t_s, ppg, t_out=grid)
    return bandpass(x, fs), valid


def reference_series(
    t_s: np.ndarray,
    ppg: np.ndarray,
    window_s: float = SETTINGS.window_s,
    step_s: float = SETTINGS.step_s,
    fs: float = SETTINGS.fs,
) -> HRSeries:
    """Ground-truth HR series from a PPG recording.

    The PPG is resampled to ``fs`` on a grid starting at ``t_s[0]`` and band-passed, then cut
    into the same windows as the video analysis. ``t_center`` is on the ground-truth clock
    (``t_s[0]`` + seconds into the grid), so it can be compared with the video's window centres.

    Raises ``ValueError`` if the timestamps do not look like increasing seconds.
    """
    t_s = np.asarray(t_s, float)
    ppg = np.asarray(ppg, float)
    _check_seconds(t_s)
    n = int(np.floor((t_s[-1] - t_s[0]) * fs + 1e-9)) + 1
    x, valid = _reference_pulse(t_s, ppg, n, fs)
    series = hr_series(x, fs, window_s=window_s, step_s=step_s, valid=valid)
    return HRSeries(series.t_center + t_s[0], series.bpm, series.snr_db)


def reference_hr(
    t_s: np.ndarray, ppg: np.ndarray, duration_s: float, fs: float = SETTINGS.fs
) -> HRResult:
    """Whole-clip ground-truth HR over the first ``duration_s`` seconds of a PPG recording.

    The counterpart of a video's whole-clip ``face.hr``: the PPG is resampled onto the same
    number of ``fs`` samples as a video grid spanning ``duration_s`` (from ``t_s[0]``),
    band-passed, and read with ``estimate_hr``. NaN when more than ``MAX_INVALID_FRACTION`` of
    that grid is not covered by the recording (the rule the video estimate follows).

    Raises ``ValueError`` if the timestamps do not look like increasing seconds.
    """
    t_s = np.asarray(t_s, float)
    ppg = np.asarray(ppg, float)
    _check_seconds(t_s)
    n = int(np.floor(duration_s * fs + 1e-9)) + 1
    x, valid = _reference_pulse(t_s, ppg, n, fs)
    if (~valid).mean() > MAX_INVALID_FRACTION:
        return HRResult(np.nan, np.nan, np.nan)
    return estimate_hr(x, fs)


def align_series(pred: HRSeries, ref: HRSeries) -> tuple[np.ndarray, np.ndarray]:
    """Pair prediction and reference windows by index, truncated to the shorter series.

    Both series start at the video start with identical windows, so matching centres must agree
    within ``ALIGN_TOL_S``; otherwise the windows do not line up and ``ValueError`` is raised.
    """
    n = min(len(pred.bpm), len(ref.bpm))
    if n == 0:
        raise ValueError("no windows to compare")
    gap = np.abs(pred.t_center[:n] - ref.t_center[:n]).max()
    if gap > ALIGN_TOL_S:
        raise ValueError(
            f"prediction and reference windows do not align (centres differ by {gap:.2f} s)"
        )
    return pred.bpm[:n], ref.bpm[:n]
