"""Per-video features: each video becomes one row of explainable numbers.

Four groups, every feature name prefixed with its group letter:

* ``A``: pulse presence, the classic "is there a pulse?" baseline (how strong and how plausible
  the heart-rate peak is in the face signal and in each skin region).
* ``B``, ``C``: cross-region consistency features (added in a later task).
* ``D``: video quality (duration, face size, motion, brightness).

All features are computed on sliding windows of ``SETTINGS.feature_window_s`` seconds with a step
of ``SETTINGS.feature_step_s``. A "valid" window is one whose estimated bpm is not NaN. A feature
that cannot be computed (no valid window, unusable region) is NaN; nothing here warns or raises
for that.
"""

from __future__ import annotations

import numpy as np

from pulsecheck.config import SETTINGS
from pulsecheck.extract import RegionTraces
from pulsecheck.pipeline import Analysis, analyze, face_signal
from pulsecheck.regions import SKIN_REGIONS

REST_BPM_RANGE = (45.0, 150.0)  # plausible resting heart rate (bpm), inclusive

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "A": (
        "a_snr_face",  # median window SNR of the face signal (dB)
        "a_snr_regions_median",  # median over skin regions of each region's median SNR (dB)
        "a_snr_regions_max",  # best skin region's median SNR (dB)
        "a_hr_std_face",  # std of the face's window bpm (bpm)
        "a_hr_in_rest_range",  # fraction of valid face windows with bpm in REST_BPM_RANGE
        "a_valid_frac",  # fraction of the grid where the face signal has data
    ),
    "B": (),  # filled in by the consistency features
    "C": (),
    "D": (
        "d_duration_s",  # clip length (s)
        "d_face_found_ratio",  # fraction of frames with a face
        "d_face_width_px",  # median face width (px)
        "d_motion",  # median landmark motion (face widths per frame)
        "d_brightness",  # median green level of the face signal (0-255 scale)
        "d_n_windows",  # number of feature windows
    ),
}


def _median(values: np.ndarray) -> float:
    """Median ignoring NaN; NaN (without a warning) when nothing is finite."""
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _valid_snr(result) -> np.ndarray:
    """SNR (dB) of the windows of a ``RegionResult`` whose bpm is not NaN."""
    return result.series.snr_db[np.isfinite(result.series.bpm)]


def group_a(analysis: Analysis) -> dict[str, float]:
    """Pulse-presence features (see ``FEATURE_GROUPS["A"]``); NaN where not computable."""
    face = analysis.face
    bpm = face.series.bpm[np.isfinite(face.series.bpm)]
    region_snr = np.array([_median(_valid_snr(analysis.regions[n])) for n in SKIN_REGIONS])
    region_snr = region_snr[np.isfinite(region_snr)]
    low, high = REST_BPM_RANGE
    return {
        "a_snr_face": _median(_valid_snr(face)),
        "a_snr_regions_median": _median(region_snr),
        "a_snr_regions_max": float(region_snr.max()) if region_snr.size else float("nan"),
        "a_hr_std_face": float(bpm.std()) if bpm.size else float("nan"),
        "a_hr_in_rest_range": float(((bpm >= low) & (bpm <= high)).mean())
        if bpm.size
        else float("nan"),
        "a_valid_frac": float(analysis.valid.mean()),
    }


def group_d(traces: RegionTraces, analysis: Analysis) -> dict[str, float]:
    """Video-quality features (see ``FEATURE_GROUPS["D"]``)."""
    quality = traces.quality_summary()
    green = face_signal(traces.rgb, traces.region_names)[:, 1]
    return {
        "d_duration_s": quality["duration_s"],
        "d_face_found_ratio": quality["face_found_ratio"],
        "d_face_width_px": quality["median_face_width_px"],
        "d_motion": quality["median_motion"],
        "d_brightness": _median(green),
        "d_n_windows": float(len(analysis.face.series.t_center)),
    }


def video_features(
    traces: RegionTraces, groups: str = "ABCD", method: str = "pos"
) -> dict[str, float]:
    """The requested feature groups of one video, merged in ``FEATURE_GROUPS`` order.

    Runs ``analyze`` with ``SETTINGS.feature_window_s`` / ``feature_step_s``. Unusable regions
    give NaN features. Raises ``ValueError`` for an unknown group letter or when ``analyze``
    does (face unusable, clip shorter than ``SETTINGS.min_seconds``), and ``NotImplementedError``
    for groups B and C, which are not built yet.
    """
    unknown = sorted(set(groups) - set(FEATURE_GROUPS))
    if unknown:
        raise ValueError(f"unknown feature group(s) {unknown}; choose from {list(FEATURE_GROUPS)}")
    missing = sorted(set(groups) & {"B", "C"})
    if missing:
        raise NotImplementedError(f"feature group(s) {missing} are not implemented yet")
    analysis = analyze(
        traces,
        method,
        window_s=SETTINGS.feature_window_s,
        step_s=SETTINGS.feature_step_s,
    )
    out: dict[str, float] = {}
    if "A" in groups:
        out.update(group_a(analysis))
    if "D" in groups:
        out.update(group_d(traces, analysis))
    return out
