"""Per-video features: each video becomes one row of explainable numbers.

Four groups, every feature name prefixed with its group letter:

* ``A``: pulse presence, the classic "is there a pulse?" baseline (how strong and how plausible
  the heart-rate peak is in the face signal and in each skin region).
* ``B``: spatial agreement. Do the skin regions share one heartbeat (correlation, coherence,
  agreeing heart rates)?
* ``C``: physiological consistency. Is the *pattern* of that heartbeat across the face that of a
  living face: when it arrives in each region (phase lag behind the forehead), how strong it is
  in each region (amplitude map), its waveform shape (harmonics), and that the background does
  not pulse?
* ``D``: video quality (duration, face size, motion, brightness).

Why B and C: a high-quality deepfake can inherit a believable pulse from its driver video, so
"is there a pulse?" (group A) no longer separates it from a real face. But the inherited pulse is
pasted onto the whole face at once, so the timing and strength differences that blood flow
creates between regions are missing. Group C measures exactly those differences. Group B only
checks that the regions share one pulse at all; it catches faces with no common pulse (generator
noise), not an inherited one, whose regions share it perfectly.

All features are computed on sliding windows of ``SETTINGS.feature_window_s`` seconds with a step
of ``SETTINGS.feature_step_s`` (``b_coherence`` alone uses non-overlapping windows; see
``group_b``). A "valid" window is one whose estimated face bpm is not NaN; the face bpm of that
window is the frequency ``f0`` at which every per-window B/C quantity is read. A feature that
cannot be computed (no valid window, unusable region) is NaN; nothing here warns or raises for
that.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from pulsecheck.config import SETTINGS
from pulsecheck.dsp import (
    MAX_INVALID_FRACTION,
    SNR_FUNDAMENTAL_HALFWIDTH_HZ,
    band_coherence,
    debias_coherence,
    estimate_hr,
    harmonic_ratio,
    phase_lag_s,
    power_spectrum,
)
from pulsecheck.extract import RegionTraces
from pulsecheck.pipeline import Analysis, RegionResult, analyze, face_signal
from pulsecheck.regions import CONTROL, SKIN_REGIONS

REST_BPM_RANGE = (45.0, 150.0)  # plausible resting heart rate (bpm), inclusive
REF_REGION = "forehead_center"  # lags are measured relative to this region (pulse arrives first)
LAG_REGIONS = tuple(r for r in SKIN_REGIONS if r != REF_REGION)  # the 6 regions with a lag
MIN_COHERENCE_WINDOWS = 2  # one window cannot separate shared signal from chance (ruling P2-R8)

# Note on the C columns: the real-face reference (``reference.RealFaceReference``) adds two more
# columns, ``c_amp_mahal`` and ``c_lag_mahal``, at model time. They are deliberately NOT listed
# here: they depend on which videos are the training reals, so they are not a property of one
# video and cannot be stored in the per-video feature table. Listing them would also make the
# table builder write all-NaN columns that the reference would then collide with.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "A": (
        "a_snr_face",  # median window SNR of the face signal (dB)
        "a_snr_regions_median",  # median over skin regions of each region's median SNR (dB)
        "a_snr_regions_max",  # best skin region's median SNR (dB)
        "a_hr_std_face",  # std of the face's window bpm (bpm)
        "a_hr_in_rest_range",  # fraction of valid face windows with bpm in REST_BPM_RANGE
        "a_valid_frac",  # fraction of the grid where the face signal has data
    ),
    "B": (
        "b_corr_median",  # median over windows of the median pairwise skin-region correlation
        "b_corr_min",  # median over windows of the weakest pairwise correlation
        "b_hr_spread",  # median over windows of the std of the region bpm values (bpm)
        "b_coherence",  # mean pairwise debiased coherence at the heart rate (non-overlapping)
    ),
    "C": (
        *(f"c_lag_{r}" for r in LAG_REGIONS),  # median lag behind REF_REGION (s)
        "c_lag_stability",  # mean over regions of the std of the window lags (s)
        *(f"c_amp_{r}" for r in SKIN_REGIONS),  # median share of the pulse amplitude (sums to ~1)
        "c_harmonic_face",  # median 2nd-harmonic / fundamental power ratio of the face pulse
        "c_harmonic_spread",  # std across skin regions of their median harmonic ratio
        "c_control_ratio",  # median background / face power at the heart rate
        "c_corr_stability",  # std over windows of the median pairwise correlation
    ),
    "D": (
        "d_duration_s",  # clip length (s)
        "d_face_found_ratio",  # fraction of frames with a face
        "d_face_width_px",  # median face width (px)
        "d_motion",  # median landmark motion (face widths per frame)
        "d_brightness",  # median green level of the face signal (0-255 scale)
        "d_n_windows",  # number of feature windows
    ),
}


def _finite(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _median(values) -> float:
    """Median ignoring NaN; NaN (without a warning) when nothing is finite."""
    values = _finite(values)
    return float(np.median(values)) if values.size else float("nan")


def _mean(values) -> float:
    """Mean ignoring NaN; NaN (without a warning) when nothing is finite."""
    values = _finite(values)
    return float(values.mean()) if values.size else float("nan")


def _std(values) -> float:
    """Population std ignoring NaN; NaN when fewer than two values are finite (a spread needs
    at least two)."""
    values = _finite(values)
    return float(values.std()) if values.size >= 2 else float("nan")


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


# --- shared helpers for groups B and C ----------------------------------------------------------


@dataclass(frozen=True)
class _Window:
    """One sliding analysis window whose face heart rate is valid."""

    index: int  # position in every ``series`` array (all regions share the window grid)
    span: slice  # its samples on the shared time grid
    f0: float  # the face's heart rate in this window (Hz); every B/C quantity is read here


def _window_length(analysis: Analysis) -> int:
    """Samples per analysis window.

    ``hr_series`` starts its first window at sample 0, so the first centre sits half a window in:
    ``t_center[0] = (win / 2) / fs``. This recovers ``win`` (including the short-clip case where
    the single window is the whole clip) without assuming which ``window_s`` was used.
    """
    return round(2 * analysis.face.series.t_center[0] * analysis.fs)


def _valid_windows(analysis: Analysis) -> list[_Window]:
    """The sliding windows with a finite face bpm, in time order (NaN windows are skipped)."""
    win = _window_length(analysis)
    series = analysis.face.series
    out = []
    for i, (t_center, bpm) in enumerate(zip(series.t_center, series.bpm, strict=True)):
        if np.isfinite(bpm):
            start = round(t_center * analysis.fs - win / 2)
            out.append(_Window(i, slice(start, start + win), bpm / 60.0))
    return out


def _usable(result: RegionResult, span: slice) -> bool:
    """Whether a region has a pulse over ``span``: finite (the region is not unusable) and with at
    most ``MAX_INVALID_FRACTION`` invalid samples (the same rule ``hr_series`` applies)."""
    return bool(
        np.isfinite(result.pulse[span]).all()
        and (~result.valid[span]).mean() <= MAX_INVALID_FRACTION
    )


def _skin_pulses(analysis: Analysis, span: slice) -> dict[str, np.ndarray]:
    """``{region: pulse[span]}`` for the skin regions usable over ``span``, in SKIN_REGIONS order."""
    out = {}
    for name in SKIN_REGIONS:
        result = analysis.regions.get(name)
        if result is not None and _usable(result, span):
            out[name] = result.pulse[span]
    return out


def _pairwise_corr(pulses: list[np.ndarray]) -> np.ndarray:
    """Pearson correlation of every pair of equal-length signals, shape ``(R*(R-1)/2,)``.

    Constant signals (std 0) have no defined correlation and are left out; fewer than two
    remaining signals give an empty array.
    """
    x = np.array([p for p in pulses if p.std() > 0])  # (R, L)
    if len(x) < 2:
        return np.empty(0)
    z = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)  # standardised rows
    corr = z @ z.T / x.shape[1]  # (R, R) correlation matrix
    return corr[np.triu_indices(len(x), k=1)]  # the pairs above the diagonal


def _band_power(x: np.ndarray, fs: float, f0: float) -> float:
    """Power of ``x`` within ``SNR_FUNDAMENTAL_HALFWIDTH_HZ`` of ``f0`` (the heart-rate peak)."""
    freqs, power = power_spectrum(x, fs)
    return float(power[np.abs(freqs - f0) <= SNR_FUNDAMENTAL_HALFWIDTH_HZ].sum())


def _window_corr_stats(analysis: Analysis, windows: list[_Window]) -> tuple[list, list]:
    """Per window: (median, min) pairwise correlation of the usable skin-region pulses."""
    medians, minima = [], []
    for w in windows:
        corr = _pairwise_corr(list(_skin_pulses(analysis, w.span).values()))
        if corr.size:
            medians.append(np.median(corr))
            minima.append(corr.min())
    return medians, minima


# --- group B: spatial agreement ----------------------------------------------------------------


def _coherence(analysis: Analysis) -> float:
    """``b_coherence``: mean over skin-region pairs of the debiased band coherence at the heart
    rate (rulings P2-R8, P2-R9).

    Coherence is a frequency-domain correlation: near ``f0``, does region ``y`` follow region
    ``x`` with a *consistent* phase and amplitude ratio from window to window? It is pooled over
    several windows, and overlapping windows would share samples and fake that consistency, so
    this feature alone uses NON-overlapping windows: starts every window length from the clip
    start, each kept when the face is usable there and has a finite heart rate. With ``k`` such
    windows, chance alone gives about ``1/k``; ``debias_coherence`` removes that floor so clips of
    different length are comparable. Needs ``k >= MIN_COHERENCE_WINDOWS`` for a pair (else that
    pair is skipped; NaN when no pair qualifies). Coherence needs one frequency, so ``f0`` is the
    median face heart rate over the kept windows.
    """
    fs = analysis.fs
    win = _window_length(analysis)
    face = analysis.face
    spans, bpms = [], []
    for start in range(0, len(analysis.t) - win + 1, win):
        span = slice(start, start + win)
        if _usable(face, span):
            # Same estimator, window and validity rule as the sliding face series.
            bpm = estimate_hr(face.pulse[span], fs).bpm
            if np.isfinite(bpm):
                spans.append(span)
                bpms.append(bpm)
    if len(spans) < MIN_COHERENCE_WINDOWS:
        return float("nan")
    f0 = float(np.median(bpms)) / 60.0

    values = []
    for a, b in combinations(SKIN_REGIONS, 2):
        ra, rb = analysis.regions.get(a), analysis.regions.get(b)
        if ra is None or rb is None:
            continue
        used = [s for s in spans if _usable(ra, s) and _usable(rb, s)]
        if len(used) < MIN_COHERENCE_WINDOWS:
            continue
        raw = band_coherence([ra.pulse[s] for s in used], [rb.pulse[s] for s in used], fs, f0)
        values.append(debias_coherence(raw, len(used)))
    return _mean(values)


def group_b(analysis: Analysis) -> dict[str, float]:
    """Spatial-agreement features (see ``FEATURE_GROUPS["B"]``); NaN where not computable.

    Do the skin regions carry the same heartbeat? In a real face one heart drives every region,
    so their pulses correlate, agree on the heart rate, and stay coherent at that rate. Generator
    noise or independent flicker per region does not.

    What B does NOT do: penalise an inherited pulse. A pulse pasted uniformly onto the whole face
    is shared perfectly, so its regions are as alike as a real face's, or more. On the synthetic
    data inherited faces score slightly higher (``b_corr_min`` about 0.59 vs 0.50), mainly
    because real regions with a weaker pulse (chin, nose) have a lower SNR; any separation of
    real from inherited by B comes from that, not from a missing shared heartbeat.

    * ``b_corr_median`` / ``b_corr_min``: per valid window, the median and the minimum Pearson
      correlation over all pairs of usable skin-region pulses; each then median over windows.
    * ``b_hr_spread``: per valid window, the std of the region bpm values (regions with a valid
      bpm in that window, at least two); median over windows (bpm).
    * ``b_coherence``: see ``_coherence`` (non-overlapping windows, debiased; NaN for clips with
      fewer than two such windows, e.g. under 8 s with 4 s windows).
    """
    windows = _valid_windows(analysis)
    corr_median, corr_min = _window_corr_stats(analysis, windows)
    skin = [analysis.regions[r] for r in SKIN_REGIONS if r in analysis.regions]
    hr_spread = [_std([r.series.bpm[w.index] for r in skin]) for w in windows]
    return {
        "b_corr_median": _median(corr_median),
        "b_corr_min": _median(corr_min),
        "b_hr_spread": _median(hr_spread),
        "b_coherence": _coherence(analysis),
    }


# --- group C: physiological consistency --------------------------------------------------------


def group_c(analysis: Analysis) -> dict[str, float]:
    """Physiological-consistency features (see ``FEATURE_GROUPS["C"]``); NaN where not computable.

    Per valid window (face heart rate ``f0``), on the usable skin-region pulses:

    * ``c_lag_<region>`` (the 6 regions other than ``REF_REGION``): ``phase_lag_s(ref, region,
      f0)``, the time by which the region's pulse trails the forehead centre's (positive = later),
      median over windows. Blood reaches different parts of the face at slightly different times;
      a pulse pasted onto the whole face arrives everywhere at once.
    * ``c_lag_stability``: per region, the std over windows of its lag; mean over regions. A real
      pulse wave keeps its timing; noise-driven lags wander.
    * ``c_amp_<region>`` (all 7 skin regions): the region's amplitude at the heart rate,
      ``sqrt(power within f0 +- SNR_FUNDAMENTAL_HALFWIDTH_HZ)``, divided by the sum of that over
      the 7 skin regions, so each window's map is a set of shares summing to 1; median over
      windows. ``analysis.regions[r].pulse`` comes from ``pulse_from_rgb`` on the region's own
      colour, which normalises by that region's mean colour, so the map compares pulse strength,
      not skin brightness. Only windows where all 7 skin regions are usable count (a share of an
      incomplete sum would be silently inflated), so one missing region makes all 7 NaN.
    * ``c_harmonic_face``: ``harmonic_ratio`` (power at ``2 f0`` over power at ``f0``) of the face
      pulse, median over windows: the waveform shape of a heartbeat (sharp systolic upstroke)
      has a second harmonic. ``c_harmonic_spread``: each skin region's median harmonic ratio,
      then the std across regions; one heart should give one waveform shape everywhere.
    * ``c_control_ratio``: power at ``f0`` of the control (background) region's pulse signal
      divided by the face's, median over windows. Background does not pulse, so a high value
      means the "pulse" is really lighting or compression flicker affecting the whole frame. NaN
      when the control region is unusable.
    * ``c_corr_stability``: std over windows of the per-window median pairwise correlation.

    **Rolling-shutter caveat.** Most cameras read the sensor row by row, so a lower row (the chin)
    is captured a few milliseconds after a higher one (the forehead). Part of a measured lag is
    therefore this readout time between vertical positions, not blood flow. The offset is the
    same for every face filmed with that camera, so the learned real-face reference
    (``reference.RealFaceReference``, fitted on training reals) absorbs any camera-consistent
    offset; what remains informative is the deviation from the real-face pattern.
    """
    fs = analysis.fs
    windows = _valid_windows(analysis)
    control = analysis.regions.get(CONTROL)
    lags: dict[str, list[float]] = {r: [] for r in LAG_REGIONS}
    amps: dict[str, list[float]] = {r: [] for r in SKIN_REGIONS}
    harmonics: dict[str, list[float]] = {r: [] for r in SKIN_REGIONS}
    harmonic_face, control_ratio = [], []

    for w in windows:
        pulses = _skin_pulses(analysis, w.span)
        if REF_REGION in pulses:
            for r in LAG_REGIONS:
                if r in pulses:
                    lags[r].append(phase_lag_s(pulses[REF_REGION], pulses[r], fs, w.f0))
        if len(pulses) == len(SKIN_REGIONS):
            amplitude = {r: np.sqrt(_band_power(p, fs, w.f0)) for r, p in pulses.items()}
            total = sum(amplitude.values())
            if total > 0:
                for r in SKIN_REGIONS:
                    amps[r].append(amplitude[r] / total)
        for r, p in pulses.items():
            harmonics[r].append(harmonic_ratio(p, fs, w.f0))

        face_pulse = analysis.face.pulse[w.span]
        harmonic_face.append(harmonic_ratio(face_pulse, fs, w.f0))
        if control is not None and _usable(control, w.span):
            face_power = _band_power(face_pulse, fs, w.f0)
            if face_power > 0:
                control_ratio.append(_band_power(control.pulse[w.span], fs, w.f0) / face_power)

    corr_median, _ = _window_corr_stats(analysis, windows)
    out: dict[str, float] = {f"c_lag_{r}": _median(lags[r]) for r in LAG_REGIONS}
    out["c_lag_stability"] = _mean([_std(lags[r]) for r in LAG_REGIONS])
    out.update({f"c_amp_{r}": _median(amps[r]) for r in SKIN_REGIONS})
    out["c_harmonic_face"] = _median(harmonic_face)
    out["c_harmonic_spread"] = _std([_median(harmonics[r]) for r in SKIN_REGIONS])
    out["c_control_ratio"] = _median(control_ratio)
    out["c_corr_stability"] = _std(corr_median)
    return out


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
    does (face unusable, clip shorter than ``SETTINGS.min_seconds``).
    """
    unknown = sorted(set(groups) - set(FEATURE_GROUPS))
    if unknown:
        raise ValueError(f"unknown feature group(s) {unknown}; choose from {list(FEATURE_GROUPS)}")
    analysis = analyze(
        traces,
        method,
        window_s=SETTINGS.feature_window_s,
        step_s=SETTINGS.feature_step_s,
    )
    out: dict[str, float] = {}
    if "A" in groups:
        out.update(group_a(analysis))
    if "B" in groups:
        out.update(group_b(analysis))
    if "C" in groups:
        out.update(group_c(analysis))
    if "D" in groups:
        out.update(group_d(traces, analysis))
    return out
