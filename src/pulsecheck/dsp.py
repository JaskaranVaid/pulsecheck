"""Signal processing: uniform resampling, bandpass, power spectrum, heart rate and SNR.

Pure NumPy/SciPy; no video or face-tracking dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from pulsecheck.config import SETTINGS

MAX_INVALID_FRACTION = 0.2  # a window with more invalid samples than this is skipped
FLAT_REL_TOL = 1e-9  # ptp below this fraction of max|x| counts as constant
FLAT_ABS_TOL = 1e-10  # ptp below this absolute value counts as constant
SPAN_TOL_S = 1e-9  # slack when testing whether a grid point lies within the valid span
SNR_RANGE_HZ = (0.5, 5.0)  # frequency range over which signal and noise power are compared
SNR_FUNDAMENTAL_HALFWIDTH_HZ = 0.1
SNR_HARMONIC_HALFWIDTH_HZ = 0.2


def _next_pow2(n: int) -> int:
    return 1 << max(int(n) - 1, 0).bit_length()


def resample_uniform(
    t: np.ndarray,
    x: np.ndarray,
    fs: float = SETTINGS.fs,
    max_gap_s: float = SETTINGS.max_gap_s,
    t_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly resample irregular samples onto a uniform ``fs`` grid.

    ``x`` has shape ``(T, ...)``. Samples with a NaN in any channel are dropped first. By default
    the grid runs from the first to the last valid timestamp. Grid points that fall inside a gap
    longer than ``max_gap_s`` are still filled (so filters can run) but flagged ``valid=False``.

    With ``t_out`` the samples are interpolated onto that grid instead (``fs`` is then unused).
    Points before the first or after the last valid sample are filled with the nearest value and
    flagged ``valid=False``, in addition to the long-gap rule.

    Returns ``(t_u, x_u, valid)``.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(t) & np.isfinite(x.reshape(len(x), -1)).all(axis=1)
    if finite.sum() < 2:
        raise ValueError("not enough valid samples")
    tv, xv = t[finite], x[finite]

    if t_out is None:
        n = int(np.floor((tv[-1] - tv[0]) * fs + 1e-9)) + 1
        t_u = tv[0] + np.arange(n) / fs
    else:
        t_u = np.asarray(t_out, dtype=float)
        n = len(t_u)

    flat = xv.reshape(len(xv), -1)
    x_u = np.stack([np.interp(t_u, tv, flat[:, c]) for c in range(flat.shape[1])], axis=1)
    x_u = x_u.reshape((n,) + xv.shape[1:])

    left = np.clip(np.searchsorted(tv, t_u, side="right") - 1, 0, len(tv) - 2)
    valid = (tv[left + 1] - tv[left]) <= max_gap_s
    if t_out is not None:
        valid &= (t_u >= tv[0] - SPAN_TOL_S) & (t_u <= tv[-1] + SPAN_TOL_S)
    return t_u, x_u, valid


def bandpass(
    x: np.ndarray,
    fs: float,
    low_hz: float = SETTINGS.band[0],
    high_hz: float = SETTINGS.band[1],
    order: int = SETTINGS.filter_order,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass along axis 0."""
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    padlen = min(3 * (2 * order + 1) * 2, len(x) - 1)
    return signal.sosfiltfilt(sos, x, axis=0, padlen=padlen)


def power_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed periodogram, zero-padded to at least 4096 points. Returns ``(freqs, power)``."""
    nfft = max(4096, _next_pow2(len(x)))
    return signal.periodogram(x, fs=fs, window="hann", nfft=nfft)


@dataclass(frozen=True)
class HRResult:
    bpm: float
    peak_hz: float
    snr_db: float


def estimate_hr(
    pulse: np.ndarray, fs: float, band: tuple[float, float] = SETTINGS.band
) -> HRResult:
    """Heart rate from the spectral peak inside ``band``, plus a de Haan-style SNR.

    SNR = 10 log10(signal / noise) where signal is the power within 0.1 Hz of the peak ``f0``
    and 0.2 Hz of ``2*f0``, and noise is all other power in 0.5-5 Hz. A flat (zero-power) input
    yields NaN for every field.
    """
    pulse = np.asarray(pulse, dtype=float)
    # A constant input leaves only floating-point residue after detrending (about 1e-16 of its
    # level, or ~1e-17 absolute once bandpassed), which would otherwise yield a confident-looking
    # peak. Treat the signal as flat when its peak-to-peak range is negligible relative to its
    # level, or below an absolute floor (for already zero-mean input such as bandpass output).
    spread = np.ptp(pulse) if pulse.size else 0.0
    if not spread > max(FLAT_REL_TOL * np.max(np.abs(pulse), initial=0.0), FLAT_ABS_TOL):
        return HRResult(np.nan, np.nan, np.nan)
    freqs, power = power_spectrum(pulse, fs)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not in_band.any() or not power[in_band].max() > 0:
        return HRResult(np.nan, np.nan, np.nan)
    f0 = float(freqs[in_band][np.argmax(power[in_band])])

    in_range = (freqs >= SNR_RANGE_HZ[0]) & (freqs <= SNR_RANGE_HZ[1])
    is_signal = (np.abs(freqs - f0) <= SNR_FUNDAMENTAL_HALFWIDTH_HZ) | (
        np.abs(freqs - 2 * f0) <= SNR_HARMONIC_HALFWIDTH_HZ
    )
    sig = power[in_range & is_signal].sum()
    noise = power[in_range & ~is_signal].sum()
    snr_db = 10 * np.log10(sig / noise) if sig > 0 and noise > 0 else np.nan
    return HRResult(bpm=f0 * 60.0, peak_hz=f0, snr_db=float(snr_db))


@dataclass
class HRSeries:
    t_center: np.ndarray
    bpm: np.ndarray
    snr_db: np.ndarray  # NaN for invalid windows


def hr_series(
    pulse: np.ndarray,
    fs: float,
    window_s: float = SETTINGS.window_s,
    step_s: float = SETTINGS.step_s,
    valid: np.ndarray | None = None,
) -> HRSeries:
    """Sliding-window heart rate. Windows with >20% invalid samples give NaN.

    A signal shorter than ``window_s`` (but at least ``SETTINGS.min_seconds``) yields a single
    window over the whole signal; anything shorter raises ``ValueError("clip too short")``.
    """
    n = len(pulse)
    if n < SETTINGS.min_seconds * fs:
        raise ValueError("clip too short")
    win = round(window_s * fs)
    step = max(round(step_s * fs), 1)
    if valid is None:
        valid = np.ones(n, dtype=bool)

    if n < win:
        starts, win = [0], n
    else:
        starts = range(0, n - win + 1, step)

    t_center, bpm, snr = [], [], []
    for s in starts:
        t_center.append((s + win / 2) / fs)
        if (~valid[s : s + win]).mean() > MAX_INVALID_FRACTION:
            bpm.append(np.nan)
            snr.append(np.nan)
            continue
        r = estimate_hr(pulse[s : s + win], fs)
        bpm.append(r.bpm)
        snr.append(r.snr_db)
    return HRSeries(np.array(t_center), np.array(bpm), np.array(snr))
