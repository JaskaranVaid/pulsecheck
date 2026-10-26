import numpy as np
import pytest

from pulsecheck.dsp import (
    band_coherence,
    bandpass,
    cross_spectrum,
    debias_coherence,
    estimate_hr,
    harmonic_ratio,
    hr_series,
    phase_lag_s,
    power_spectrum,
    resample_uniform,
)
from synth import synth_rgb


def test_bandpass_keeps_72bpm_rejects_out_of_band():
    fs = 30
    t = np.arange(0, 20, 1 / fs)
    keep = bandpass(np.sin(2 * np.pi * 1.2 * t), fs)
    assert keep[60:-60].std() > 0.6
    for f in (0.15, 5.0):
        assert bandpass(np.sin(2 * np.pi * f * t), fs)[60:-60].std() < 0.1


@pytest.mark.parametrize("bpm", [48, 72, 110, 180])
def test_estimate_hr_on_pure_pulse(bpm):
    _t, rgb = synth_rgb(bpm, 20)
    r = estimate_hr(bandpass(rgb[:, 1] - rgb[:, 1].mean(), 30), 30)
    assert abs(r.bpm - bpm) < 2 and r.snr_db > 5


def test_snr_drops_with_noise():
    rng = np.random.default_rng(0)
    _t, rgb = synth_rgb(72, 20)
    g = rgb[:, 1] - rgb[:, 1].mean()
    clean = estimate_hr(bandpass(g, 30), 30).snr_db
    noisy = estimate_hr(bandpass(g + rng.normal(0, 3 * g.std(), g.size), 30), 30).snr_db
    assert noisy < clean - 3


@pytest.mark.parametrize(
    "x",
    [
        np.zeros(600),
        np.full(600, 5.0),
        np.full(150, 0.1),
        np.full(150, 7.7),
        bandpass(np.full(300, 0.1), 30),
        bandpass(np.full(300, 7.7), 30),
    ],
)
def test_estimate_hr_flat_input_is_nan_not_a_crash(x):
    r = estimate_hr(x, 30)
    assert np.isnan(r.bpm) and np.isnan(r.peak_hz) and np.isnan(r.snr_db)


def test_estimate_hr_tiny_amplitude_pulse_still_works():
    t = np.arange(0, 20, 1 / 30)
    r = estimate_hr(1e-6 * np.sin(2 * np.pi * 1.2 * t), 30)
    assert abs(r.bpm - 72) < 2 and r.snr_db > 5


def test_resample_jittered_24fps_recovers_hr():
    rng = np.random.default_rng(1)
    t = np.cumsum(rng.uniform(0.8, 1.2, 24 * 15) / 24)
    x = np.sin(2 * np.pi * 1.25 * t)
    tu, xu, valid = resample_uniform(t, x, fs=30)
    assert valid.all() and np.allclose(np.diff(tu), 1 / 30)
    assert abs(estimate_hr(bandpass(xu, 30), 30).bpm - 75) < 2


def test_resample_marks_long_gap_invalid_and_fills_short_gap():
    t = np.arange(0, 10, 1 / 30)
    x = np.sin(t)
    x[30:39] = np.nan
    x[150:180] = np.nan  # 0.3 s and 1.0 s gaps
    tu, xu, valid = resample_uniform(t, x)
    assert not np.isnan(xu).any()
    assert valid[(tu > 1.05) & (tu < 1.25)].all()
    assert not valid[(tu > 5.1) & (tu < 5.9)].any()


def test_resample_handles_multichannel():
    t = np.arange(0, 5, 1 / 25)
    x = np.stack([np.sin(t), np.cos(t), t], axis=1)
    x[10, 1] = np.nan  # any NaN channel drops the whole sample
    tu, xu, valid = resample_uniform(t, x, fs=30)
    assert xu.shape == (len(tu), 3) and valid.all() and not np.isnan(xu).any()
    assert np.allclose(xu[:, 2], tu)


def test_resample_t_out_values_match_on_given_grid():
    rng = np.random.default_rng(2)
    t = np.cumsum(rng.uniform(0.8, 1.2, 300) / 25)
    x = np.stack([np.sin(2 * t), t], axis=1)
    grid = np.arange(0.0, t[-1], 1 / 30)
    tu, xu, valid = resample_uniform(t, x, t_out=grid)
    assert np.array_equal(tu, grid)
    inside = (grid >= t[0]) & (grid <= t[-1])
    assert valid[inside].all()
    assert np.allclose(xu[inside, 1], grid[inside])  # linear signal is reproduced exactly
    assert np.allclose(xu[inside, 0], np.sin(2 * grid[inside]), atol=0.02)


def test_resample_t_out_flags_span_and_long_gaps():
    t = np.arange(0, 10, 1 / 30)
    x = np.sin(t)
    x[:30] = np.nan  # no data in the first second
    x[-30:] = np.nan  # ... and the last
    x[150:180] = np.nan  # 1.0 s hole in the middle
    grid = np.arange(0, 10, 1 / 30)
    tu, xu, valid = resample_uniform(t, x, t_out=grid)
    assert not np.isnan(xu).any() and len(tu) == len(grid)
    assert not valid[:30].any() and not valid[-29:].any()  # outside the valid span
    assert valid[30] and valid[60:140].all()  # first valid sample and good data
    assert not valid[(grid > 5.1) & (grid < 5.9)].any()  # long gap
    assert np.allclose(xu[:30], xu[30])  # edge values held, not extrapolated


def test_resample_t_out_default_behaviour_unchanged():
    t = np.arange(0, 5, 1 / 25)
    a = resample_uniform(t, np.sin(t), fs=30)
    b = resample_uniform(t, np.sin(t), fs=30, t_out=None)
    for u, v in zip(a, b, strict=True):
        assert np.array_equal(u, v)


@pytest.mark.parametrize("x", [np.full(50, np.nan), np.r_[1.0, np.full(49, np.nan)]])
def test_resample_needs_two_valid_samples(x):
    with pytest.raises(ValueError, match="not enough valid samples"):
        resample_uniform(np.arange(50) / 30, x)


def test_hr_series_windows_and_short_clip_rules():
    _t, rgb = synth_rgb(72, 20)
    p = bandpass(rgb[:, 1] - rgb[:, 1].mean(), 30)
    s = hr_series(p, 30)
    assert len(s.bpm) == 13 and np.nanmax(np.abs(s.bpm - 72)) < 3
    assert np.allclose(s.t_center, 4.0 + np.arange(13))
    s5 = hr_series(p[: 5 * 30], 30)
    assert len(s5.bpm) == 1 and abs(s5.bpm[0] - 72) < 4
    assert np.allclose(s5.t_center, 2.5)
    with pytest.raises(ValueError, match="too short"):
        hr_series(p[: 2 * 30], 30)


def test_hr_series_invalid_window_is_nan():
    _t, rgb = synth_rgb(72, 20)
    p = bandpass(rgb[:, 1] - rgb[:, 1].mean(), 30)
    valid = np.ones_like(p, bool)
    valid[0:150] = False
    s = hr_series(p, 30, valid=valid)
    assert np.isnan(s.bpm[0]) and np.isnan(s.snr_db[0])
    assert not np.isnan(s.bpm[-1])


def test_cross_spectrum_shapes_and_autospectrum_is_real():
    fs = 30.0
    x = np.sin(2 * np.pi * 1.2 * np.arange(0, 8, 1 / fs))
    freqs, sxy = cross_spectrum(x, x, fs)
    assert freqs.shape == sxy.shape and freqs.size == 4096 // 2 + 1
    assert np.iscomplexobj(sxy) and np.allclose(sxy.imag, 0) and (sxy.real >= 0).all()
    pf, pw = power_spectrum(x, fs)
    assert np.allclose(freqs, pf) and np.allclose(sxy.real, pw)
    with pytest.raises(ValueError):
        cross_spectrum(x, x[:-1], fs)


def test_phase_lag_recovers_known_delay():
    fs, f0 = 30.0, 1.2
    t = np.arange(0, 8, 1 / fs)
    x = np.sin(2 * np.pi * f0 * t)
    y = np.sin(2 * np.pi * f0 * (t - 0.05))  # y lags x by 50 ms
    assert phase_lag_s(x, y, fs, f0) == pytest.approx(0.05, abs=0.005)
    assert phase_lag_s(y, x, fs, f0) == pytest.approx(-0.05, abs=0.005)


def test_phase_lag_wraps_into_half_period():
    fs, f0 = 30.0, 1.2
    t = np.arange(0, 8, 1 / fs)
    x = np.sin(2 * np.pi * f0 * t)
    y = np.sin(2 * np.pi * f0 * (t - 0.7))  # 0.7 s > half period (0.4167 s); wraps to -0.1333 s
    assert phase_lag_s(x, y, fs, f0) == pytest.approx(0.7 - 1 / f0, abs=0.005)


def _windows(rng, shared, n_win=6, fs=30.0, f0=1.2, noise=0.3):
    t = np.arange(0, 4, 1 / fs)
    xs, ys = [], []
    for _ in range(n_win):
        phase = rng.uniform(0, 2 * np.pi)
        pulse = np.sin(2 * np.pi * f0 * t + phase) if shared else 0.0
        xs.append(pulse + rng.normal(0, noise, t.size))
        ys.append(pulse + rng.normal(0, noise, t.size))
    return xs, ys


def test_band_coherence_high_for_shared_low_for_independent():
    # Mean over seeds: a single independent draw exceeds 0.3 for ~14% of seeds with 6 windows.
    shared, indep = [], []
    for seed in range(50):
        rng = np.random.default_rng(seed)
        xs, ys = _windows(rng, shared=True)
        shared.append(band_coherence(xs, ys, 30.0, 1.2))
        xs, ys = _windows(rng, shared=False, noise=1.0)
        indep.append(band_coherence(xs, ys, 30.0, 1.2))
    assert np.mean(shared) > 0.9
    assert np.mean(indep) < 0.3


@pytest.mark.parametrize(
    ("c", "k", "expected"),
    [(1.0, 6, 1.0), (0.5, 2, 0.0), (0.25, 4, 0.0), (0.75, 3, 0.625), (0.9, 2, 0.8), (0.1, 6, -0.08)],
)
def test_debias_coherence_exact_values(c, k, expected):
    assert debias_coherence(c, k) == pytest.approx(expected)


def test_debias_coherence_nan_cases():
    assert np.isnan(debias_coherence(np.nan, 6))
    assert np.isnan(debias_coherence(0.5, 1))
    assert np.isnan(debias_coherence(0.5, 0))


def _spread(d):
    return max(d.values()) - min(d.values())


def test_debiased_coherence_removes_window_count_dependence():
    raw_null, deb_null, deb_shared = {}, {}, {}
    for k in (2, 3, 6):
        raw, deb, sh = [], [], []
        for seed in range(50):
            rng = np.random.default_rng(seed)
            xs, ys = _windows(rng, shared=False, n_win=k, noise=1.0)
            c = band_coherence(xs, ys, 30.0, 1.2)
            raw.append(c)
            deb.append(debias_coherence(c, k))
            xs, ys = _windows(rng, shared=True, n_win=k)
            sh.append(debias_coherence(band_coherence(xs, ys, 30.0, 1.2), k))
        raw_null[k], deb_null[k], deb_shared[k] = np.mean(raw), np.mean(deb), np.mean(sh)
    for k in (2, 3, 6):
        assert abs(deb_null[k]) < 0.15
        assert deb_shared[k] > 0.8
    assert _spread(deb_null) < 0.5 * _spread(raw_null)


def test_band_coherence_is_bounded_and_validates_lengths():
    rng = np.random.default_rng(1)
    xs, ys = _windows(rng, shared=True)
    assert 0.0 <= band_coherence(xs, ys, 30.0, 1.2) <= 1.0
    with pytest.raises(ValueError):
        band_coherence(xs, ys[:-1], 30.0, 1.2)


def test_harmonic_ratio_pure_vs_harmonic():
    fs, f0 = 30.0, 1.2
    t = np.arange(0, 10, 1 / fs)
    pure = np.sin(2 * np.pi * f0 * t)
    rich = pure + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
    assert harmonic_ratio(pure, fs, f0) < 0.01
    assert harmonic_ratio(rich, fs, f0) == pytest.approx(0.09, abs=0.03)


def test_helpers_nan_safe():
    fs = 30.0
    t = np.arange(0, 8, 1 / fs)
    x = np.sin(2 * np.pi * 1.2 * t)
    flat = np.full_like(x, 3.0)
    assert np.isnan(phase_lag_s(x, x, fs, np.nan))
    assert np.isnan(phase_lag_s(flat, x, fs, 1.2))
    assert np.isnan(phase_lag_s(x, flat, fs, 1.2))
    assert np.isnan(harmonic_ratio(x, fs, np.nan))
    assert np.isnan(harmonic_ratio(flat, fs, 1.2))
    assert np.isnan(harmonic_ratio(np.zeros(0), fs, 1.2))
    assert np.isnan(band_coherence([], [], fs, 1.2))
    assert np.isnan(band_coherence([x], [x], fs, np.nan))
    assert np.isnan(band_coherence([flat], [flat], fs, 1.2))
    assert np.isnan(band_coherence([x], [flat], fs, 1.2))
