import numpy as np
import pytest

from pulsecheck.dsp import bandpass, estimate_hr, hr_series, resample_uniform
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
