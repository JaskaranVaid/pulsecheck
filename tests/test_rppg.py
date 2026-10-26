import numpy as np
import pytest

from pulsecheck.dsp import bandpass, estimate_hr
from pulsecheck.rppg import METHODS, green, pos
from synth import synth_rgb


@pytest.mark.parametrize("name", ["green", "chrom", "pos"])
def test_all_methods_find_clean_pulse(name):
    _t, rgb = synth_rgb(72, 20)
    assert abs(estimate_hr(bandpass(METHODS[name](rgb, 30), 30), 30).bpm - 72) < 2


@pytest.mark.parametrize("name", ["chrom", "pos"])
def test_chrom_pos_cancel_in_band_lighting_flicker(name):
    _t, rgb = synth_rgb(72, 20, intensity_noise=(1.8, 0.01))  # 108-bpm flicker, 5x the pulse
    assert abs(estimate_hr(bandpass(METHODS[name](rgb, 30), 30), 30).bpm - 72) < 2


def test_green_is_fooled_by_lighting_flicker():
    """Documents WHY we use POS: the green channel alone locks onto the flicker."""
    _t, rgb = synth_rgb(72, 20, intensity_noise=(1.8, 0.01))
    assert abs(estimate_hr(bandpass(green(rgb, 30), 30), 30).bpm - 108) < 2


def test_pos_output_shape_and_zero_mean():
    _t, rgb = synth_rgb(72, 10)
    h = pos(rgb, 30)
    assert h.shape == (300,) and abs(h.mean()) < 1e-9


@pytest.mark.parametrize("name", ["green", "chrom", "pos"])
@pytest.mark.parametrize(
    "bad",
    [
        np.ones(300),  # 1-D
        np.ones((300, 2)),  # two channels
        np.ones((300, 4)),  # four channels
        np.ones((3, 100, 3)),  # 3-D
    ],
    ids=["1d", "2ch", "4ch", "3d"],
)
def test_rgb_must_be_n_by_3(name, bad):
    with pytest.raises(ValueError):
        METHODS[name](bad, 30)


def test_pos_rejects_signal_shorter_than_window():
    # POS window is round(1.6 * 30) = 48 samples; 47 is too short.
    rgb = np.ones((47, 3)) + np.random.default_rng(0).normal(0, 0.01, (47, 3))
    with pytest.raises(ValueError, match="signal shorter than POS window"):
        pos(rgb, 30)
