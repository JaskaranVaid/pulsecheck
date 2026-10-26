import numpy as np
import pytest

from pulsecheck.config import SETTINGS
from pulsecheck.dsp import hr_series
from pulsecheck.extract import RegionTraces
from pulsecheck.pipeline import Analysis, analyze, face_signal, pulse_from_rgb
from pulsecheck.regions import ALL_REGIONS, CONTROL, SKIN_REGIONS
from synth import synth_rgb

FS = 30


def make_traces(bpm=72, seconds=20, fs=FS) -> RegionTraces:
    """Same pulse in every skin region (own sensor noise); pulse-free noise in the control."""
    rgb = []
    for i, name in enumerate(ALL_REGIONS):
        amp = 0.0 if name == CONTROL else 0.002
        t, x = synth_rgb(bpm, seconds, fs, pulse_amp=amp, seed=i)
        rgb.append(x)
    n = len(t)
    return RegionTraces(
        rgb=np.stack(rgb, axis=1).astype(np.float32),
        t=t,
        region_names=ALL_REGIONS,
        source="synthetic",
        native_fps=float(fs),
        face_found=np.ones(n, bool),
        face_width_px=np.full(n, 200, np.float32),
        motion=np.full(n, 0.001, np.float32),
    )


def test_analyze_finds_hr_in_every_skin_region_and_face():
    a = analyze(make_traces(72))
    assert isinstance(a, Analysis) and a.method == "pos" and a.fs == SETTINGS.fs
    assert abs(a.face.hr.bpm - 72) < 2
    for n in SKIN_REGIONS:
        assert abs(a.regions[n].hr.bpm - 72) < 3
    assert a.regions[CONTROL].hr.snr_db < a.face.hr.snr_db - 6
    assert set(a.regions) == set(ALL_REGIONS)
    assert a.quality["face_found_ratio"] == 1.0


@pytest.mark.parametrize("method", ["green", "chrom", "pos"])
def test_analyze_every_method(method):
    a = analyze(make_traces(90), method=method)
    assert abs(a.face.hr.bpm - 90) < 3
    assert a.method == method


def test_analysis_shapes_share_one_grid():
    a = analyze(make_traces(72, seconds=12))
    assert a.t.shape == a.valid.shape and a.valid.all()
    assert a.t[0] == 0 and np.allclose(np.diff(a.t), 1 / FS)
    grids = [r.series.t_center for r in a.regions.values()] + [a.face.series.t_center]
    for g in grids:
        np.testing.assert_array_equal(g, grids[0])
    for r in [*a.regions.values(), a.face]:
        assert r.pulse.shape == a.t.shape


def test_analyze_short_clip_raises():
    with pytest.raises(ValueError, match="too short"):
        analyze(make_traces(seconds=2))


def test_analyze_unknown_method_raises():
    with pytest.raises(KeyError):
        analyze(make_traces(), method="magic")


def test_unusable_control_does_not_crash():
    tr = make_traces(72)
    tr.rgb[:, ALL_REGIONS.index(CONTROL), :] = np.nan
    a = analyze(tr)
    c = a.regions[CONTROL]
    assert np.isnan(c.hr.bpm) and np.isnan(c.hr.snr_db) and np.isnan(c.hr.peak_hz)
    assert np.isnan(c.pulse).all() and c.pulse.shape == a.t.shape
    assert np.isnan(c.series.bpm).all() and np.isnan(c.series.snr_db).all()
    np.testing.assert_array_equal(c.series.t_center, a.face.series.t_center)
    assert abs(a.face.hr.bpm - 72) < 2


def test_region_with_non_positive_channel_mean_is_nan_not_crash():
    tr = make_traces(72)
    tr.rgb[:, ALL_REGIONS.index("nose"), 0] = 0.0  # black red channel -> zero mean
    a = analyze(tr)
    assert np.isnan(a.regions["nose"].hr.bpm)
    assert abs(a.regions["chin"].hr.bpm - 72) < 3


def test_region_with_single_valid_sample_is_nan():
    tr = make_traces(72)
    tr.rgb[1:, ALL_REGIONS.index("chin"), :] = np.nan
    a = analyze(tr)
    assert np.isnan(a.regions["chin"].hr.bpm)


def test_gap_in_one_region_marks_windows_nan_but_keeps_grid():
    tr = make_traces(72, seconds=20)
    tr.rgb[200:400, ALL_REGIONS.index("cheek_left"), :] = np.nan  # ~6.7 s hole in one region
    a = analyze(tr)
    s = a.regions["cheek_left"].series
    assert np.isnan(s.bpm).any() and np.isfinite(s.bpm).any()
    assert len(s.bpm) == len(a.face.series.bpm)
    assert a.valid.all()  # the face itself is fine


def test_unusable_face_raises():
    tr = make_traces(72)
    for n in SKIN_REGIONS:
        tr.rgb[:, ALL_REGIONS.index(n), :] = np.nan
    with pytest.raises(ValueError):
        analyze(tr)


def test_irregular_timestamps_with_dropped_frames():
    tr = make_traces(72, seconds=20)
    keep = np.ones(len(tr.t), bool)
    keep[::7] = False
    sub = RegionTraces(
        rgb=tr.rgb[keep],
        t=tr.t[keep],
        region_names=tr.region_names,
        source="s",
        native_fps=30.0,
        face_found=tr.face_found[keep],
        face_width_px=tr.face_width_px[keep],
        motion=tr.motion[keep],
    )
    a = analyze(sub)
    assert abs(a.face.hr.bpm - 72) < 3


def test_leading_face_loss_keeps_grid_anchored_to_video_start():
    clean = analyze(make_traces(72, seconds=20))
    tr = make_traces(72, seconds=20)
    tr.rgb[:30] = np.nan  # no face for the first 1.0 s
    a = analyze(tr)
    assert a.t[0] == 0 and len(a.t) == len(clean.t)
    assert not a.valid[:30].any() and a.valid[30:].all()
    ref = hr_series(np.sin(2 * np.pi * 1.2 * np.arange(600) / FS), FS)
    assert ref.t_center[0] == 4.0
    np.testing.assert_allclose(a.face.series.t_center, ref.t_center)
    np.testing.assert_allclose(a.regions["nose"].series.t_center, ref.t_center)
    # 30 of the first window's 240 samples are invalid (12.5%): under the 20% limit, so it
    # still gets an estimate; later windows are clean
    assert np.nanmax(np.abs(a.face.series.bpm[1:] - 72)) < 3 and np.isfinite(a.face.series.bpm).all()


def test_long_leading_face_loss_makes_first_window_nan():
    tr = make_traces(72, seconds=20)
    tr.rgb[:60] = np.nan  # 2.0 s without a face = 25% of the first 8 s window
    a = analyze(tr)
    assert a.t[0] == 0 and not a.valid[:60].any()
    assert a.face.series.t_center[0] == 4.0
    assert np.isnan(a.face.series.bpm[0]) and np.isnan(a.face.series.snr_db[0])
    assert np.nanmax(np.abs(a.face.series.bpm[1:] - 72)) < 3 and np.isfinite(a.face.series.bpm[1:]).all()


def test_trailing_face_loss_is_flagged_invalid():
    tr = make_traces(72, seconds=20)
    tr.rgb[-90:] = np.nan
    a = analyze(tr)
    assert len(a.t) == 600 and not a.valid[-89:].any() and a.valid[:500].all()
    assert np.isnan(a.face.series.bpm[-1])


def test_face_signal_nan_mean_over_skin_regions_only():
    rgb = np.zeros((3, len(ALL_REGIONS), 3))
    rgb[:, : len(SKIN_REGIONS), :] = 10.0
    rgb[:, -1, :] = 1000.0  # control region must not contribute
    rgb[1, :3, :] = np.nan  # some skin regions missing: mean of the rest
    rgb[2, : len(SKIN_REGIONS), :] = np.nan  # no skin at all -> NaN
    out = face_signal(rgb, ALL_REGIONS)
    assert out.shape == (3, 3)
    assert np.allclose(out[:2], 10.0) and np.isnan(out[2]).all()


def test_pulse_from_rgb_rejects_unusable_signals():
    t, rgb = synth_rgb(72, 10.0)
    pulse = pulse_from_rgb(rgb, "pos", 30.0)
    assert pulse is not None and pulse.shape == (len(t),)
    assert pulse_from_rgb(np.zeros_like(rgb), "pos", 30.0) is None  # non-positive mean colour
    bad = rgb.copy()
    bad[5, 0] = np.nan
    assert pulse_from_rgb(bad, "green", 30.0) is None  # non-finite
