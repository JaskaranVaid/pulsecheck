import numpy as np
import pytest

from pulsecheck.config import SETTINGS
from pulsecheck.extract import RegionTraces
from pulsecheck.pipeline import Analysis, analyze
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
