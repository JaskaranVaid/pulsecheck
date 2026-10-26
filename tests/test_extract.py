from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import FIXTURES, FakeTracker, write_video
from pulsecheck import extract
from pulsecheck.extract import NoFaceError, RegionTraces, cached_traces, extract_traces
from pulsecheck.regions import ALL_REGIONS

LM = np.load(FIXTURES / "landmarks_face.npy")
FPS = 30
W, H = 320, 400  # frame size; regions keep >= 20 px after scaling the 640x800 fixture


def _pulse_video(path: Path, seconds: float = 3.0, hz: float = 1.2) -> Path:
    """Uniform frames whose green value is 100 + 5*sin(2*pi*hz*t)."""
    frames = []
    for i in range(int(seconds * FPS)):
        g = 100 + 5 * np.sin(2 * np.pi * hz * i / FPS)
        frames.append(np.full((H, W, 3), (120, round(g), 180), np.uint8))
    return write_video(path, frames, FPS)


@pytest.fixture
def video(tmp_path):
    return _pulse_video(tmp_path / "with space" / "v.avi")


def test_traces_follow_color_modulation(video):
    tr = extract_traces(video, tracker=FakeTracker(LM))
    g = tr.rgb[:, ALL_REGIONS.index("nose"), 1]
    expected = 100 + 5 * np.sin(2 * np.pi * 1.2 * tr.t)
    assert np.corrcoef(g, expected)[0, 1] > 0.98  # MJPG is lossy; correlation, not equality
    assert tr.rgb.shape == (90, len(ALL_REGIONS), 3) and abs(tr.native_fps - 30) < 0.5
    assert tr.rgb.dtype == np.float32 and tr.t.dtype == np.float64
    assert tr.region_names == ALL_REGIONS
    assert tr.face_found.all()
    assert np.all(np.diff(tr.t) > 0)
    assert tr.t[1] - tr.t[0] == pytest.approx(1 / FPS, abs=1e-3)
    # red and blue channels are the constant colors we wrote (RGB order, not BGR)
    assert np.nanmean(tr.rgb[:, 0, 0]) == pytest.approx(180, abs=3)
    assert np.nanmean(tr.rgb[:, 0, 2]) == pytest.approx(120, abs=3)


def test_face_geometry_and_motion(video):
    tr = extract_traces(video, tracker=FakeTracker(LM))
    expected_width = np.ptp(LM[:, 0]) * W / 640
    np.testing.assert_allclose(tr.face_width_px, expected_width, rtol=1e-4)
    assert np.isnan(tr.motion[0])
    np.testing.assert_allclose(tr.motion[1:], 0.0, atol=1e-6)  # the fake face never moves


def test_motion_measures_displacement_relative_to_face_width(video):
    class Drifting(FakeTracker):
        def detect(self, frame_bgr, timestamp_ms):
            lm = super().detect(frame_bgr, timestamp_ms)
            return None if lm is None else lm + np.float32([self.calls, 0])  # +1 px/frame in x

    tr = extract_traces(video, tracker=Drifting(LM, lost_frames={5}))
    width = tr.face_width_px[1]
    assert tr.motion[1] == pytest.approx(1 / width, rel=1e-3)
    assert np.isnan(tr.motion[5]) and np.isnan(tr.motion[6])  # lost frame, then re-acquired
    assert tr.motion[7] == pytest.approx(1 / width, rel=1e-3)


def test_lost_frames_are_nan(video):
    tr = extract_traces(video, tracker=FakeTracker(LM, lost_frames={10, 11, 12}))
    assert np.isnan(tr.rgb[10:13]).all() and not tr.face_found[10:13].any()
    assert np.isnan(tr.face_width_px[10:13]).all()
    assert tr.face_found[:10].all() and tr.face_found[13:].all()
    assert tr.quality_summary()["face_found_ratio"] == pytest.approx(87 / 90)


def test_no_face_raises(video):
    with pytest.raises(NoFaceError):
        extract_traces(video, tracker=FakeTracker(LM, lost_frames=set(range(1000))))


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        extract_traces(Path("nope.avi"), tracker=FakeTracker(LM))


def test_unreadable_file_raises_value_error(tmp_path):
    bad = tmp_path / "garbage.avi"
    bad.write_bytes(b"this is not a video")
    with pytest.raises(ValueError, match="cannot open video"):
        extract_traces(bad, tracker=FakeTracker(LM))


def test_max_seconds_caps_frames(video):
    tr = extract_traces(video, max_seconds=1.0, tracker=FakeTracker(LM))
    assert tr.rgb.shape[0] == 30
    assert tr.t[-1] < 1.0


def test_max_seconds_none_reads_everything(video):
    assert extract_traces(video, max_seconds=None, tracker=FakeTracker(LM)).rgb.shape[0] == 90


def test_quality_summary(video):
    q = extract_traces(video, tracker=FakeTracker(LM)).quality_summary()
    assert set(q) == {"duration_s", "face_found_ratio", "median_face_width_px", "median_motion"}
    assert q["duration_s"] == pytest.approx(3.0, abs=0.05)
    assert q["face_found_ratio"] == 1.0
    assert q["median_face_width_px"] == pytest.approx(np.ptp(LM[:, 0]) * W / 640, rel=1e-4)
    assert q["median_motion"] == pytest.approx(0.0, abs=1e-6)


def test_debug_video_is_written_with_regions_drawn(video, tmp_path):
    debug = tmp_path / "out" / "debug.avi"
    tr = extract_traces(video, tracker=FakeTracker(LM), debug_video=debug)
    cap = cv2.VideoCapture(str(debug))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    assert len(frames) == tr.rgb.shape[0]
    # the source frames are flat color; outlines/labels make the debug frames non-uniform
    assert frames[0].std() > 1.0


def test_save_load_roundtrip(video, tmp_path):
    a = extract_traces(video, tracker=FakeTracker(LM, lost_frames={4}))
    a.save(tmp_path / "sub" / "t.npz")
    b = RegionTraces.load(tmp_path / "sub" / "t.npz")
    for name in ("rgb", "t", "face_found", "face_width_px", "motion"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        assert getattr(a, name).dtype == getattr(b, name).dtype
    assert a.region_names == b.region_names and isinstance(b.region_names, tuple)
    assert a.source == b.source and a.native_fps == b.native_fps


def test_cache_hit_and_invalidation(video, tmp_path, monkeypatch):
    monkeypatch.setenv("PULSECHECK_DATA", str(tmp_path / "data"))
    a = cached_traces(video, tracker=FakeTracker(LM))
    assert len(list((tmp_path / "data" / "cache" / "traces").glob("*.npz"))) == 1

    def boom(*args, **kwargs):
        raise AssertionError("cache miss: extraction ran again")

    monkeypatch.setattr(extract, "FaceTracker", boom)
    b = cached_traces(video, tracker=None)  # 2nd call must hit the cache
    np.testing.assert_array_equal(a.rgb, b.rgb)
    assert a.region_names == b.region_names

    # a different cap is a different cache entry
    short = cached_traces(video, max_seconds=1.0, tracker=FakeTracker(LM))
    assert short.rgb.shape[0] == 30
    assert len(list((tmp_path / "data" / "cache" / "traces").glob("*.npz"))) == 2


def test_corrupt_cache_entry_is_recomputed(video, tmp_path, monkeypatch):
    monkeypatch.setenv("PULSECHECK_DATA", str(tmp_path / "data"))
    cached_traces(video, tracker=FakeTracker(LM))
    (entry,) = (tmp_path / "data" / "cache" / "traces").glob("*.npz")
    entry.write_bytes(b"truncated")
    again = cached_traces(video, tracker=FakeTracker(LM))
    assert again.rgb.shape[0] == 90
    assert RegionTraces.load(entry).rgb.shape[0] == 90
