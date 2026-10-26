import subprocess
import sys
from pathlib import Path
from typing import ClassVar

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

    monkeypatch.setattr("pulsecheck.face.FaceTracker", boom)
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


def test_saved_traces_carry_the_schema_version(video, tmp_path):
    a = extract_traces(video, tracker=FakeTracker(LM))
    a.save(tmp_path / "t.npz")
    with np.load(tmp_path / "t.npz") as z:
        assert int(z["traces_version"]) == extract.TRACES_VERSION == 2


def _rewrite_npz(path: Path, version: int | None) -> None:
    """Rewrite a saved traces file with another (or no) ``traces_version``."""
    with np.load(path) as z:
        data = {k: z[k] for k in z.files if k != "traces_version"}
    if version is not None:
        data["traces_version"] = np.array(version)
    with open(path, "wb") as f:
        np.savez_compressed(f, **data)


@pytest.mark.parametrize("version", [None, 1, 99])
def test_load_rejects_missing_or_other_version(video, tmp_path, version):
    extract_traces(video, tracker=FakeTracker(LM)).save(tmp_path / "t.npz")
    _rewrite_npz(tmp_path / "t.npz", version)
    with pytest.raises(ValueError, match="version"):
        RegionTraces.load(tmp_path / "t.npz")


@pytest.mark.parametrize("version", [None, 1])
def test_cache_entry_with_old_version_is_a_miss(video, tmp_path, monkeypatch, version):
    monkeypatch.setenv("PULSECHECK_DATA", str(tmp_path / "data"))
    cached_traces(video, tracker=FakeTracker(LM))
    (entry,) = (tmp_path / "data" / "cache" / "traces").glob("*.npz")
    _rewrite_npz(entry, version)
    tracker = FakeTracker(LM)
    again = cached_traces(video, tracker=tracker)
    assert tracker.calls == 90  # recomputed, not loaded
    assert again.rgb.shape[0] == 90
    assert RegionTraces.load(entry).rgb.shape[0] == 90  # rewritten with the current version


def test_cache_key_includes_the_schema_version(video, tmp_path, monkeypatch):
    monkeypatch.setenv("PULSECHECK_DATA", str(tmp_path / "data"))
    before = extract._cache_path(video, 15.0)
    monkeypatch.setattr(extract, "TRACES_VERSION", extract.TRACES_VERSION + 1)
    assert extract._cache_path(video, 15.0) != before


_RealCapture = cv2.VideoCapture


class _CaptureProxy:
    """cv2.VideoCapture wrapper that lies about fps and/or POS_MSEC."""

    fps: float | None = None
    zero_pos_msec = False

    def __init__(self, *args):
        self._cap = _RealCapture(*args)

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS and self.fps is not None:
            return self.fps
        if prop == cv2.CAP_PROP_POS_MSEC and self.zero_pos_msec:
            return 0.0
        return self._cap.get(prop)

    def __getattr__(self, name):
        return getattr(self._cap, name)


class _RecordingTracker(FakeTracker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timestamps: list[int] = []

    def detect(self, frame_bgr, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        return super().detect(frame_bgr, timestamp_ms)


def _lying_capture(monkeypatch, **attrs):
    proxy = type("Proxy", (_CaptureProxy,), attrs)
    monkeypatch.setattr(extract.cv2, "VideoCapture", proxy)


def test_importing_extract_does_not_import_mediapipe():
    code = "import sys, pulsecheck.extract; sys.exit('mediapipe' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_cap_follows_timestamps_not_container_fps(video, monkeypatch):
    _lying_capture(monkeypatch, fps=120.0)  # fps metadata says 4x the real rate
    tr = extract_traces(video, max_seconds=1.0, tracker=FakeTracker(LM))
    assert tr.rgb.shape[0] == 30
    assert tr.t[-1] < 1.0


def test_missing_fps_defaults_to_30(video, monkeypatch):
    _lying_capture(monkeypatch, fps=0.0)
    tr = extract_traces(video, tracker=FakeTracker(LM))
    assert tr.native_fps == 30.0 and tr.rgb.shape[0] == 90


def test_unusable_container_timestamps_fall_back_to_frame_index(video, monkeypatch):
    _lying_capture(monkeypatch, zero_pos_msec=True)
    tracker = _RecordingTracker(LM)
    tr = extract_traces(video, max_seconds=1.0, tracker=tracker)
    assert tr.rgb.shape[0] == 30  # the cap still works on index / fps
    np.testing.assert_allclose(tr.t, np.arange(30) / tr.native_fps)
    assert tracker.timestamps == [round(i * 1000 / tr.native_fps) for i in range(30)]


def test_tracker_receives_container_timestamps_in_ms(video):
    tracker = _RecordingTracker(LM)
    tr = extract_traces(video, tracker=tracker)
    assert tracker.timestamps == [round(t * 1000) for t in tr.t]
    assert tracker.timestamps[0] == 0 and np.all(np.diff(tracker.timestamps) > 0)


def test_timestamps_keeps_increasing_container_values():
    t = extract._timestamps([0.0, 40.0, 80.0, 120.0], fps=30.0)
    np.testing.assert_allclose(t, [0.0, 0.04, 0.08, 0.12])


@pytest.mark.parametrize(
    "raw_ms",
    [[0.0, 0.0, 0.0, 0.0], [0.0, 33.0, 33.0, 100.0], [0.0, 66.0, 33.0, 100.0]],
    ids=["all-zero", "duplicate", "decreasing"],
)
def test_timestamps_fall_back_to_index_over_fps(raw_ms):
    t = extract._timestamps(raw_ms, fps=25.0)
    np.testing.assert_allclose(t, [0.0, 0.04, 0.08, 0.12])


class _FakeFaceTracker(FakeTracker):
    """Replaces pulsecheck.face.FaceTracker: records construction and closing."""

    instances: ClassVar[list]  # per-test subclass supplies fresh containers
    lost: ClassVar[set]

    def __init__(self, model_path=None):
        super().__init__(LM, lost_frames=type(self).lost)
        self.closed = False
        type(self).instances.append(self)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_face_tracker(monkeypatch):
    cls = type("FakeFaceTracker", (_FakeFaceTracker,), {"instances": [], "lost": set()})
    monkeypatch.setattr("pulsecheck.face.FaceTracker", cls)
    return cls


def test_own_tracker_is_closed_after_success(video, fake_face_tracker):
    tr = extract_traces(video)
    (made,) = fake_face_tracker.instances
    assert made.closed and made.calls == tr.rgb.shape[0]


def test_own_tracker_is_closed_after_error(video, fake_face_tracker):
    fake_face_tracker.lost = set(range(1000))
    with pytest.raises(NoFaceError):
        extract_traces(video)
    (made,) = fake_face_tracker.instances
    assert made.closed


def test_own_tracker_is_closed_when_detect_raises(video, fake_face_tracker):
    def explode(self, frame, ts):
        raise RuntimeError("boom")

    fake_face_tracker.detect = explode
    with pytest.raises(RuntimeError, match="boom"):
        extract_traces(video)
    assert fake_face_tracker.instances[0].closed


def test_no_tracker_constructed_for_missing_or_unreadable_video(tmp_path, fake_face_tracker):
    with pytest.raises(FileNotFoundError):
        extract_traces(tmp_path / "nope.avi")
    bad = tmp_path / "garbage.avi"
    bad.write_bytes(b"this is not a video")
    with pytest.raises(ValueError, match="cannot open video"):
        extract_traces(bad)
    assert fake_face_tracker.instances == []


def test_supplied_tracker_is_not_closed(video):
    tracker = FakeTracker(LM)
    tracker.close = lambda: pytest.fail("caller-owned tracker must not be closed")
    extract_traces(video, tracker=tracker)
