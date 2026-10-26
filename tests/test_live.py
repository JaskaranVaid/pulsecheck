import csv
import time

import cv2
import numpy as np
import pytest

from conftest import FIXTURES, FakeTracker
from pulsecheck import cli, live
from pulsecheck.cli import main
from pulsecheck.live import (
    RollingBuffer,
    compute_bpm,
    record_clip,
    render_waveform,
    run_live,
    status_text,
)
from pulsecheck.regions import SKIN_REGIONS
from synth import synth_rgb

LM = np.load(FIXTURES / "landmarks_face.npy")
FACE = cv2.imread(str(FIXTURES / "face.jpg"))


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PULSECHECK_DATA", str(tmp_path / "data"))


class FakeCamera:
    """Delivers a fixed frame at ~``fps``; fails after ``n_frames`` if given."""

    def __init__(self, frame, fps=30.0, n_frames=None):
        self.frame, self.fps, self.n_frames = frame, fps, n_frames
        self.reads = 0
        self.released = False

    def read(self):
        if self.n_frames is not None and self.reads >= self.n_frames:
            return False, None
        self.reads += 1
        time.sleep(1 / self.fps)
        return True, self.frame.copy()

    def release(self):
        self.released = True


@pytest.fixture
def headless(monkeypatch):
    """No windows; ``keys`` controls what cv2.waitKey returns."""
    state = {"shown": [], "key": -1}
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda name, img: state["shown"].append(img))
    monkeypatch.setattr(cv2, "waitKey", lambda ms=0: state["key"])
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a: 1.0)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    return state


def test_rolling_buffer_drops_old_samples():
    b = RollingBuffer(seconds=2.0, n_regions=8)
    for i in range(100):
        b.append(i / 30, np.full((8, 3), i, np.float32))
    t, rgb = b.arrays()
    assert t[0] >= t[-1] - 2.0 and rgb.shape[1:] == (8, 3) and b.duration <= 2.0


def test_rolling_buffer_empty_and_bad_shape():
    b = RollingBuffer(seconds=2.0, n_regions=7)
    t, rgb = b.arrays()
    assert t.shape == (0,) and rgb.shape == (0, 7, 3) and b.duration == 0.0
    with pytest.raises(ValueError):
        b.append(0.0, np.zeros((8, 3)))


def test_render_waveform_shape_and_blank_cases():
    img = render_waveform(np.sin(np.linspace(0, 10, 300)), 400, 120)
    assert img.shape == (120, 400, 3) and img.dtype == np.uint8 and img.any()
    assert not render_waveform(np.array([]), 400, 120).any()
    assert not render_waveform(np.full(10, np.nan), 400, 120).any()


def test_render_waveform_flat_single_and_gappy_input():
    assert render_waveform(np.full(50, 3.0), 400, 120).any()
    assert render_waveform(np.array([1.0]), 400, 120).any()
    gappy = np.sin(np.linspace(0, 10, 300))
    gappy[100:150] = np.nan
    assert render_waveform(gappy, 400, 120).any()


def test_status_text():
    assert status_text(False, 20.0, 70.0) == "No face"
    assert status_text(True, 2.0, None) == "Hold still... 3 s"
    assert status_text(True, 7.0, None) == "-- bpm"
    assert status_text(True, 7.0, 71.6) == "72 bpm"


def test_compute_bpm_recovers_synthetic_pulse():
    t, rgb = synth_rgb(72, 10.0)
    regions = np.repeat(rgb[:, None, :], len(SKIN_REGIONS), axis=1)
    bpm = compute_bpm(t, regions, "pos", 30.0)
    assert bpm is not None and abs(bpm - 72) <= 3


def test_compute_bpm_ignores_clock_offset_and_nan_regions():
    t, rgb = synth_rgb(90, 10.0)
    regions = np.repeat(rgb[:, None, :], len(SKIN_REGIONS), axis=1)
    regions[:, :3, :] = np.nan  # three regions unusable
    bpm = compute_bpm(t + 1234.5, regions, "chrom", 30.0)
    assert bpm is not None and abs(bpm - 90) <= 3


def test_compute_bpm_is_not_stepped_by_a_region_dropping_out():
    # live shares pipeline.face_signal: regions are normalised before averaging, so a bright
    # region that is missing on a random 10% of frames does not inject steps (GREEN would read
    # them as broadband noise).
    t, rgb = synth_rgb(72, 10.0)
    regions = np.repeat(rgb[:, None, :], len(SKIN_REGIONS), axis=1)
    regions *= np.array([1.0, 1.05, 0.95, 1.2, 0.85, 1.1, 0.9])[None, :, None]
    regions[np.random.default_rng(1).random(len(t)) < 0.10, 3] = np.nan
    bpm = compute_bpm(t, regions, "green", 30.0)
    assert bpm is not None and abs(bpm - 72) <= 0.5


def test_compute_bpm_none_when_not_usable():
    t, rgb = synth_rgb(72, 10.0)
    regions = np.repeat(rgb[:, None, :], len(SKIN_REGIONS), axis=1)
    assert compute_bpm(t[:1], regions[:1]) is None  # too few samples
    assert compute_bpm(t[:20], regions[:20]) is None  # shorter than the POS window + filter
    assert compute_bpm(t, np.full_like(regions, np.nan)) is None  # no face at all
    regions[: len(t) // 2] = np.nan  # face missing for half the buffer
    assert compute_bpm(t, regions) is None
    assert compute_bpm(t[:0], regions[:0]) is None


def test_run_live_bad_camera_returns_2(capsys):
    assert run_live(camera=99) == 2 and "cannot open camera" in capsys.readouterr().err


def test_run_live_checks_camera_before_building_tracker(monkeypatch, capsys):
    def boom():
        raise AssertionError("tracker built before the camera was checked")

    monkeypatch.setattr(live, "_build_tracker", boom)
    monkeypatch.setattr(live, "_open_camera", lambda c, size=None: (_ for _ in ()).throw(
        live.CameraError(f"cannot open camera {c}")))
    assert run_live(camera=3) == 2
    assert "error: cannot open camera 3" in capsys.readouterr().err


def test_run_live_unknown_method_returns_2(capsys):
    assert run_live(method="nope") == 2 and "unknown method" in capsys.readouterr().err


def test_run_live_loop_shows_frames_and_quits(monkeypatch, headless):
    cam = FakeCamera(FACE, fps=100)
    monkeypatch.setattr(live, "_open_camera", lambda c, size=None: cam)
    tracker = FakeTracker(LM, lost_frames={2})  # one lost frame mid-stream
    seen = []
    orig = tracker.detect

    def spy(frame, ts):
        seen.append(ts)
        if len(seen) >= 25:
            headless["key"] = ord("q")
        return orig(frame, ts)

    tracker.detect = spy
    assert run_live(camera=0, tracker=tracker) == 0
    assert cam.released
    assert len(headless["shown"]) == 25
    assert seen == sorted(set(seen))  # strictly increasing millisecond timestamps
    h, w = FACE.shape[:2]
    assert headless["shown"][-1].shape == (h + live.PANEL_HEIGHT, w, 3)


def test_run_live_camera_dying_returns_2(monkeypatch, headless, capsys):
    cam = FakeCamera(FACE, fps=1000, n_frames=3)
    monkeypatch.setattr(live, "_open_camera", lambda c, size=None: cam)
    assert run_live(camera=0, tracker=FakeTracker(LM)) == 2
    assert "stopped delivering frames" in capsys.readouterr().err and cam.released


def test_record_clip_writes_avi_at_constant_fps(tmp_path, monkeypatch, headless):
    small = cv2.resize(FACE, (320, 400))
    monkeypatch.setattr(live, "_open_camera", lambda c, size=None: FakeCamera(small, fps=40))
    out = record_clip(tmp_path / "sub dir" / "c.avi", seconds=1.0, camera=0, countdown_s=0)
    assert out.is_file()
    cap = cv2.VideoCapture(str(out))
    n = 0
    while cap.read()[0]:
        n += 1
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(30.0, abs=0.1)
    assert 28 <= n <= 32
    assert headless["shown"]  # timer overlay was displayed


def test_record_clip_countdown_shown_before_recording(tmp_path, monkeypatch, headless):
    small = cv2.resize(FACE, (160, 200))
    monkeypatch.setattr(live, "_open_camera", lambda c, size=None: FakeCamera(small, fps=50))
    t0 = time.perf_counter()
    record_clip(tmp_path / "c.avi", seconds=0.3, countdown_s=1)
    assert time.perf_counter() - t0 >= 1.3
    assert len(headless["shown"]) > 15


def test_record_clip_cancel_removes_partial_file(tmp_path, monkeypatch, headless):
    small = cv2.resize(FACE, (160, 200))
    monkeypatch.setattr(live, "_open_camera", lambda c, size=None: FakeCamera(small, fps=50))
    headless["key"] = ord("q")
    out = tmp_path / "c.avi"
    with pytest.raises(live.LiveError, match="cancelled"):
        record_clip(out, seconds=5.0, countdown_s=0)
    assert not out.exists()


def test_record_clip_keeps_preexisting_file_when_stopped_in_countdown(
    tmp_path, monkeypatch, headless
):
    small = cv2.resize(FACE, (160, 200))
    out = tmp_path / "existing.avi"
    out.write_bytes(b"precious")

    monkeypatch.setattr(live, "_open_camera", lambda c, size=None: FakeCamera(small, fps=50))
    headless["key"] = ord("q")  # cancelled during the countdown
    with pytest.raises(live.LiveError, match="cancelled"):
        record_clip(out, seconds=5.0, countdown_s=1)
    assert out.read_bytes() == b"precious"

    headless["key"] = -1  # camera dies during the countdown
    monkeypatch.setattr(
        live, "_open_camera", lambda c, size=None: FakeCamera(small, fps=500, n_frames=3)
    )
    with pytest.raises(live.CameraError):
        record_clip(out, seconds=5.0, countdown_s=1)
    assert out.read_bytes() == b"precious"


def test_record_clip_bad_camera_and_extension(tmp_path):
    with pytest.raises(live.CameraError, match="cannot open camera 99"):
        record_clip(tmp_path / "c.avi", seconds=1.0, camera=99)
    with pytest.raises(live.LiveError, match="avi"):
        record_clip(tmp_path / "c.mp4", seconds=1.0)
    with pytest.raises(live.LiveError, match="positive"):
        record_clip(tmp_path / "c.avi", seconds=0)


# ------------------------------------------------------------------------------ CLI


def test_cli_live_bad_camera_exit_2_without_tracker(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_make_tracker", lambda: pytest.fail("tracker built"))
    assert main(["live", "--camera", "99"]) == 2
    assert "error: cannot open camera 99" in capsys.readouterr().err


def test_cli_live_passes_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(live, "run_live", lambda **kw: calls.append(kw) or 0)
    assert main(["live", "--camera", "2", "--method", "chrom"]) == 0
    assert calls == [{"camera": 2, "method": "chrom"}]


def test_cli_record_bad_camera_exit_2(capsys):
    assert main(["record", "--camera", "99", "--seconds", "1"]) == 2
    assert "error: cannot open camera 99" in capsys.readouterr().err


def _rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def _fake_record(monkeypatch):
    def fake(out, seconds, camera):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x")
        return out

    monkeypatch.setattr(live, "record_clip", fake)


def _answers(monkeypatch, *answers):
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))


def test_cli_record_logs_counted_pulse(tmp_path, monkeypatch, capsys):
    _fake_record(monkeypatch)
    _answers(monkeypatch, "abc", "72")  # first answer invalid, second accepted
    out = tmp_path / "spaced dir" / "a.avi"
    assert main(["record", "--out", str(out), "--seconds", "5"]) == 0
    csv_path = tmp_path / "data" / "own" / "clips.csv"
    rows = _rows(csv_path)
    assert rows[0] == ["path", "counted_bpm", "recorded_at"]
    assert rows[1][:2] == [str(out), "72"] and len(rows) == 2
    assert "T" in rows[1][2]  # ISO-8601

    _answers(monkeypatch, "65.5")
    assert main(["record", "--out", str(tmp_path / "b.avi")]) == 0
    rows = _rows(csv_path)
    assert len(rows) == 3 and rows[2][1] == "65.5"  # appended, header not repeated


def test_cli_record_default_path_is_under_data_own(tmp_path, monkeypatch):
    _fake_record(monkeypatch)
    _answers(monkeypatch, "70")
    assert main(["record"]) == 0
    rows = _rows(tmp_path / "data" / "own" / "clips.csv")
    assert rows[1][0].startswith(str(tmp_path / "data" / "own" / "clip_"))
    assert rows[1][0].endswith(".avi")


def test_cli_record_three_bad_answers_saves_clip_without_row(tmp_path, monkeypatch, capsys):
    _fake_record(monkeypatch)
    _answers(monkeypatch, "x", "5", "")  # non-numeric, implausible, empty
    out = tmp_path / "a.avi"
    assert main(["record", "--out", str(out)]) == 0
    assert out.exists()
    assert not (tmp_path / "data" / "own" / "clips.csv").exists()
    assert "warning" in capsys.readouterr().err


def test_cli_record_eof_on_prompt_is_not_a_crash(tmp_path, monkeypatch, capsys):
    _fake_record(monkeypatch)

    def eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert main(["record", "--out", str(tmp_path / "a.avi")]) == 0
    assert "warning" in capsys.readouterr().err

