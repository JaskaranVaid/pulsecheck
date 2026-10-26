import numpy as np
import pytest

from conftest import FIXTURES, FakeTracker, write_video
from pulsecheck import cli
from pulsecheck.cli import main
from pulsecheck.extract import RegionTraces

LM = np.load(FIXTURES / "landmarks_face.npy")
FPS = 30
W, H = 320, 400


def _video(path, seconds=10.0, bpm=72):
    """Uniform frames; the colour pulses with a different amplitude per channel."""
    frames = []
    for i in range(int(seconds * FPS)):
        w = np.sin(2 * np.pi * (bpm / 60) * i / FPS)
        r, g, b = 180 + 2 * w, 130 + 4 * w, 110 + 3 * w
        frames.append(np.full((H, W, 3), (round(b), round(g), round(r)), np.uint8))
    return write_video(path, frames, FPS)


@pytest.fixture
def video(tmp_path):
    return _video(tmp_path / "with space" / "clip.avi")


@pytest.fixture
def fake_tracker(monkeypatch):
    monkeypatch.setattr(cli, "_make_tracker", lambda: FakeTracker(LM))


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PULSECHECK_DATA", str(tmp_path / "data"))


def test_cli_hr_prints_table(video, fake_tracker, capsys):
    assert main(["hr", str(video), "--method", "all"]) == 0
    out = capsys.readouterr().out
    assert "face" in out and "pos" in out and "chrom" in out and "green" in out
    assert "control" in out and "nose" in out
    last = out.strip().splitlines()[-1]
    assert last.startswith("processed 300 frames in ") and "frames/s" in last


def test_cli_hr_single_method_default_is_pos(video, fake_tracker, capsys):
    assert main(["hr", str(video)]) == 0
    out = capsys.readouterr().out
    assert "pos" in out and "chrom" not in out


def test_cli_hr_max_seconds(video, fake_tracker, capsys):
    assert main(["hr", str(video), "--max-seconds", "5"]) == 0
    assert "processed 150 frames" in capsys.readouterr().out


@pytest.mark.parametrize("args", [["hr"], ["extract"], ["debug-video", "--out", "x.avi"]])
def test_cli_missing_file_exit_2_without_building_tracker(args, monkeypatch, capsys):
    def boom():
        raise AssertionError("tracker must not be built for a missing file")

    monkeypatch.setattr(cli, "_make_tracker", boom)
    sub, *rest = args
    assert main([sub, "does-not-exist.mp4", *rest]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_no_face_exit_2(video, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_make_tracker", lambda: FakeTracker(LM, lost_frames=set(range(999))))
    assert main(["hr", str(video)]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_short_clip_exit_2(tmp_path, fake_tracker, capsys):
    short = _video(tmp_path / "short.avi", seconds=2.0)
    assert main(["hr", str(short)]) == 2
    assert "too short" in capsys.readouterr().err


def test_cli_extract_writes_npz_to_data_dir(video, fake_tracker, tmp_path, capsys):
    assert main(["extract", str(video)]) == 0
    out = capsys.readouterr().out
    expected = tmp_path / "data" / "traces" / "clip.npz"
    assert expected.exists() and str(expected) in out
    assert "processed 300 frames" in out
    assert RegionTraces.load(expected).rgb.shape[0] == 300


def test_cli_extract_custom_out(video, fake_tracker, tmp_path, capsys):
    target = tmp_path / "elsewhere" / "t.npz"
    assert main(["extract", str(video), "--out", str(target), "--max-seconds", "4"]) == 0
    assert RegionTraces.load(target).rgb.shape[0] == 120


def test_cli_debug_video(video, fake_tracker, tmp_path, capsys):
    target = tmp_path / "dbg" / "overlay.avi"
    assert main(["debug-video", str(video), "--out", str(target), "--max-seconds", "2"]) == 0
    assert target.exists() and target.stat().st_size > 0
    assert str(target) in capsys.readouterr().out


def test_cli_closes_tracker(video, monkeypatch):
    tr = FakeTracker(LM)
    closed = []
    tr.close = lambda: closed.append(True)
    monkeypatch.setattr(cli, "_make_tracker", lambda: tr)
    assert main(["hr", str(video)]) == 0
    assert closed == [True]


def test_cli_no_subcommand_is_usage_error(capsys):
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 2
