"""Batch extraction: per-video rows, resumability, run log, and the extract_all script.

Synthetic .avi clips and a scripted tracker; the tracker is patched in-process only. A spawned
Pool re-imports modules, so that test fills the cache in-process first and the children only load.
"""

import csv
import importlib.util
import multiprocessing
import time
from pathlib import Path

import numpy as np
import pytest

from conftest import FIXTURES, FakeTracker, write_video
from pulsecheck.batch import extract_one, run_batch
from pulsecheck.datasets.manifest import VideoItem, write_manifest

LM = np.load(FIXTURES / "landmarks_face.npy")
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
LOG_COLUMNS = ["rel_path", "dataset", "status", "seconds", "frames", "message"]
MAX_SECONDS = 3.0
N_FRAMES = 30


class ClosableTracker(FakeTracker):
    """FakeTracker plus the ``close()`` that ``extract_traces`` calls on a tracker it owns."""

    def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("PULSECHECK_DATA", str(tmp_path / "data"))


@pytest.fixture
def tracker(monkeypatch):
    """Patch the lazily imported FaceTracker with one that always finds the fixture face."""
    monkeypatch.setattr("pulsecheck.face.FaceTracker", lambda: ClosableTracker(LM))


def _items(tmp_path: Path, n: int) -> list[VideoItem]:
    root = tmp_path / "ds root"  # a space in the path, as on the real machine
    items = []
    for i in range(n):
        frames = [np.full((400, 320, 3), (120, 100 + i, 180), np.uint8)] * N_FRAMES
        path = write_video(root / "videos" / f"v{i}.avi", frames, 30)
        items.append(
            VideoItem("ffpp", path, f"videos/v{i}.avi", i % 2, f"g{i}", "k", "c23", "train")
        )
    return items


def _read_log(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_extract_one_ok_then_cached(tmp_path, monkeypatch):
    (item,) = _items(tmp_path, 1)
    monkeypatch.setattr("pulsecheck.face.FaceTracker", lambda: ClosableTracker(LM))
    first = extract_one(item, max_seconds=MAX_SECONDS)
    assert first["status"] == "ok" and first["frames"] == N_FRAMES and first["message"] == ""
    assert first["rel_path"] == "videos/v0.avi" and first["dataset"] == "ffpp"
    assert isinstance(first["seconds"], float) and first["seconds"] >= 0

    def boom():
        raise AssertionError("a tracker must not be built for a cached video")

    monkeypatch.setattr("pulsecheck.face.FaceTracker", boom)
    second = extract_one(item, max_seconds=MAX_SECONDS)
    assert second["status"] == "cached" and second["frames"] == N_FRAMES
    assert second["message"] == ""


def test_extract_one_error_row_not_raise(tmp_path, monkeypatch):
    (item,) = _items(tmp_path, 1)
    never = ClosableTracker(LM, lost_frames=set(range(1000)))
    monkeypatch.setattr("pulsecheck.face.FaceTracker", lambda: never)
    row = extract_one(item, max_seconds=MAX_SECONDS)
    assert row["status"] == "error" and row["frames"] == 0
    assert "NoFaceError" in row["message"] and "face" in row["message"]


def test_extract_one_missing_file_is_an_error_row(tmp_path):
    (item,) = _items(tmp_path, 1)
    item.path.unlink()
    row = extract_one(item, max_seconds=MAX_SECONDS)
    assert row["status"] == "error" and "FileNotFoundError" in row["message"]


def test_extract_one_unreadable_video_is_an_error_row(tmp_path, tracker):
    (item,) = _items(tmp_path, 1)
    item.path.write_bytes(b"not a video")
    row = extract_one(item, max_seconds=MAX_SECONDS)
    assert row["status"] == "error" and row["message"]


def test_extract_one_unexpected_exception_is_still_a_row(tmp_path, monkeypatch):
    (item,) = _items(tmp_path, 1)

    def broken():
        raise RuntimeError("tracker blew up")

    monkeypatch.setattr("pulsecheck.face.FaceTracker", broken)
    row = extract_one(item, max_seconds=MAX_SECONDS)
    assert row["status"] == "error" and row["message"] == "RuntimeError: tracker blew up"


def test_run_batch_in_process_logs_every_row(tmp_path, monkeypatch, capsys):
    items = _items(tmp_path, 3)
    items[1].path.unlink()  # one failing video must not stop the others
    monkeypatch.setattr("pulsecheck.face.FaceTracker", lambda: ClosableTracker(LM))
    log = tmp_path / "results" / "extract_log.csv"
    s = run_batch(items, workers=1, max_seconds=MAX_SECONDS, log_path=log, progress_every=2)
    assert s["total"] == 3 and s["ok"] == 2 and s["error"] == 1 and s["cached"] == 0
    assert s["wall_s"] > 0 and s["videos_per_hour"] > 0 and s["frames_per_s"] > 0
    assert set(s) == {"total", "ok", "cached", "error", "wall_s", "videos_per_hour", "frames_per_s"}
    rows = _read_log(log)
    assert len(rows) == 3 and list(rows[0]) == LOG_COLUMNS
    assert sorted(r["status"] for r in rows) == ["error", "ok", "ok"]
    out = capsys.readouterr().out
    assert "2/3" in out and "3/3" in out and "ETA" in out


def test_run_batch_appends_and_writes_one_header(tmp_path, tracker):
    items = _items(tmp_path, 2)
    log = tmp_path / "log.csv"
    run_batch(items, workers=1, max_seconds=MAX_SECONDS, log_path=log)
    second = run_batch(items, workers=1, max_seconds=MAX_SECONDS, log_path=log)
    assert second["cached"] == 2 and second["ok"] == 0 and second["frames_per_s"] == 0.0
    rows = _read_log(log)
    assert [r["status"] for r in rows] == ["ok", "ok", "cached", "cached"]
    assert log.read_text(encoding="utf-8").count("rel_path,dataset") == 1


def test_run_batch_dedupes_repeated_videos(tmp_path, tracker):
    items = _items(tmp_path, 2)
    log = tmp_path / "log.csv"
    s = run_batch(items + items[:1], workers=1, max_seconds=MAX_SECONDS, log_path=log)
    assert s["total"] == 2 and len(_read_log(log)) == 2


def test_run_batch_spawn_pool_resumes_from_cache(tmp_path, monkeypatch):
    items = _items(tmp_path, 3)
    monkeypatch.setattr("pulsecheck.face.FaceTracker", lambda: ClosableTracker(LM))
    monkeypatch.setattr("pulsecheck.face.ensure_model", lambda *a, **k: tmp_path)  # no download
    run_batch(items, workers=1, max_seconds=MAX_SECONDS, log_path=tmp_path / "first.csv")
    # spawned children have no monkeypatch, so any tracker build there would fail the item
    log = tmp_path / "pool.csv"
    methods = []
    real = multiprocessing.get_context
    monkeypatch.setattr(
        "pulsecheck.batch.multiprocessing.get_context",
        lambda method=None: methods.append(method) or real(method),
    )
    s = run_batch(items, workers=2, max_seconds=MAX_SECONDS, log_path=log, progress_every=1)
    assert methods == ["spawn"]  # a real Pool was used
    assert s["total"] == 3 and s["cached"] == 3 and s["ok"] == 0 and s["error"] == 0
    rows = _read_log(log)
    assert len(rows) == 3 and {r["status"] for r in rows} == {"cached"}
    assert {r["rel_path"] for r in rows} == {i.rel_path for i in items}
    assert all(int(r["frames"]) == N_FRAMES for r in rows)


def test_run_batch_keyboard_interrupt_prints_resume_hint(tmp_path, monkeypatch, capsys):
    items = _items(tmp_path, 2)

    def interrupt(item, max_seconds=MAX_SECONDS):
        raise KeyboardInterrupt

    monkeypatch.setattr("pulsecheck.batch.extract_one", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_batch(items, workers=1, max_seconds=MAX_SECONDS, log_path=tmp_path / "log.csv")
    assert "resume" in capsys.readouterr().out.lower()


def _load_script():
    spec = importlib.util.spec_from_file_location("extract_all_script", SCRIPTS / "extract_all.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_all_main_parses_args_and_limit(tmp_path, tracker, capsys):
    items = _items(tmp_path, 4)
    manifest_a, manifest_b = tmp_path / "a.csv", tmp_path / "b.csv"
    write_manifest(items[:3], manifest_a)
    write_manifest(items[2:], manifest_b)  # overlaps a.csv on one video
    script = _load_script()
    log = tmp_path / "out" / "log.csv"
    code = script.main(
        [
            *("--manifest", str(manifest_a), "--manifest", str(manifest_b)),
            *("--workers", "1", "--max-seconds", "3", "--limit", "3", "--log", str(log)),
        ]
    )
    assert code == 0
    rows = _read_log(log)
    assert [r["rel_path"] for r in rows] == ["videos/v0.avi", "videos/v1.avi", "videos/v2.avi"]
    out = capsys.readouterr().out
    assert "videos_per_hour" in out and "frames_per_s" in out

    # resume: v2 is cached, v3 is new
    args = ["--manifest", str(manifest_b), "--workers", "1", "--max-seconds", "3"]
    args += ["--log", str(log)]
    assert script.main(args) == 0
    assert [r["status"] for r in _read_log(log)[3:]] == ["cached", "ok"]


def test_extract_all_defaults(monkeypatch, tmp_path):
    script = _load_script()
    monkeypatch.setattr("os.cpu_count", lambda: 10)
    args = script.parse_args(["--manifest", "m.csv"])
    assert args.workers == 4 and args.max_seconds == 15.0 and args.limit is None
    assert args.log == tmp_path / "data" / "results" / "extract_log.csv"
    monkeypatch.setattr("os.cpu_count", lambda: None)
    assert script.parse_args(["--manifest", "m.csv"]).workers == 1


def test_extract_all_without_items_returns_nonzero(tmp_path, capsys):
    manifest = tmp_path / "empty.csv"
    write_manifest([], manifest)
    code = _load_script().main(["--manifest", str(manifest), "--log", str(tmp_path / "l.csv")])
    assert code == 1


def sleepy_worker(item, max_seconds=MAX_SECONDS):
    """Top-level (so a spawned child can unpickle it by name) stand-in for a hung extraction."""
    time.sleep(60)
    return {}


def test_run_batch_pool_ensures_model_in_parent_before_creating_pool(tmp_path, monkeypatch):
    items = _items(tmp_path, 2)
    monkeypatch.setattr("pulsecheck.face.FaceTracker", lambda: ClosableTracker(LM))
    run_batch(items, workers=1, max_seconds=MAX_SECONDS, log_path=tmp_path / "first.csv")
    events = []
    monkeypatch.setattr("pulsecheck.face.ensure_model", lambda *a, **k: events.append("model"))
    real = multiprocessing.get_context

    def spy(method=None):
        events.append("pool")
        return real(method)

    monkeypatch.setattr("pulsecheck.batch.multiprocessing.get_context", spy)
    run_batch(items, workers=2, max_seconds=MAX_SECONDS, log_path=tmp_path / "pool.csv")
    assert events == ["model", "pool"]

    events.clear()  # in-process runs need neither
    run_batch(items, workers=1, max_seconds=MAX_SECONDS, log_path=tmp_path / "one.csv")
    assert events == []


def test_run_batch_stalled_pool_is_terminated_and_raises(tmp_path, monkeypatch, capsys):
    """Children get ``sleepy_worker`` by pickle (the ``worker`` parameter), not by monkeypatch,
    which spawned processes would not see."""
    items = _items(tmp_path, 2)
    monkeypatch.setattr("pulsecheck.face.ensure_model", lambda *a, **k: tmp_path)
    log = tmp_path / "stall.csv"
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="no result for"):
        run_batch(
            items, workers=2, max_seconds=MAX_SECONDS, log_path=log,
            stall_timeout_s=2.0, worker=sleepy_worker,
        )  # fmt: skip
    assert time.monotonic() - t0 < 30
    assert multiprocessing.active_children() == []  # the pool was terminated and joined
    assert "resume" in capsys.readouterr().out.lower()
    assert _read_log(log) == []  # header only; nothing finished
