import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from conftest import FIXTURES, FakeTracker, write_video
from pulsecheck.dsp import HRSeries
from pulsecheck.evaluation import align_series, hr_metrics, reference_series

LM = np.load(FIXTURES / "landmarks_face.npy")
FPS = 30
BPM = 72.0
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_hr.py"


def test_hr_metrics_ignores_nan():
    m = hr_metrics(np.array([70, 80, np.nan]), np.array([72, 78, 75]))
    assert m["n"] == 2 and m["mae"] == pytest.approx(2.0) and m["rmse"] == pytest.approx(2.0)


def test_hr_metrics_pearson_and_degenerate_cases():
    m = hr_metrics(np.array([60.0, 70.0, 80.0]), np.array([61.0, 72.0, 79.0]))
    assert m["pearson"] > 0.99
    one = hr_metrics(np.array([70.0]), np.array([72.0]))
    assert one["n"] == 1 and one["mae"] == 2.0 and np.isnan(one["pearson"])
    const = hr_metrics(np.array([70.0, 70.0, 70.0]), np.array([60.0, 70.0, 80.0]))
    assert np.isnan(const["pearson"])
    none = hr_metrics(np.array([np.nan]), np.array([70.0]))
    assert none["n"] == 0 and np.isnan(none["mae"]) and np.isnan(none["rmse"])


def test_hr_metrics_rejects_length_mismatch():
    with pytest.raises(ValueError):
        hr_metrics(np.zeros(3), np.zeros(4))


def _gt(seconds=12.0, bpm=BPM, t0=0.0, rate=30.0):
    t = t0 + np.arange(int(seconds * rate)) / rate
    return t, np.sin(2 * np.pi * bpm / 60 * t)


def test_reference_series_recovers_ppg_rate_on_8s_windows():
    t, ppg = _gt(12.0)
    ref = reference_series(t, ppg)
    assert ref.t_center[0] == pytest.approx(4.0)
    assert np.allclose(np.diff(ref.t_center), 1.0)
    np.testing.assert_allclose(ref.bpm, BPM, atol=1.0)


def test_reference_series_uses_ground_truth_clock_for_irregular_samples():
    t, _ = _gt(12.0)
    rng = np.random.default_rng(0)
    t = np.sort(t + rng.uniform(-0.005, 0.005, t.shape))
    ppg = np.sin(2 * np.pi * BPM / 60 * t)
    np.testing.assert_allclose(reference_series(t, ppg).bpm, BPM, atol=1.0)


def test_reference_series_rejects_non_second_timestamps():
    t, ppg = _gt(12.0)
    with pytest.raises(ValueError, match="seconds"):
        reference_series(t * 1000.0, ppg)  # milliseconds


def test_align_series_truncates_to_the_shorter():
    pred = HRSeries(np.arange(4.0, 9.0), np.array([70.0, 71, 72, 73, 74]), np.zeros(5))
    ref = HRSeries(np.arange(4.0, 7.0), np.array([69.0, 70, 71]), np.zeros(3))
    p, r = align_series(pred, ref)
    np.testing.assert_allclose(p, [70, 71, 72])
    np.testing.assert_allclose(r, [69, 70, 71])


def test_align_series_raises_when_windows_do_not_line_up():
    pred = HRSeries(np.arange(4.0, 9.0), np.zeros(5), np.zeros(5))
    ref = HRSeries(np.arange(14.0, 19.0), np.zeros(5), np.zeros(5))  # GT starts ~10 s late
    with pytest.raises(ValueError, match="align"):
        align_series(pred, ref)


# ---- end to end through scripts/eval_hr.py on a fake UBFC tree -------------------------------


def _load_script():
    spec = importlib.util.spec_from_file_location("eval_hr_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pulse_video(path: Path, seconds: float = 10.0) -> Path:
    frames = []
    for i in range(int(seconds * FPS)):
        g = 100 + 6 * np.sin(2 * np.pi * BPM / 60 * i / FPS)
        frames.append(np.full((400, 320, 3), (120, round(g), 180), np.uint8))
    return write_video(path, frames, FPS)


def _write_gt(path: Path, seconds: float = 10.0) -> None:
    t, ppg = _gt(seconds)
    rows = [ppg, np.full_like(t, BPM), t]
    path.write_text("\n".join(" ".join(f"{v:.6f}" for v in r) for r in rows) + "\n")


def _fake_ubfc(root: Path, n: int = 2) -> Path:
    for i in range(1, n + 1):
        d = root / f"subject{i}"
        _pulse_video(d / "vid.avi")
        _write_gt(d / "ground_truth.txt")
    return root


class _Tracker(FakeTracker):
    def __init__(self):
        super().__init__(LM)

    def close(self):
        pass


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PULSECHECK_DATA", str(tmp_path / "data"))
    monkeypatch.setattr("pulsecheck.face.FaceTracker", _Tracker)
    return tmp_path


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_eval_script_end_to_end_on_fake_ubfc(env):
    root = _fake_ubfc(env / "DATASET_2")
    _pulse_video(env / "mine" / "clip.avi")
    clips = env / "mine" / "clips.csv"
    clips.write_text(
        "path,counted_bpm,recorded_at\n"
        "clip.avi,70,2026-10-26T10:00:00\n"  # relative to the CSV's directory
        "gone.avi,70,2026-10-26T10:05:00\n"  # missing: skipped with a warning
    )
    csv_out, md_out = env / "out" / "hr_eval.csv", env / "out" / "hr_eval.md"

    main = _load_script().main
    code = main(
        ["--ubfc-root", str(root), "--own-clips", str(clips), "--max-seconds", "60"]
        + ["--csv-out", str(csv_out), "--md-out", str(md_out)]
    )
    assert code == 0

    rows = _read_csv(csv_out)
    ubfc = [r for r in rows if r["dataset"] == "ubfc"]
    assert {(r["subject"], r["method"]) for r in ubfc} == {
        (f"subject{i}", m) for i in (1, 2) for m in ("green", "chrom", "pos")
    }
    assert all(r["error"] == "" and int(r["n"]) >= 1 for r in ubfc)
    pos = [r for r in ubfc if r["method"] == "pos"]
    assert all(float(r["mae"]) < 5.0 for r in pos)  # a clean 72 bpm modulation
    own = [r for r in rows if r["dataset"] == "own"]
    assert {r["method"] for r in own} == {"green", "chrom", "pos"}
    assert all(r["subject"].endswith("clip.avi") for r in own)

    md = md_out.read_text()
    for method in ("green", "chrom", "pos"):
        assert f"| {method} |" in md
    assert "UBFC" in md and "own clips" in md.lower()
    assert "eval_hr.py" in md and "--ubfc-root" in md  # the exact command is recorded
    assert "failed" in md.lower()


def test_eval_script_reports_failed_subjects_as_error_rows(env):
    root = _fake_ubfc(env / "DATASET_2", n=2)
    (root / "subject2" / "ground_truth.txt").write_text("1 2 3\n4 5 6\n")  # malformed
    csv_out, md_out = env / "hr_eval.csv", env / "hr_eval.md"

    code = _load_script().main(
        ["--ubfc-root", str(root), "--csv-out", str(csv_out), "--md-out", str(md_out)]
    )
    assert code == 0  # subject1 still succeeded

    rows = _read_csv(csv_out)
    bad = [r for r in rows if r["subject"] == "subject2"]
    assert len(bad) == 3 and all("ground_truth.txt" in r["error"] for r in bad)
    assert all(r["error"] == "" for r in rows if r["subject"] == "subject1")
    assert "1 failed" in md_out.read_text()


def test_eval_script_fails_when_every_subject_fails(env):
    root = _fake_ubfc(env / "DATASET_2", n=1)
    (root / "subject1" / "ground_truth.txt").write_text("nonsense\n")
    code = _load_script().main(
        ["--ubfc-root", str(root), "--csv-out", str(env / "a.csv"), "--md-out", str(env / "a.md")]
    )
    assert code == 1


def test_eval_script_argument_errors(env, capsys):
    main = _load_script().main
    assert main(["--ubfc-root", str(env / "missing")]) == 2
    assert "missing" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main([])  # needs --ubfc-root or --own-clips
    with pytest.raises(SystemExit):
        main(["--ubfc-root", str(env), "--methods", "pos,bogus"])
