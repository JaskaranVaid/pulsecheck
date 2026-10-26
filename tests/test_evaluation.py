import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from conftest import FIXTURES, FakeTracker, write_video
from pulsecheck.dsp import HRSeries, bandpass, estimate_hr
from pulsecheck.evaluation import align_series, hr_metrics, reference_hr, reference_series

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


def test_reference_hr_is_estimate_hr_on_the_band_passed_grid_of_the_video_duration():
    t, ppg = _gt(12.0, t0=3.0)
    r = reference_hr(t, ppg, duration_s=10.0)
    grid = 3.0 + np.arange(301) / 30.0  # 10 s at 30 Hz, on the ground-truth clock
    assert r == estimate_hr(bandpass(np.interp(grid, t, ppg), 30.0), 30.0)
    assert r.bpm == pytest.approx(BPM, abs=0.5)


def test_reference_hr_only_looks_at_the_video_duration():
    t = np.arange(20 * 30) / 30.0
    bpm = np.where(t < 10.0, 72.0, 96.0)  # the pulse speeds up after the video ends
    ppg = np.sin(2 * np.pi * np.cumsum(bpm / 60.0) / 30.0)
    assert reference_hr(t, ppg, duration_s=9.5).bpm == pytest.approx(72.0, abs=1.0)
    assert reference_hr(t[300:], ppg[300:], duration_s=9.5).bpm == pytest.approx(96.0, abs=1.0)


def test_reference_hr_is_nan_when_ground_truth_covers_too_little():
    t, ppg = _gt(7.0)  # ends 3 s (30%) before the 10 s video does
    r = reference_hr(t, ppg, duration_s=10.0)
    assert np.isnan(r.bpm) and np.isnan(r.snr_db) and np.isnan(r.peak_hz)
    with pytest.raises(ValueError, match="seconds"):
        reference_hr(t * 1000.0, ppg, duration_s=10.0)


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

    assert "Windowed (8 s / 1 s, pooled)" in md and "Per-video (whole clip)" in md
    assert "different subject sets" not in md  # every method has the same pairs here

    # The rendered numbers match a recomputation from the written CSV (pooled over windows:
    # MAE weighted by n, RMSE from the n-weighted mean square error).
    tables = {"ubfc": _md_rows(md, "### Windowed"), "own": _md_rows(md, "## Own clips")}
    for dataset, table in tables.items():
        for method in ("green", "chrom", "pos"):
            sel = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            n = np.array([int(r["n"]) for r in sel], float)
            mae = np.array([float(r["mae"]) for r in sel])
            rmse = np.array([float(r["rmse"]) for r in sel])
            cells = table[method]
            assert float(cells[0]) == pytest.approx((mae * n).sum() / n.sum(), abs=0.011)
            assert float(cells[1]) == pytest.approx(
                np.sqrt((rmse**2 * n).sum() / n.sum()), abs=0.011
            )
            assert int(cells[3]) == len(sel)  # subjects / clips evaluated
            assert int(cells[4]) == int(n.sum())  # pairs

    # Per-video table: one (whole-clip prediction, whole-clip reference) pair per subject.
    video = _md_rows(md, "### Per-video")
    for method in ("green", "chrom", "pos"):
        sel = [r for r in ubfc if r["method"] == method]
        err = np.array([float(r["clip_pred_bpm"]) - float(r["clip_ref_bpm"]) for r in sel])
        assert float(video[method][0]) == pytest.approx(np.abs(err).mean(), abs=0.006)
        assert float(video[method][1]) == pytest.approx(np.sqrt((err**2).mean()), abs=0.006)
        assert video[method][3:] == ["2", "2"]  # 2 subjects, 2 pairs
    pos_clip = [float(r["clip_pred_bpm"]) for r in pos]
    assert all(abs(b - BPM) < 1.0 for b in pos_clip)


def _md_rows(md: str, heading: str) -> dict[str, list[str]]:
    """``{method: [MAE, RMSE, Pearson, count, pairs]}`` from the table under ``heading``."""
    section = md.split(heading, 1)[1].split("\n#", 1)[0]
    out = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and cells[0] in ("green", "chrom", "pos"):
            out[cells[0]] = cells[1:]
    return out


def test_summary_table_pools_windows_and_renders_exact_numbers():
    # two subjects with varying HR; one NaN window is dropped from the pooled metrics
    a_pred, a_ref = np.array([70.0, 80.0, 90.0, np.nan]), np.array([72.0, 78.0, 95.0, 60.0])
    b_pred, b_ref = np.array([60.0, 65.0]), np.array([61.0, 70.0])
    pairs = {"pos": ([a_pred, b_pred], [a_ref, b_ref]), "green": ([], [])}
    rows = [
        {"dataset": "ubfc", "subject": s, "method": m, "error": ""}
        for s in ("subject1", "subject2")
        for m in ("pos", "green")
    ]
    rows[-1]["error"] = "boom"  # subject2/green failed
    rows.append({"dataset": "ubfc", "subject": "subject3", "method": "pos", "error": "x"})

    lines = _load_script()._summary_table("ubfc", rows, pairs, ["pos", "green"], "subject")

    # pooled errors: -2, 2, -5, -1, -5 -> MAE 15/5 = 3, RMSE sqrt(59/5) = 3.435
    p = np.array([70.0, 80.0, 90.0, 60.0, 65.0])
    r = np.array([72.0, 78.0, 95.0, 61.0, 70.0])
    pearson = f"{np.corrcoef(p, r)[0, 1]:.2f}"
    assert pearson == "0.97"  # also by hand: 592 / sqrt(580 * 638.8)
    assert "| pos | 3.00 | 3.44 | 0.97 | 2 | 5 |" in lines
    assert "| green | -- | -- | -- | 1 | 0 |" in lines
    assert (
        lines[0] == "1 of 3 subjects evaluated; 2 failed (excluded below)."
    )  # subject2 (green) and subject3
    assert any("different subject sets" in line for line in lines)  # 5 pairs vs 0


def test_per_video_table_renders_exact_numbers_and_flags_unequal_pairs(capsys):
    # one whole-clip pair per subject; pos has no valid estimate for subject3 (NaN)
    pairs = {
        "pos": ([np.array([70.0]), np.array([80.0]), np.array([np.nan])],
                [np.array([72.0]), np.array([77.0]), np.array([75.0])]),
        "green": ([np.array([70.0]), np.array([90.0]), np.array([75.0])],
                  [np.array([72.0]), np.array([77.0]), np.array([75.0])]),
    }
    rows = [
        {"dataset": "ubfc", "subject": f"subject{i}", "method": m, "error": ""}
        for i in (1, 2, 3)
        for m in ("pos", "green")
    ]
    lines = _load_script()._metrics_table("ubfc", rows, pairs, ["pos", "green"], "subject")

    # pos: errors -2, 3 -> MAE 2.5, RMSE sqrt(13/2) = 2.55, two points -> r = 1
    assert "| pos | 2.50 | 2.55 | 1.00 | 3 | 2 |" in lines
    # green: errors -2, 13, 0 -> MAE 15/3 = 5, RMSE sqrt(173/3) = 7.59,
    # r = 48.33 / sqrt(216.67 * 12.667) = 0.92
    assert "| green | 5.00 | 7.59 | 0.92 | 3 | 3 |" in lines
    warning = [line for line in lines if "different subject sets" in line]
    assert len(warning) == 1 and "pos 2" in warning[0] and "green 3" in warning[0]
    assert "different subject sets" in capsys.readouterr().err

    same = {"pos": pairs["green"], "green": pairs["green"]}
    lines = _load_script()._metrics_table("ubfc", rows, same, ["pos", "green"], "subject")
    assert not any("different subject sets" in line for line in lines)


class _LateTracker(_Tracker):
    """No face for the first 100 frames (a third of a 10 s clip)."""

    def detect(self, frame_bgr, timestamp_ms):
        landmarks = super().detect(frame_bgr, timestamp_ms)
        return None if self.calls <= 100 else landmarks


def test_own_clip_without_a_valid_whole_clip_estimate_is_an_error_row(env, monkeypatch):
    monkeypatch.setattr("pulsecheck.face.FaceTracker", _LateTracker)
    _pulse_video(env / "mine" / "clip.avi")
    clips = env / "mine" / "clips.csv"
    clips.write_text("path,counted_bpm,recorded_at" + chr(10) + "clip.avi,70,2026-10-26T10:00:00" + chr(10))
    csv_out, md_out = env / "hr_eval.csv", env / "hr_eval.md"

    code = _load_script().main(
        ["--own-clips", str(clips), "--csv-out", str(csv_out), "--md-out", str(md_out)]
    )
    assert code == 1  # the only clip gave no usable result

    rows = _read_csv(csv_out)
    assert len(rows) == 3
    assert all(r["error"] == "no valid face signal" and r["n"] == "" for r in rows)
    assert all(r["ref_bpm"] == "70.0000" for r in rows)
    md = md_out.read_text()
    assert "0 of 1 clips evaluated; 1 failed" in md
    for method in ("green", "chrom", "pos"):
        assert f"| {method} | -- | -- | -- | 0 | 0 |" in md


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
