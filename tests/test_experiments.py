"""``scripts/run_experiments.py`` end to end on synthetic feature tables (never the repo docs)."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

from pulsecheck import model
from synth import synth_celebdf_table, synth_ffpp_table

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_experiments.py"
REPO_MD = ROOT / "docs" / "results" / "detection.md"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_experiments_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while being defined
    spec.loader.exec_module(module)
    return module


def _section(md: str, heading: str) -> str:
    """Text of the ``## <heading>...`` section, up to the next level-2 heading."""
    start = md.index(f"\n## {heading}")
    end = md.find("\n## ", start + 1)
    return md[start : end if end != -1 else len(md)]


def _table_rows(section: str) -> list[dict[str, str]]:
    """The first Markdown table of ``section`` as dicts keyed by its header cells."""
    all_lines = section.splitlines()
    first = next(i for i, ln in enumerate(all_lines) if ln.startswith("|"))
    lines = []
    for ln in all_lines[first:]:
        if not ln.startswith("|"):
            break
        lines.append(ln)
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    return [
        dict(zip(header, (c.strip() for c in ln.strip("|").split("|")), strict=True))
        for ln in lines[2:]
    ]


def _repo_md_state():
    return REPO_MD.read_bytes() if REPO_MD.exists() else None


def test_run_experiments_end_to_end_synthetic(tmp_path):
    features = tmp_path / "features"
    features.mkdir()
    c23 = synth_ffpp_table()
    c23.to_csv(features / "ffpp_c23.csv", index=False)
    synth_celebdf_table().to_csv(features / "celebdf_test.csv", index=False)
    md_out, fig_dir, csv_out = tmp_path / "detection.md", tmp_path / "figs", tmp_path / "results"
    before = _repo_md_state()

    script = _load_script()
    code = script.main([
        "--features-dir", str(features), "--md-out", str(md_out),
        "--fig-dir", str(fig_dir), "--csv-out", str(csv_out),
    ])
    assert code == 0
    assert _repo_md_state() == before  # the repo's results file is never touched by tests
    md = md_out.read_text(encoding="utf-8")

    # Header: date, exact command, row counts, NaN rates.
    assert "Run date: " in md
    assert "python scripts/run_experiments.py --features-dir" in md
    inputs = _section(md, "Inputs")
    assert "| ffpp_c23.csv | train | 6 | 18 | 24 |" in inputs
    assert "| celebdf_test.csv | test | 6 | 6 | 12 |" in inputs
    assert "ffpp_c40.csv: not found" in inputs
    assert "| c_amp_nose |" in inputs and "NaN rate" in inputs

    # E3 needs the c40 table: not run, and says why.
    assert "not run: missing ffpp_c40.csv" in _section(md, "E3")

    # E2: one row per group set and model; the printed AUCs are the ones fit_and_test gives.
    e2 = _table_rows(_section(md, "E2"))
    assert {(r["groups"], r["model"]) for r in e2} == {
        (g, k) for g in ("A", "AB", "ABC", "ABCD") for k in ("logreg", "hgb")
    }
    table = pd.read_csv(features / "ffpp_c23.csv")
    train, val, test = (table[table["split"] == s] for s in ("train", "val", "test"))
    for groups in ("A", "ABC"):
        row = next(r for r in e2 if r["groups"] == groups and r["model"] == "logreg")
        cols = model.feature_columns(table, groups)
        expected = model.fit_and_test(train, val, test, cols, "logreg")
        assert row["test AUC"] == f"{expected['metrics']['auc']:.3f}"
        assert row["Deepfakes"] == f"{expected['per_kind_auc']['Deepfakes']:.3f}"
        cv = model.grouped_cv(train, cols, "logreg")
        assert row["CV AUC (train)"].startswith(f"{cv['auc_mean']:.3f} ± {cv['auc_std']:.3f}")
    e2_text = _section(md, "E2")
    assert "predicted fake" in e2_text  # the best model's confusion matrix

    # E4 ran (Celeb-DF table present), E6 ran on the best ABCD model.
    e4 = _table_rows(_section(md, "E4"))
    assert len(e4) == 8 and all(r["test AUC"] != "" for r in e4)
    e6 = _section(md, "E6")
    assert "c_amp_mahal" in e6  # explains where the reference distances' effect shows up
    assert "without C" in e6

    for name in ("roc_e2.png", "roc_e4.png"):
        assert (fig_dir / name).stat().st_size > 0
    assert (csv_out / "detection_results.csv").is_file()
    assert (csv_out / "detection_importances.csv").is_file()


def test_run_experiments_without_c23_writes_nothing(tmp_path, capsys):
    features = tmp_path / "features"
    features.mkdir()
    md_out = tmp_path / "detection.md"
    code = _load_script().main([
        "--features-dir", str(features), "--md-out", str(md_out),
        "--fig-dir", str(tmp_path / "figs"), "--csv-out", str(tmp_path / "results"),
    ])
    assert code == 1
    assert not md_out.exists()  # a "pending" results file is not replaced by an empty one
    assert "ffpp_c23.csv" in capsys.readouterr().err


def test_run_experiments_kind_option_and_missing_group(tmp_path):
    features = tmp_path / "features"
    features.mkdir()
    for compression in ("c23", "c40"):  # no group-D columns; c40 reuses the cached features
        table = synth_ffpp_table(compression=compression)
        table.drop(columns=[c for c in table.columns if c.startswith("d_")]).to_csv(
            features / f"ffpp_{compression}.csv", index=False
        )
    md_out = tmp_path / "detection.md"
    code = _load_script().main([
        "--features-dir", str(features), "--kind", "logreg", "--md-out", str(md_out),
        "--fig-dir", str(tmp_path / "figs"), "--csv-out", str(tmp_path / "results"),
    ])
    assert code == 0
    md = md_out.read_text(encoding="utf-8")
    e2 = _section(md, "E2")
    assert {r["model"] for r in _table_rows(e2)} == {"logreg"}
    assert "not run: no group D columns in ffpp_c23.csv" in e2
    assert "not run" in _section(md, "E6")
    assert "not run: missing celebdf_test.csv" in _section(md, "E4")
    e3 = _section(md, "E3")
    assert [(r["groups"], r["model"]) for r in _table_rows(e3)] == [
        ("A", "logreg"), ("AB", "logreg"), ("ABC", "logreg")
    ]
    assert "not run: no group D columns in ffpp_c23.csv" in e3
