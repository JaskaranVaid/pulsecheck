"""``scripts/run_experiments.py`` end to end on synthetic feature tables (never the repo docs)."""

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
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


N_BOOT = "50"  # bootstrap resamples in tests (production default: 1000)
ABCD_LABEL = "ABCD (+ quality covariates — confound check, not the detector)"
D_LABEL = "D only (quality covariates — shortcut check)"
CI = re.compile(r"^(\d\.\d{3}) \[(\d\.\d{3}), (\d\.\d{3})\]$")  # "0.912 [0.871, 0.948]"


def _args(features: Path, tmp_path: Path, *extra: str) -> list[str]:
    return ["--features-dir", str(features), "--md-out", str(tmp_path / "detection.md"),
            "--fig-dir", str(tmp_path / "figs"), "--csv-out", str(tmp_path / "results"),
            "--n-boot", N_BOOT, *extra]


def test_run_experiments_end_to_end_synthetic(tmp_path):
    features = tmp_path / "features"
    features.mkdir()
    c23 = synth_ffpp_table()
    c23.to_csv(features / "ffpp_c23.csv", index=False)
    synth_celebdf_table().to_csv(features / "celebdf_test.csv", index=False)
    md_out, fig_dir, csv_out = tmp_path / "detection.md", tmp_path / "figs", tmp_path / "results"
    before = _repo_md_state()

    script = _load_script()
    code = script.main(_args(features, tmp_path))
    assert code == 0
    assert _repo_md_state() == before  # the repo's results file is never touched by tests
    md = md_out.read_text(encoding="utf-8")

    # Header: date, commit, library versions, exact command, protocol, row counts, NaN rates.
    assert "Run date: " in md
    assert re.search(r"^Commit: ([0-9a-f]{4,}|unknown)$", md, re.MULTILINE)
    versions = next(ln for ln in md.splitlines() if ln.startswith("Versions: "))
    for lib in ("Python", "numpy", "scipy", "scikit-learn", "pandas", "mediapipe"):
        assert f"{lib} " in versions, versions
    assert "python scripts/run_experiments.py --features-dir" in md
    assert "recording quality, not physiology" in md  # the group-D protocol bullet
    assert "group-disjoint by construction" in md  # CV leakage wording (finding 6)
    assert "a_valid_frac" in md and "weakened classic baseline" in md  # group-A caveat
    inputs = _section(md, "Inputs")
    assert "| ffpp_c23.csv | train | 6 | 18 | 24 |" in inputs
    assert "| celebdf_test.csv | test | 6 | 6 | 12 |" in inputs
    assert "ffpp_c40.csv: not found" in inputs
    assert "| c_amp_nose |" in inputs and "NaN rate" in inputs

    # E3 needs the c40 table: not run, and says why.
    assert "not run: missing ffpp_c40.csv" in _section(md, "E3")

    # E2: one row per group set and model, ABCD and D-only labelled as checks; the printed AUCs
    # are the ones fit_and_test gives, each with a grouped-bootstrap 95% CI around it.
    e2 = _table_rows(_section(md, "E2"))
    assert {(r["groups"], r["model"]) for r in e2} == {
        (g, k) for g in ("A", "AB", "ABC", ABCD_LABEL, D_LABEL) for k in ("logreg", "hgb")
    }
    table = pd.read_csv(features / "ffpp_c23.csv")
    train, val, test = (table[table["split"] == s] for s in ("train", "val", "test"))
    for groups in ("A", "ABC"):
        row = next(r for r in e2 if r["groups"] == groups and r["model"] == "logreg")
        cols = model.feature_columns(table, groups)
        expected = model.fit_and_test(train, val, test, cols, "logreg")
        auc, lo, hi = CI.match(row["test AUC [95% CI]"]).groups()
        assert auc == f"{expected['metrics']['auc']:.3f}"
        ci = model.grouped_bootstrap_auc(test["label"], expected["test_scores"], test["group"],
                                         n_boot=int(N_BOOT), seed=0)
        assert (lo, hi) == (f"{ci[0]:.3f}", f"{ci[1]:.3f}")
        assert row["Deepfakes"] == f"{expected['per_kind_auc']['Deepfakes']:.3f}"
        cv = model.grouped_cv(train, cols, "logreg")
        assert row["CV AUC (train)"].startswith(f"{cv['auc_mean']:.3f} ± {cv['auc_std']:.3f}")
    assert all(CI.match(r["test AUC [95% CI]"]) for r in e2)
    e2_text = _section(md, "E2")
    assert "standard error" in e2_text and "0.02–0.03" in e2_text  # per-manipulation note

    # Headline: only a physiological set; its numbers sit next to its selection.
    head = _section(md, "Headline model")
    best = re.search(r"\*\*(\w+) (logreg|hgb)\*\*", head)
    assert best and best.group(1) in ("A", "AB", "ABC"), head
    for needle in ("validation AUC", "E2 c23 test AUC", "95% CI", "5-fold grouped CV",
                   "ddof=0", "E4 Celeb-DF test AUC", "E3: not run", "predicted fake",
                   "ABC − A"):
        assert needle in head, needle
    row = next(r for r in e2 if r["groups"] == best.group(1) and r["model"] == best.group(2))
    assert row["test AUC [95% CI]"] in head and row["val AUC"] in head

    # E4 ran (Celeb-DF table present) with the D-only and ABCD rows; E6 ran on the headline.
    e4 = _table_rows(_section(md, "E4"))
    assert len(e4) == 10 and all(CI.match(r["test AUC [95% CI]"]) for r in e4)
    assert {r["groups"] for r in e4} >= {ABCD_LABEL, D_LABEL}
    e6 = _section(md, "E6")
    assert f"Model: {best.group(1)} {best.group(2)}" in e6
    assert "c_amp_mahal" in e6  # explains where the reference distances' effect shows up
    assert "off-manifold" in e6 and "whole-block" in e6  # single c_amp_* ranks are inflated
    assert "| D |" not in e6  # quality features never enter the headline's importances
    ablation = _table_rows(e6[e6.index("### Group ablation"):])
    assert [r["model"] for r in ablation] == [
        "ABC", "BC (without A)", "AC (without B)", "AB (without C)"
    ]

    for name in ("roc_e2.png", "roc_e4.png"):
        assert (fig_dir / name).stat().st_size > 0
    results = pd.read_csv(csv_out / "detection_results.csv")
    assert {"auc_ci_lo", "auc_ci_hi", "auc_ci_n_used"} <= set(results.columns)
    assert set(results["groups"]) == {"A", "AB", "ABC", "ABCD", "D"}
    assert (csv_out / "detection_importances.csv").is_file()


def _run(groups: str, kind: str, val_auc: float):
    script = _load_script()
    result = {"val_auc": val_auc, "metrics": {"auc": 0.5}}
    return script.Run("E2", groups, kind, result, np.array([0, 1]))


def test_headline_is_never_a_quality_model():
    script = _load_script()
    runs = [_run("A", "logreg", 0.70), _run("ABC", "hgb", 0.80), _run("ABC", "logreg", 0.75),
            _run("ABCD", "hgb", 0.99), _run("D", "hgb", 0.98)]
    best = script._headline(runs)
    assert (best.groups, best.kind) == ("ABC", "hgb")
    assert script._headline([_run("ABCD", "hgb", 0.99), _run("D", "logreg", 0.9)]) is None
    assert set(script.PHYSIO_SETS) == {"A", "AB", "ABC"}
    assert script.GROUP_LABELS["ABCD"] == ABCD_LABEL and script.GROUP_LABELS["D"] == D_LABEL


def test_run_experiments_without_c23_writes_nothing(tmp_path, capsys):
    features = tmp_path / "features"
    features.mkdir()
    md_out = tmp_path / "detection.md"
    code = _load_script().main(_args(features, tmp_path))
    assert code == 1
    assert not md_out.exists()  # a "pending" results file is not replaced by an empty one
    assert "ffpp_c23.csv" in capsys.readouterr().err


def test_run_experiments_split_missing_a_class_writes_nothing(tmp_path, capsys):
    features = tmp_path / "features"
    features.mkdir()
    table = synth_ffpp_table()
    table = table[~((table["split"] == "val") & (table["label"] == 0))]  # val: fakes only
    table.to_csv(features / "ffpp_c23.csv", index=False)
    md_out = tmp_path / "detection.md"
    md_out.write_text("pending", encoding="utf-8")
    code = _load_script().main(_args(features, tmp_path))
    assert code == 1
    assert md_out.read_text(encoding="utf-8") == "pending"  # untouched
    assert not (tmp_path / "figs").exists() and not (tmp_path / "results").exists()
    assert "split 'val' does not hold both real and fake rows" in capsys.readouterr().err


def test_defaults_resolve_from_the_repo_root(monkeypatch, tmp_path):
    script = _load_script()
    assert script.DEFAULT_MD_OUT == ROOT / "docs" / "results" / "detection.md"
    assert script.DEFAULT_FIG_DIR == ROOT / "docs" / "figures"
    assert script.DEFAULT_MD_OUT.is_absolute() and script.DEFAULT_FIG_DIR.is_absolute()
    monkeypatch.chdir(tmp_path)  # the working directory does not move them
    assert _load_script().DEFAULT_MD_OUT == script.DEFAULT_MD_OUT


def test_git_sha_unknown_without_git(monkeypatch):
    script = _load_script()
    sha = script._git_sha()
    assert sha == "unknown" or re.fullmatch(r"[0-9a-f]{4,}", sha)

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(script.subprocess, "run", no_git)
    assert script._git_sha() == "unknown"
    versions = script._versions()
    assert versions.startswith("Python ") and "scikit-learn " in versions


def test_run_experiments_kind_option_and_missing_group(tmp_path):
    features = tmp_path / "features"
    features.mkdir()
    for compression in ("c23", "c40"):  # no group-D columns; c40 reuses the cached features
        table = synth_ffpp_table(compression=compression)
        table.drop(columns=[c for c in table.columns if c.startswith("d_")]).to_csv(
            features / f"ffpp_{compression}.csv", index=False
        )
    md_out = tmp_path / "detection.md"
    code = _load_script().main(_args(features, tmp_path, "--kind", "logreg"))
    assert code == 0
    md = md_out.read_text(encoding="utf-8")
    e2 = _section(md, "E2")
    assert {r["model"] for r in _table_rows(e2)} == {"logreg"}
    assert "not run: no group D columns in ffpp_c23.csv" in e2
    # E6 needs no group D: it runs on the headline physiological model.
    assert "not run" not in _section(md, "E6")
    assert "E3 c40 test AUC" in _section(md, "Headline model")
    assert "not run: missing celebdf_test.csv" in _section(md, "E4")
    e3 = _section(md, "E3")
    assert [(r["groups"], r["model"]) for r in _table_rows(e3)] == [
        ("A", "logreg"), ("AB", "logreg"), ("ABC", "logreg")
    ]
    assert "not run: no group D columns in ffpp_c23.csv" in e3
