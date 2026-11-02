"""Train and evaluate the deepfake detector on the per-video feature tables.

    python scripts/run_experiments.py --features-dir C:/dev/pulsecheck-data/features

Reads ``ffpp_c23.csv``, ``ffpp_c40.csv`` and ``celebdf_test.csv`` (written by
``build_features.py``) from ``--features-dir`` and runs every experiment whose tables exist:

* E2: FF++ c23, fit on the official train split, threshold on val, scored on test, for the
  physiological feature groups A, AB and ABC, plus two checks that are never the detector:
  ABCD (adds the group-D quality covariates: a confound check) and D only (quality covariates
  alone: a shortcut check). Test AUC per manipulation; 5-fold grouped CV on train.
* E3: the same models, scored on the FF++ c40 test split (heavier compression).
* E4: the same models, scored on the Celeb-DF test list (a dataset never trained on).
* E6: permutation importance (top 15 inputs, and per feature group) of the headline model, and a
  group ablation (refit ABC without each of A, B, C).

The headline model is the one with the best validation AUC among A, AB and ABC (never by test
AUC, never a set with group D). Every test AUC gets a 95% CI from a bootstrap over whole leakage
groups. Every experiment needs ``ffpp_c23.csv`` (training and validation); without it, or when
one of its splits lacks real or fake rows, nothing is written and the exit code is 1. An
experiment whose other table is missing gets a "not run: missing <file>" line. Writes the
Markdown summary, ROC figures ``roc_e2.png`` / ``roc_e4.png`` and two CSVs
(``detection_results.csv``, ``detection_importances.csv``).
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from importlib import metadata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive: figures are only written to files

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from pulsecheck import model
from pulsecheck.config import data_dir
from pulsecheck.features import FEATURE_GROUPS

REPO_ROOT = Path(__file__).resolve().parents[1]  # defaults never depend on the working directory
DEFAULT_MD_OUT = REPO_ROOT / "docs" / "results" / "detection.md"
DEFAULT_FIG_DIR = REPO_ROOT / "docs" / "figures"
C23, C40, CELEBDF = "ffpp_c23.csv", "ffpp_c40.csv", "celebdf_test.csv"  # feature tables
TABLES = (C23, C40, CELEBDF)
PHYSIO_SETS = ("A", "AB", "ABC")  # physiological group sets: the only candidates for headline
GROUP_SETS = (*PHYSIO_SETS, "ABCD", "D")  # feature groups of the E2/E3/E4 models
GROUP_LABELS = {  # how a group set is named in the tables (quality sets say what they check)
    **{g: g for g in PHYSIO_SETS},
    "ABCD": "ABCD (+ quality covariates — confound check, not the detector)",
    "D": "D only (quality covariates — shortcut check)",
}
FIGURE_LABELS = {**{g: g for g in PHYSIO_SETS}, "ABCD": "ABCD (+ quality: confound check)",
                 "D": "D only (shortcut check)"}  # shorter names for the ROC legends
ABLATION_BASE = "ABC"  # E6 ablation: refit this set without each of its groups
REQUIRED = ("group", "split", "label", "kind")  # table columns the experiments read
CV_SPLITS = 5  # grouped CV folds on the c23 train split
TOP_FEATURES = 15  # E6: inputs listed
N_REPEATS = 10  # E6: shuffles per permutation importance
N_BOOT = 1000  # grouped bootstrap resamples per test-AUC confidence interval
SEED = 0  # CV folds, permutation shuffles and bootstrap resamples
LIBRARIES = ("numpy", "scipy", "scikit-learn", "pandas", "mediapipe")  # versions in the header
LABEL_NAMES = {0: "real", 1: "fake"}
# Categorical series colours, one per group set in GROUP_SETS order (fixed, never cycled)
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cc7")
CHANCE_COLOR = "#8a8984"  # the diagonal of a random classifier


@dataclass
class Run:
    """One fitted configuration and its test results."""

    experiment: str
    groups: str
    kind: str
    result: dict  # model.fit_and_test output
    y_test: np.ndarray
    cv: dict = field(default_factory=dict)  # E2 only: model.grouped_cv output
    ci: tuple = (float("nan"), float("nan"))  # model.grouped_bootstrap_auc of the test AUC
    test_groups: np.ndarray | None = None  # leakage group of each test row (bootstrap unit)


def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _fmt_ci(value: float, ci) -> str:
    """``0.912 [0.871, 0.948]``: a value and its 95% CI."""
    return f"{_fmt(value)} [{_fmt(ci[0])}, {_fmt(ci[1])}]"


def _quote(arg: str) -> str:
    return f'"{arg}"' if not arg or any(c.isspace() for c in arg) else arg


def _md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


def _git_sha() -> str:
    """Short commit SHA of the repository, or "unknown" when git or the repository is absent."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def _versions() -> str:
    """``Python 3.13.1, numpy 2.2.0, ...``: the interpreter and the libraries in ``LIBRARIES``."""
    parts = [f"Python {platform.python_version()}"]
    for lib in LIBRARIES:
        try:
            parts.append(f"{lib} {metadata.version(lib)}")
        except metadata.PackageNotFoundError:
            parts.append(f"{lib} not installed")
    return ", ".join(parts)


def _load(path: Path) -> pd.DataFrame:
    """A feature table; raises ``ValueError`` on a missing column or a label other than 0/1."""
    df = pd.read_csv(path, dtype={"group": str, "split": str, "kind": str})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}")
    if not df["label"].isin([0, 1]).all():
        raise ValueError(f"{path}: labels must be 0 (real) or 1 (fake)")
    df["label"] = df["label"].astype(int)
    return df


def _both_labels(df: pd.DataFrame) -> bool:
    return set(df["label"]) == {0, 1}


def _split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[df["split"] == split]


def _missing_groups(df: pd.DataFrame, groups: str, name: str) -> str | None:
    """'not run' reason when a letter of ``groups`` has no column in ``df``, else ``None``."""
    for g in groups:
        if not model.feature_columns(df, g):
            return f"not run: no group {g} columns in {name}"
    return None


# --- the experiments ----------------------------------------------------------------------------


def _ffpp_problem(c23: pd.DataFrame) -> str | None:
    for split in ("train", "val", "test"):
        if not _both_labels(_split(c23, split)):
            return f"{C23} split {split!r} does not hold both real and fake rows"
    return None


def _fit(experiment: str, train, val, test, cols, groups: str, kind: str, n_boot: int) -> Run:
    """``model.fit_and_test`` plus the grouped bootstrap CI of the test AUC."""
    print(f"{experiment} {groups} {kind}", flush=True)
    result = model.fit_and_test(train, val, test, cols, kind)
    y, test_groups = test["label"].to_numpy(int), test["group"].to_numpy(str)
    ci = model.grouped_bootstrap_auc(y, result["test_scores"], test_groups, n_boot, SEED)
    return Run(experiment, groups, kind, result, y, ci=ci, test_groups=test_groups)


def _run_e2(c23: pd.DataFrame, kinds: list[str], n_boot: int) -> tuple[list[Run], list[str]]:
    train, val, test = (_split(c23, s) for s in ("train", "val", "test"))
    runs, notes = [], []
    for groups in GROUP_SETS:
        reason = _missing_groups(c23, groups, C23)
        if reason:
            notes.append(f"{GROUP_LABELS[groups]}: {reason}")
            continue
        cols = model.feature_columns(c23, groups)
        for kind in kinds:
            run = _fit("E2", train, val, test, cols, groups, kind, n_boot)
            run.cv = model.grouped_cv(train, cols, kind, n_splits=CV_SPLITS, seed=SEED)
            runs.append(run)
    return runs, notes


def _run_transfer(
    experiment: str, c23: pd.DataFrame, target: pd.DataFrame, name: str, kinds: list[str],
    n_boot: int,
) -> tuple[list[Run], list[str]]:
    """Fit on c23 train, threshold on c23 val, score the ``split == "test"`` rows of ``target``."""
    train, val = _split(c23, "train"), _split(c23, "val")
    test = _split(target, "test")
    if not _both_labels(test):
        return [], [f"not run: {name} test rows do not hold both real and fake videos"]
    runs, notes = [], []
    for groups in GROUP_SETS:
        reason = _missing_groups(c23, groups, C23)
        if reason:
            notes.append(f"{GROUP_LABELS[groups]}: {reason}")
            continue
        cols = model.feature_columns(c23, groups)
        absent = [c for c in cols if c not in test.columns]
        if absent:
            notes.append(f"{GROUP_LABELS[groups]}: not run: {name} lacks column(s) {absent}")
            continue
        for kind in kinds:
            runs.append(_fit(experiment, train, val, test, cols, groups, kind, n_boot))
    return runs, notes


def _best(runs: list[Run], group_sets: tuple[str, ...] = GROUP_SETS) -> Run | None:
    """Highest validation AUC (ties: first in run order) among runs of ``group_sets``."""
    pool = [r for r in runs if r.groups in group_sets and np.isfinite(r.result["val_auc"])]
    return max(pool, key=lambda r: r.result["val_auc"]) if pool else None


def _headline(runs: list[Run]) -> Run | None:
    """The headline model: best validation AUC among the physiological sets only (ruling
    P2-R15). Group D describes recording quality, so a set holding it is never the detector."""
    return _best(runs, PHYSIO_SETS)


def _find(runs: list[Run], groups: str, kind: str) -> Run | None:
    return next((r for r in runs if r.groups == groups and r.kind == kind), None)


def _run_e6(c23: pd.DataFrame, e2: list[Run]) -> dict | None:
    best = _headline(e2)
    if best is None:
        return None
    train, val, test = (_split(c23, s) for s in ("train", "val", "test"))
    cols = model.feature_columns(c23, best.groups)
    fitted = best.result["model"]
    print(f"E6 importances ({best.groups} {best.kind})", flush=True)
    inputs = model.permutation_importances(fitted, test, cols, n_repeats=N_REPEATS, seed=SEED)
    groups = model.group_permutation_importances(
        fitted, test, cols, n_repeats=N_REPEATS, seed=SEED
    )
    base = _find(e2, ABLATION_BASE, best.kind)
    ablation = []
    if base is not None:
        for dropped in ABLATION_BASE:
            kept = "".join(g for g in ABLATION_BASE if g != dropped)
            cols_kept = model.feature_columns(c23, kept)
            result = model.fit_and_test(train, val, test, cols_kept, best.kind)
            ablation.append((dropped, kept, result["metrics"]["auc"]))
    return {"best": best, "inputs": inputs, "groups": groups, "base": base,
            "ablation": ablation}


# --- outputs ------------------------------------------------------------------------------------


def _curve_run(runs: list[Run], groups: str, headline: tuple[str, str] | None) -> Run | None:
    """The run drawn for ``groups``: the headline's own kind for the headline set, otherwise
    the set's model with the best validation AUC."""
    if headline is not None and headline[0] == groups:
        run = _find(runs, *headline)
        if run is not None:
            return run
    return _best(runs, (groups,))


def _roc_figure(runs: list[Run], title: str, path: Path,
                headline: tuple[str, str] | None) -> Path | None:
    """One ROC curve per group set; the headline (group set, kind) is the thick highlighted
    curve, the other sets use their best-validation model, the quality sets are dashed."""
    chosen = [r for g in GROUP_SETS if (r := _curve_run(runs, g, headline)) is not None]
    if not chosen:
        return None
    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=150)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=CHANCE_COLOR, label="chance")
    for run in chosen:
        if len(set(run.y_test)) < 2:
            continue
        fpr, tpr, _ = roc_curve(run.y_test, run.result["test_scores"])
        color = SERIES_COLORS[GROUP_SETS.index(run.groups)]
        is_headline = headline == (run.groups, run.kind)
        label = (f"{FIGURE_LABELS[run.groups]} ({run.kind}), AUC "
                 f"{_fmt(run.result['metrics']['auc'])}" + (", headline" if is_headline else ""))
        ax.plot(fpr, tpr, linewidth=3 if is_headline else 1.2, color=color, label=label,
                linestyle="-" if run.groups in PHYSIO_SETS else ":",
                zorder=3 if is_headline else 2)
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="false positive rate (reals called fake)",
           ylabel="true positive rate (fakes caught)", title=title)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _link(target: Path, md_out: Path) -> str:
    try:
        return Path(os.path.relpath(target, md_out.parent)).as_posix()
    except ValueError:  # different drives on Windows
        return target.resolve().as_posix()


def _counts_rows(name: str, df: pd.DataFrame) -> list[list[str]]:
    rows = []
    for split in ("train", "val", "test"):
        part = _split(df, split)
        if len(part):
            n_real, n_fake = int((part["label"] == 0).sum()), int((part["label"] == 1).sum())
            rows.append([name, split, str(n_real), str(n_fake), str(len(part))])
    return rows


def _inputs_section(tables: dict[str, pd.DataFrame | None], features_dir: Path) -> list[str]:
    out = ["## Inputs", "", f"Feature tables from `{features_dir.as_posix()}`.", ""]
    rows = []
    for name, df in tables.items():
        if df is not None:
            rows += _counts_rows(name, df)
    out += _md_table(["table", "split", "real", "fake", "total"], rows)
    out += [""]
    out += [f"- {name}: not found" for name, df in tables.items() if df is None]
    present = {name: df for name, df in tables.items() if df is not None}
    names = [n for g in FEATURE_GROUPS for n in FEATURE_GROUPS[g]]
    extra = sorted({c for df in present.values() for c in model.feature_columns(df, "ABCD")}
                   - set(names))
    out += ["", "### NaN rate per feature", "",
            ("Share of videos whose feature could not be computed (imputed with the training "
            "median inside the model). The `c_amp_*` map only counts windows in which all 7 skin "
            "regions are usable, so one unusable region in every window makes all 7 NaN."), ""]
    rows = []
    for feature in [*names, *extra]:
        cells = [f"{df[feature].isna().mean():.1%}" if feature in df.columns else "absent"
                 for df in present.values()]
        rows.append([feature, *cells])
    out += _md_table(["feature", *(n.removesuffix(".csv") for n in present)], rows)
    return out + [""]


def _result_rows(runs: list[Run], kinds_of_fakes: list[str], with_cv: bool) -> list[list[str]]:
    rows = []
    for run in runs:
        m = run.result["metrics"]
        row = [GROUP_LABELS[run.groups], run.kind, _fmt(run.result["val_auc"]),
               _fmt_ci(m["auc"], run.ci), _fmt(m["balanced_acc"]), _fmt(m["precision"]),
               _fmt(m["recall"])]
        if with_cv:
            row.append(_fmt_cv(run.cv))
        row += [_fmt(run.result["per_kind_auc"].get(k, float("nan"))) for k in kinds_of_fakes]
        rows.append(row)
    return rows


def _fmt_cv(cv: dict) -> str:
    return (f"{_fmt(cv['auc_mean'])} ± {_fmt(cv['auc_std'])} "
            f"({cv['n_folds_used']}/{CV_SPLITS} folds)")


def _fake_kinds(runs: list[Run]) -> list[str]:
    return sorted({k for r in runs for k in r.result["per_kind_auc"]})


def _results_table(runs: list[Run], with_cv: bool) -> list[str]:
    kinds = _fake_kinds(runs)
    header = ["groups", "model", "val AUC", "test AUC [95% CI]", "balanced acc", "precision",
              "recall"]
    if with_cv:
        header.append("CV AUC (train)")
    header += kinds
    return _md_table(header, _result_rows(runs, kinds, with_cv))


def _ci_note(runs: list[Run], n_boot: int) -> str:
    """How many bootstrap resamples held both classes, over the rows of one table."""
    used = [r.ci.n_used for r in runs if hasattr(r.ci, "n_used")]
    if not used:
        return ""
    span = str(min(used)) if min(used) == max(used) else f"{min(used)} to {max(used)}"
    return (f"95% CI: percentile bootstrap over whole leakage groups, {n_boot} resamples (seed "
            f"{SEED}); resamples missing a class are skipped, leaving {span} of {n_boot} per row.")


def _transfer_auc(experiment: str, name: str, result, best: Run) -> str:
    """The headline model's AUC in a transfer experiment, or why that experiment has none."""
    runs, notes = result
    run = _find(runs, best.groups, best.kind)
    if run is not None:
        return f"{experiment} {name} test AUC {_fmt_ci(run.result['metrics']['auc'], run.ci)}"
    reason = notes[0].split("not run: ")[-1] if notes else f"no {best.groups} {best.kind} model"
    return f"{experiment}: not run ({reason})"


def _headline_section(e2: list[Run], e3, e4, n_boot: int) -> list[str]:
    out = ["## Headline model", ""]
    best = _headline(e2)
    if best is None:
        return out + ["No headline model: E2 holds no A, AB or ABC model.", ""]
    m, cv = best.result["metrics"], best.cv
    parts = [
        f"Validation AUC {_fmt(best.result['val_auc'])}",
        f"E2 c23 test AUC {_fmt_ci(m['auc'], best.ci)} (95% CI)",
        (f"5-fold grouped CV on train {_fmt(cv['auc_mean'])} ± {_fmt(cv['auc_std'])} (mean ± "
         f"std, ddof=0, {cv['n_folds_used']}/{CV_SPLITS} folds)"),
        _transfer_auc("E3", "c40", e3, best),
        _transfer_auc("E4", "Celeb-DF", e4, best),
    ]
    out += [
        (f"Chosen by validation AUC among the physiological sets A, AB and ABC (never by test "
         f"AUC; sets with group D are checks, not candidates): **{best.groups} {best.kind}**. "
         + "; ".join(parts) + "."),
        "",
        (f"Confusion matrix on the c23 test split at the validation threshold "
         f"{_fmt(best.result['threshold'], 4)}:"),
        "",
        *_md_table(["", "predicted real", "predicted fake"],
                   [["real", str(m["tn"]), str(m["fp"])], ["fake", str(m["fn"]), str(m["tp"])]]),
        "",
    ]
    abc, a = _find(e2, "ABC", best.kind), _find(e2, "A", best.kind)
    if abc is not None and a is not None:
        delta, lo, hi = diff = model.grouped_bootstrap_auc_diff(
            abc.y_test, abc.result["test_scores"], a.result["test_scores"], abc.test_groups,
            n_boot, SEED)
        out += [(f"ABC − A on the c23 test split ({best.kind}): test AUC difference "
                 f"{delta:+.3f} (paired 95% CI {lo:+.3f} to {hi:+.3f}; both models scored on the "
                 f"same {diff.n_used} of {n_boot} grouped resamples). An interval that "
                 "includes 0 means the physiological groups B and C did not measurably add to "
                 "the classic baseline."
                 if np.isfinite(delta) and np.isfinite(lo) else
                 f"ABC − A on the c23 test split ({best.kind}): undefined (no usable resample)."),
                ""]
    else:
        out += [f"ABC − A: not computed (no A or ABC {best.kind} model in E2).", ""]
    return out


def _notes(notes: list[str]) -> list[str]:
    return [*(f"- {n}" for n in notes), ""] if notes else []


def _figure_line(path: Path | None, md_out: Path, alt: str) -> list[str]:
    return [f"![{alt}]({_link(path, md_out)})", ""] if path is not None else []


def _e6_section(e6: dict | None, reason: str | None) -> list[str]:
    out = ["## E6: What the headline model relies on", ""]
    if e6 is None:
        return out + [reason or "not run: no A, AB or ABC model in E2", ""]
    best = e6["best"]
    out += [
        (f"Model: {best.groups} {best.kind} (the headline model). Importance = drop in FF++ c23 "
        f"test AUC when the column(s) are shuffled across test videos (mean ± std over "
        f"{N_REPEATS} shuffles). The real-face reference distances `c_amp_mahal` and "
        "`c_lag_mahal` are computed inside the model from the raw `c_amp_*` / `c_lag_*` inputs, "
        "so they cannot be shuffled on their own: their effect appears through those raw "
        "inputs, and through group C as a whole."),
        "",
        "### Per feature group (all of a group's columns shuffled together)",
        "",
        *_md_table(["group", "AUC drop"],
                   [[r.feature, f"{_fmt(r.auc_drop_mean)} ± {_fmt(r.auc_drop_std)}"]
                    for r in e6["groups"].itertuples()]),
        "",
        f"### Top {TOP_FEATURES} inputs",
        "",
        ("Single-column `c_amp_*` ranks are inflated: the seven amplitude shares sum to 1, so "
         "shuffling one share alone builds maps no real face produces (off-manifold inputs). "
         "Read the per-group (whole-block) importance above for group C instead."),
        "",
        *_md_table(["rank", "feature", "AUC drop"],
                   [[str(i + 1), r.feature, f"{_fmt(r.auc_drop_mean)} ± {_fmt(r.auc_drop_std)}"]
                    for i, r in enumerate(e6["inputs"].head(TOP_FEATURES).itertuples())]),
        "",
        f"### Group ablation (refit {ABLATION_BASE} without one group, {best.kind})",
        "",
    ]
    base = e6["base"]
    if base is None:
        return out + [f"not run: no {ABLATION_BASE} {best.kind} model in E2", ""]
    full_auc = base.result["metrics"]["auc"]
    out += [
        *_md_table(["model", "test AUC", f"change vs {ABLATION_BASE}"],
                   [[ABLATION_BASE, _fmt(full_auc), ""]]
                   + [[f"{kept} (without {dropped})", _fmt(auc), _fmt(auc - full_auc)]
                      for dropped, kept, auc in e6["ablation"]]),
        "",
    ]
    return out


def _write_csvs(csv_out: Path, runs: list[Run], e6: dict | None) -> None:
    csv_out.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in runs:
        row = {"experiment": run.experiment, "groups": run.groups, "kind": run.kind,
               "threshold": run.result["threshold"], "val_auc": run.result["val_auc"],
               **run.result["metrics"], "auc_ci_lo": run.ci[0], "auc_ci_hi": run.ci[1],
               "auc_ci_n_used": getattr(run.ci, "n_used", 0)}
        row.update({f"cv_{k}": v for k, v in run.cv.items()})
        row.update({f"auc_{k}": v for k, v in run.result["per_kind_auc"].items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_out / "detection_results.csv", index=False)
    parts = []
    if e6 is not None:
        parts = [e6["inputs"].assign(level="feature"), e6["groups"].assign(level="group")]
    columns = ["level", "feature", "auc_drop_mean", "auc_drop_std"]
    table = pd.concat(parts, ignore_index=True)[columns] if parts else pd.DataFrame(columns=columns)
    table.to_csv(csv_out / "detection_importances.csv", index=False)


PROTOCOL = [
    "Protocol (labels: 0 = real, 1 = fake):",
    "",
    ("- Every model is fitted on the FF++ c23 official **train** split. All fitted statistics "
    "(the group-C real-face reference, fitted on training reals only; the median imputer; the "
    "scaler) are fitted inside the scikit-learn pipeline on those training rows."),
    ("- The decision threshold is Youden's J on the FF++ c23 **val** split; test rows are only "
    "scored. AUC is threshold-free; balanced accuracy, precision and recall use that threshold."),
    ("- FF++ leakage groups (an original pair and every fake made from it) never straddle "
    "train / val / test: the run checks this explicitly on the train, val and test frames "
    "before every final fit. CV folds are group-disjoint by construction (StratifiedGroupKFold, "
    "covered by the tests)."),
    ("- The headline model is the one with the best validation AUC among the physiological sets "
    "A, AB and ABC, never chosen by test AUC."),
    ("- Group D (clip duration, window count, brightness, face-found ratio, face size, motion) "
    "describes recording quality, not physiology. It is reported only to expose shortcuts: the "
    "\"D only\" row measures how much a classifier gets from quality alone, the ABCD row is a "
    "confound check. Neither is ever the headline."),
    ("- Group A at 4 s windows is a weakened classic baseline: SNR saturates near 0 dB, there is "
    "no 8 s variant and no peak-sharpness feature, and it includes one coverage feature "
    "(`a_valid_frac`, the share of windows with a usable face signal, which measures tracking, "
    "not pulse)."),
    ("- 95% CIs come from a percentile bootstrap that resamples whole leakage groups (an FF++ "
    "pair with its fakes, a Celeb-DF identity); the ABC − A interval is paired (both models on "
    "the same resamples)."),
    "- Per-manipulation AUC: that manipulation's test fakes against all test reals.",
    "",
]

PER_KIND_NOTE = ("Per-manipulation AUCs rest on few videos: on the full FF++ test split each is "
                 "140 reals against 140 fakes, a standard error of about 0.02–0.03 for AUCs "
                 "between 0.7 and 0.9 (Hanley and McNeil), so small differences between "
                 "manipulations are noise.")


def _write_markdown(md_out: Path, command: str, args, tables, e2: list[Run], e2_notes,
                    e3, e4, e6: dict | None, e6_reason: str | None, figures) -> None:
    out = ["# Detection results", "", f"Run date: {date.today().isoformat()}", "",
           f"Commit: {_git_sha()}", "", f"Versions: {_versions()}", "",
           "Command:", "", "```", command, "```", "", *PROTOCOL]
    out += _inputs_section(tables, args.features_dir)
    out += _headline_section(e2, e3, e4, args.n_boot)

    out += ["## E2: FF++ c23, train → val → test", ""]
    if e2:
        out += [("Test = the c23 test split. CV AUC: mean ± std (ddof=0) of the test-fold AUC "
                f"over {CV_SPLITS}-fold stratified grouped CV on the train split (a fold whose "
                "test part holds one class is skipped and not counted). "
                + _ci_note(e2, args.n_boot)), ""]
        out += [*_results_table(e2, with_cv=True), "", PER_KIND_NOTE, ""]
    out += _notes(e2_notes)
    out += _figure_line(figures.get("e2"), md_out, "ROC, FF++ c23 test")

    out += ["## E3: Trained on FF++ c23, tested on FF++ c40", ""]
    out += _transfer_lines(e3, "the c40 test split", args.n_boot)
    out += ["## E4: Trained on FF++ c23, tested on Celeb-DF", ""]
    out += _transfer_lines(e4, "the Celeb-DF test list", args.n_boot)
    out += _figure_line(figures.get("e4"), md_out, "ROC, Celeb-DF test")
    out += _e6_section(e6, e6_reason)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(out), encoding="utf-8")


def _transfer_lines(result: tuple[list[Run], list[str]], test_name: str,
                    n_boot: int) -> list[str]:
    runs, notes = result
    out = []
    if runs:
        out += [(f"Test = {test_name}; threshold from the FF++ c23 val split, so threshold "
                "metrics include the domain shift. " + _ci_note(runs, n_boot)), "",
                *_results_table(runs, with_cv=False), ""]
    return out + _notes(notes)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--features-dir", type=Path, help="default: <data dir>/features")
    p.add_argument("--kind", nargs="+", choices=model.MODEL_KINDS, default=list(model.MODEL_KINDS))
    p.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    p.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    p.add_argument("--csv-out", type=Path, help="CSV directory; default: <data dir>/results")
    p.add_argument("--n-boot", type=int, default=N_BOOT,
                   help=f"bootstrap resamples per confidence interval; default {N_BOOT}")
    args = p.parse_args(argv)
    if args.n_boot < 1:
        p.error("--n-boot must be at least 1")
    args.features_dir = args.features_dir or data_dir() / "features"
    csv_out = args.csv_out or data_dir() / "results"

    tables: dict[str, pd.DataFrame | None] = {}
    try:
        for name in TABLES:
            path = args.features_dir / name
            tables[name] = _load(path) if path.is_file() else None
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    c23 = tables[C23]
    if c23 is None:
        print(f"error: {args.features_dir / C23} not found; every experiment trains on it, so "
              "nothing was run or written", file=sys.stderr)
        return 1
    problem = _ffpp_problem(c23)
    if problem:
        print(f"error: {problem}; every experiment trains, thresholds and tests on it, so "
              "nothing was run or written", file=sys.stderr)
        return 1

    e2, e2_notes = _run_e2(c23, args.kind, args.n_boot)
    e3 = (_run_transfer("E3", c23, tables[C40], C40, args.kind, args.n_boot)
          if tables[C40] is not None else ([], [f"not run: missing {C40}"]))
    e4 = (_run_transfer("E4", c23, tables[CELEBDF], CELEBDF, args.kind, args.n_boot)
          if tables[CELEBDF] is not None else ([], [f"not run: missing {CELEBDF}"]))
    e6 = _run_e6(c23, e2) if e2 else None
    e6_reason = None if e6 is not None else "not run: no A, AB or ABC model in E2"

    best = _headline(e2)
    headline = (best.groups, best.kind) if best is not None else None
    figures = {
        "e2": _roc_figure(e2, "FF++ c23 test", args.fig_dir / "roc_e2.png", headline),
        "e4": _roc_figure(e4[0], "Celeb-DF test (trained on FF++ c23)",
                          args.fig_dir / "roc_e4.png", headline),
    }
    command = " ".join(["python", "scripts/run_experiments.py", *map(_quote, argv)])
    _write_markdown(args.md_out, command, args, tables, e2, e2_notes, e3, e4, e6, e6_reason,
                    figures)
    _write_csvs(csv_out, [*e2, *e3[0], *e4[0]], e6)
    print(f"wrote {args.md_out}\nwrote {csv_out / 'detection_results.csv'}")
    for path in figures.values():
        if path is not None:
            print(f"wrote {path}")
    return 0 if e2 else 1


if __name__ == "__main__":
    sys.exit(main())
