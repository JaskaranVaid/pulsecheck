"""Train and evaluate the deepfake detector on the per-video feature tables.

    python scripts/run_experiments.py --features-dir C:/dev/pulsecheck-data/features

Reads ``ffpp_c23.csv``, ``ffpp_c40.csv`` and ``celebdf_test.csv`` (written by
``build_features.py``) from ``--features-dir`` and runs every experiment whose tables exist:

* E2: FF++ c23, fit on the official train split, threshold on val, scored on test, for the
  feature groups A, AB, ABC and ABCD; test AUC per manipulation; 5-fold grouped CV on train.
* E3: the same models, scored on the FF++ c40 test split (heavier compression).
* E4: the same models, scored on the Celeb-DF test list (a dataset never trained on).
* E6: permutation importance (top 15 inputs, and per feature group) of the best ABCD model, and a
  group ablation (refit without each group).

Every experiment needs ``ffpp_c23.csv`` (training and validation); without it nothing is
written and the exit code is 1. An experiment whose other table is missing gets a
"not run: missing <file>" line. "Best" models are chosen by validation AUC, never by test AUC.
Writes the Markdown summary, ROC figures ``roc_e2.png`` / ``roc_e4.png`` and two CSVs
(``detection_results.csv``, ``detection_importances.csv``).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import date
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

DEFAULT_MD_OUT = Path("docs/results/detection.md")
DEFAULT_FIG_DIR = Path("docs/figures")
C23, C40, CELEBDF = "ffpp_c23.csv", "ffpp_c40.csv", "celebdf_test.csv"  # feature tables
TABLES = (C23, C40, CELEBDF)
GROUP_SETS = ("A", "AB", "ABC", "ABCD")  # feature groups of the E2/E3/E4 models
REQUIRED = ("group", "split", "label", "kind")  # table columns the experiments read
CV_SPLITS = 5  # grouped CV folds on the c23 train split
TOP_FEATURES = 15  # E6: inputs listed
N_REPEATS = 10  # E6: shuffles per permutation importance
SEED = 0  # CV folds and permutation shuffles
LABEL_NAMES = {0: "real", 1: "fake"}
# Categorical series colours, one per group set in GROUP_SETS order (fixed, never cycled)
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
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


def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _quote(arg: str) -> str:
    return f'"{arg}"' if not arg or any(c.isspace() for c in arg) else arg


def _md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


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
            return f"not run: {C23} split {split!r} does not hold both real and fake rows"
    return None


def _run_e2(c23: pd.DataFrame, kinds: list[str]) -> tuple[list[Run], list[str]]:
    train, val, test = (_split(c23, s) for s in ("train", "val", "test"))
    runs, notes = [], []
    for groups in GROUP_SETS:
        reason = _missing_groups(c23, groups, C23)
        if reason:
            notes.append(f"{groups}: {reason}")
            continue
        cols = model.feature_columns(c23, groups)
        for kind in kinds:
            print(f"E2 {groups} {kind}", flush=True)
            result = model.fit_and_test(train, val, test, cols, kind)
            cv = model.grouped_cv(train, cols, kind, n_splits=CV_SPLITS, seed=SEED)
            runs.append(Run("E2", groups, kind, result, test["label"].to_numpy(int), cv))
    return runs, notes


def _run_transfer(
    experiment: str, c23: pd.DataFrame, target: pd.DataFrame, name: str, kinds: list[str]
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
            notes.append(f"{groups}: {reason}")
            continue
        cols = model.feature_columns(c23, groups)
        absent = [c for c in cols if c not in test.columns]
        if absent:
            notes.append(f"{groups}: not run: {name} lacks column(s) {absent}")
            continue
        for kind in kinds:
            print(f"{experiment} {groups} {kind}", flush=True)
            result = model.fit_and_test(train, val, test, cols, kind)
            runs.append(Run(experiment, groups, kind, result, test["label"].to_numpy(int)))
    return runs, notes


def _best(runs: list[Run], groups: str | None = None) -> Run | None:
    """Highest validation AUC (ties: first in run order), optionally within one group set."""
    pool = [r for r in runs if groups is None or r.groups == groups]
    pool = [r for r in pool if np.isfinite(r.result["val_auc"])]
    return max(pool, key=lambda r: r.result["val_auc"]) if pool else None


def _run_e6(c23: pd.DataFrame, e2: list[Run]) -> dict | None:
    best = _best(e2, "ABCD")
    if best is None:
        return None
    train, val, test = (_split(c23, s) for s in ("train", "val", "test"))
    cols = model.feature_columns(c23, "ABCD")
    fitted = best.result["model"]
    print(f"E6 importances ({best.kind})", flush=True)
    inputs = model.permutation_importances(fitted, test, cols, n_repeats=N_REPEATS, seed=SEED)
    groups = model.group_permutation_importances(
        fitted, test, cols, n_repeats=N_REPEATS, seed=SEED
    )
    ablation = []
    for dropped in "ABCD":
        kept = "".join(g for g in "ABCD" if g != dropped)
        result = model.fit_and_test(train, val, test, model.feature_columns(c23, kept), best.kind)
        ablation.append((dropped, kept, result["metrics"]["auc"]))
    return {"best": best, "inputs": inputs, "groups": groups, "ablation": ablation}


# --- outputs ------------------------------------------------------------------------------------


def _roc_figure(runs: list[Run], title: str, path: Path) -> Path | None:
    """One ROC curve per group set: its model with the best validation AUC."""
    chosen = [r for g in GROUP_SETS if (r := _best(runs, g)) is not None]
    if not chosen:
        return None
    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=150)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=CHANCE_COLOR, label="chance")
    for run in chosen:
        if len(set(run.y_test)) < 2:
            continue
        fpr, tpr, _ = roc_curve(run.y_test, run.result["test_scores"])
        color = SERIES_COLORS[GROUP_SETS.index(run.groups)]
        label = f"{run.groups} ({run.kind}), AUC {_fmt(run.result['metrics']['auc'])}"
        ax.plot(fpr, tpr, linewidth=2, color=color, label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="false positive rate (reals called fake)",
           ylabel="true positive rate (fakes caught)", title=title)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
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
        row = [run.groups, run.kind, _fmt(run.result["val_auc"]), _fmt(m["auc"]),
               _fmt(m["balanced_acc"]), _fmt(m["precision"]), _fmt(m["recall"])]
        if with_cv:
            cv = run.cv
            row.append(f"{_fmt(cv['auc_mean'])} ± {_fmt(cv['auc_std'])} "
                       f"({cv['n_folds_used']}/{CV_SPLITS} folds)")
        row += [_fmt(run.result["per_kind_auc"].get(k, float("nan"))) for k in kinds_of_fakes]
        rows.append(row)
    return rows


def _fake_kinds(runs: list[Run]) -> list[str]:
    return sorted({k for r in runs for k in r.result["per_kind_auc"]})


def _results_table(runs: list[Run], with_cv: bool) -> list[str]:
    kinds = _fake_kinds(runs)
    header = ["groups", "model", "val AUC", "test AUC", "balanced acc", "precision", "recall"]
    if with_cv:
        header.append("CV AUC (train)")
    header += kinds
    return _md_table(header, _result_rows(runs, kinds, with_cv))


def _confusion(run: Run) -> list[str]:
    m = run.result["metrics"]
    return [
        (f"Best E2 model by validation AUC: **{run.groups} {run.kind}** (val AUC "
        f"{_fmt(run.result['val_auc'])}). Its confusion matrix on the c23 test split at the "
        f"validation threshold {_fmt(run.result['threshold'], 4)}:"),
        "",
        *_md_table(["", "predicted real", "predicted fake"],
                   [["real", str(m["tn"]), str(m["fp"])], ["fake", str(m["fn"]), str(m["tp"])]]),
    ]


def _notes(notes: list[str]) -> list[str]:
    return [*(f"- {n}" for n in notes), ""] if notes else []


def _figure_line(path: Path | None, md_out: Path, alt: str) -> list[str]:
    return [f"![{alt}]({_link(path, md_out)})", ""] if path is not None else []


def _e6_section(e6: dict | None, reason: str | None) -> list[str]:
    out = ["## E6: What the best ABCD model relies on", ""]
    if e6 is None:
        return out + [reason or "not run: no ABCD model in E2", ""]
    best = e6["best"]
    full_auc = best.result["metrics"]["auc"]
    out += [
        (f"Model: ABCD {best.kind} (the ABCD model with the best validation AUC). Importance = "
        f"drop in FF++ c23 test AUC when the column(s) are shuffled across test videos (mean ± "
        f"std over {N_REPEATS} shuffles). The real-face reference distances `c_amp_mahal` and "
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
        *_md_table(["rank", "feature", "AUC drop"],
                   [[str(i + 1), r.feature, f"{_fmt(r.auc_drop_mean)} ± {_fmt(r.auc_drop_std)}"]
                    for i, r in enumerate(e6["inputs"].head(TOP_FEATURES).itertuples())]),
        "",
        f"### Group ablation (refit without one group, {best.kind})",
        "",
        *_md_table(["model", "test AUC", "change vs ABCD"],
                   [["ABCD", _fmt(full_auc), ""]]
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
               **run.result["metrics"]}
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
    "train / val / test or CV folds; the run checks this before every fit."),
    "- \"Best\" models are chosen by validation AUC, never by test AUC.",
    "- Per-manipulation AUC: that manipulation's test fakes against all test reals.",
    "",
]


def _write_markdown(md_out: Path, command: str, args, tables, e2: list[Run], e2_notes,
                    e3, e4, e6: dict | None, e6_reason: str | None, figures) -> None:
    out = ["# Detection results", "", f"Run date: {date.today().isoformat()}", "",
           "Command:", "", "```", command, "```", "", *PROTOCOL]
    out += _inputs_section(tables, args.features_dir)

    out += ["## E2: FF++ c23, train → val → test", ""]
    if e2:
        out += [("Test = the c23 test split. CV AUC: mean ± std of the test-fold AUC over "
                f"{CV_SPLITS}-fold stratified grouped CV on the train split (a fold whose test "
                "part holds one class is skipped and not counted)."), ""]
        out += [*_results_table(e2, with_cv=True), ""]
    out += _notes(e2_notes)
    best = _best(e2)
    if best is not None:
        out += [*_confusion(best), ""]
    out += _figure_line(figures.get("e2"), md_out, "ROC, FF++ c23 test")

    out += ["## E3: Trained on FF++ c23, tested on FF++ c40", ""]
    out += _transfer_lines(e3, "the c40 test split")
    out += ["## E4: Trained on FF++ c23, tested on Celeb-DF", ""]
    out += _transfer_lines(e4, "the Celeb-DF test list")
    out += _figure_line(figures.get("e4"), md_out, "ROC, Celeb-DF test")
    out += _e6_section(e6, e6_reason)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(out), encoding="utf-8")


def _transfer_lines(result: tuple[list[Run], list[str]], test_name: str) -> list[str]:
    runs, notes = result
    out = []
    if runs:
        out += [(f"Test = {test_name}; threshold from the FF++ c23 val split, so threshold "
                "metrics include the domain shift."), "",
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
    args = p.parse_args(argv)
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
        e2, e2_notes = [], [problem]
        e3 = e4 = ([], [problem])
    else:
        e2, e2_notes = _run_e2(c23, args.kind)
        e3 = (_run_transfer("E3", c23, tables[C40], C40, args.kind) if tables[C40] is not None
              else ([], [f"not run: missing {C40}"]))
        e4 = (_run_transfer("E4", c23, tables[CELEBDF], CELEBDF, args.kind)
              if tables[CELEBDF] is not None else ([], [f"not run: missing {CELEBDF}"]))
    e6_reason = None
    e6 = _run_e6(c23, e2) if e2 else None
    if e6 is None:
        e6_reason = "not run: no ABCD model in E2" if not problem else problem

    figures = {
        "e2": _roc_figure(e2, "FF++ c23 test", args.fig_dir / "roc_e2.png"),
        "e4": _roc_figure(e4[0], "Celeb-DF test (trained on FF++ c23)",
                          args.fig_dir / "roc_e4.png"),
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
