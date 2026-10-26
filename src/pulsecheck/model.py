"""Detector models and their leak-free evaluation.

A model is a scikit-learn ``Pipeline`` over the per-video feature table (one row per video,
labels 0 = real, 1 = fake):

    [RealFaceReference] -> median imputer -> [StandardScaler] -> classifier

Every fitted statistic (the real-face reference of group C, the imputer's medians, the scaler)
is fitted inside the pipeline on the rows passed to ``fit`` only, so cross-validation and the
train -> val -> test protocol never let a held-out video shape the model that scores it. The
decision threshold is chosen on validation scores (Youden's J), never on test scores.

Scores are the predicted probability of "fake"; a video is called fake when its score is
``>= threshold``. Metrics that are undefined (AUC with one class present, precision with nothing
predicted fake, ...) are NaN rather than a warning or an arbitrary 0.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pulsecheck.features import FEATURE_GROUPS
from pulsecheck.reference import DISTANCE_SUFFIX, RealFaceReference
from pulsecheck.splits import check_no_leakage_df, grouped_folds_df

FEATURE_PATTERN = re.compile(r"^[abcd]_")  # feature columns: group letter + "_" (ruling P2-R2)
MODEL_KINDS = ("logreg", "hgb")  # accepted by make_model
REFERENCE_PREFIXES = ("c_amp_", "c_lag_")  # map blocks that the real-face reference reads
REFERENCE_COLUMNS = tuple(p + DISTANCE_SUFFIX for p in REFERENCE_PREFIXES)  # added in the model
ROLES = ("train", "val", "test")  # the three frames of fit_and_test, in order


def _check_groups(groups: str) -> set[str]:
    unknown = sorted(set(groups) - set(FEATURE_GROUPS))
    if unknown or not groups:
        raise ValueError(f"unknown feature group(s) {unknown or groups!r}; "
                         f"choose from {list(FEATURE_GROUPS)}")
    return {g.lower() for g in groups}


def feature_columns(df: pd.DataFrame, groups: str) -> list[str]:
    """Columns of ``df`` matching ``^[abcd]_`` whose letter is one of ``groups`` (e.g. "ABC").

    Table order is kept. Manifest fields (``dataset``, ``compression``, ...) never match.
    Raises ``ValueError`` for a group letter not in ``FEATURE_GROUPS``.
    """
    letters = _check_groups(groups)
    return [c for c in df.columns if FEATURE_PATTERN.match(c) and c[0] in letters]


def make_model(kind: str, columns: list[str]) -> Pipeline:
    """Unfitted pipeline that reads ``columns`` of a feature table; pandas output throughout.

    Steps: ``reference`` (``RealFaceReference``, only if a ``c_amp_``/``c_lag_`` column is in
    ``columns``; appends ``c_amp_mahal`` and ``c_lag_mahal``), ``impute`` (median
    ``SimpleImputer`` with ``keep_empty_features=True`` inside a ``ColumnTransformer``: an
    all-NaN training column becomes 0 instead of disappearing, and any column not listed is
    dropped), ``scale`` (``StandardScaler``, logreg only) and ``clf``. ``kind`` is "logreg"
    (balanced ``LogisticRegression``) or "hgb" (balanced ``HistGradientBoostingClassifier``,
    ``random_state=0``). Raises ``ValueError`` for another kind.
    """
    if kind == "logreg":
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    elif kind == "hgb":
        clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=0)
    else:
        raise ValueError(f"unknown model kind {kind!r}; choose from {MODEL_KINDS}")
    columns = list(columns)
    steps = []
    if any(c.startswith(REFERENCE_PREFIXES) for c in columns):
        steps.append(("reference", RealFaceReference(*REFERENCE_PREFIXES)))
        columns += list(REFERENCE_COLUMNS)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    steps.append(("impute", ColumnTransformer(
        [("impute", imputer, columns)], remainder="drop", verbose_feature_names_out=False
    )))
    if kind == "logreg":
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps).set_output(transform="pandas")


def scores_of(fitted: Pipeline, df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Predicted probability of "fake" for each row of ``df`` (shape ``(n,)``)."""
    return fitted.predict_proba(df[columns])[:, 1]


def _auc(y_true, scores) -> float:
    """ROC-AUC, or NaN when only one class is present."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def youden_threshold(y_true, scores) -> float:
    """Score threshold maximising Youden's J = TPR - FPR (fake when ``score >= threshold``).

    Candidates are the distinct scores (``roc_curve`` without dropping points, minus its
    "nothing is fake" +inf point). Ties go to the highest threshold. Raises ``ValueError`` unless
    both classes are present.
    """
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Youden threshold needs both real and fake rows")
    fpr, tpr, thresholds = roc_curve(y_true, scores, drop_intermediate=False)
    j = (tpr - fpr)[1:]
    return float(thresholds[1:][np.argmax(j)])


def _ratio(num: int, den: int) -> float:
    return num / den if den else float("nan")


def metrics(y_true, scores, threshold: float) -> dict[str, float]:
    """Test metrics at ``threshold`` (fake when ``score >= threshold``).

    Keys: ``auc`` (threshold-free), ``balanced_acc`` (mean of the recalls of the classes
    present), ``precision``, ``recall`` (of fakes), ``n_real``, ``n_fake`` and the confusion
    matrix ``tn, fp, fn, tp`` (real = negative). Undefined values are NaN.
    """
    y = np.asarray(y_true).astype(int)
    pred = np.asarray(scores) >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    present = [r for r in (recall, specificity) if np.isfinite(r)]
    return {
        "auc": _auc(y, scores),
        "balanced_acc": float(np.mean(present)) if present else float("nan"),
        "precision": _ratio(tp, tp + fp),
        "recall": recall,
        "n_real": tn + fp,
        "n_fake": tp + fn,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def grouped_cv(
    df: pd.DataFrame, columns: list[str], kind: str, n_splits: int = 5, seed: int = 0
) -> dict[str, float]:
    """Mean and std (``ddof=0``) of the test-fold AUC over stratified grouped folds.

    Folds come from ``splits.grouped_folds_df`` (``StratifiedGroupKFold`` on ``df.label`` /
    ``df.group``; raises ``ValueError`` when a label has fewer than ``n_splits`` groups). A
    fresh model is fitted on each training part. A fold whose test or training part holds only
    one class is skipped (AUC or fit undefined); ``n_folds_used`` counts the folds that
    contributed. With no usable fold, mean and std are NaN.
    """
    y = df["label"].to_numpy(dtype=int)
    aucs = []
    for train_idx, test_idx in grouped_folds_df(df, n_splits=n_splits, seed=seed):
        if len(np.unique(y[test_idx])) < 2 or len(np.unique(y[train_idx])) < 2:
            continue
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        fitted = make_model(kind, columns).fit(train[columns], y[train_idx])
        aucs.append(_auc(y[test_idx], scores_of(fitted, test, columns)))
    if not aucs:
        return {"auc_mean": float("nan"), "auc_std": float("nan"), "n_folds_used": 0}
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "n_folds_used": len(aucs),
    }


def check_roles(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if a leakage group appears in more than one of the three frames.

    The frames' roles (train / val / test) are checked, not their ``split`` column, so passing
    the same rows twice, or c23 training rows next to their c40 copies as test, is caught too.
    """
    roles = [f[["group"]].assign(split=role)
             for f, role in zip((train_df, val_df, test_df), ROLES, strict=True)]
    check_no_leakage_df(pd.concat(roles, ignore_index=True))


def fit_and_test(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list[str],
    kind: str,
) -> dict:
    """Fit on ``train_df``, pick the threshold on ``val_df``, score ``test_df``.

    Checks for leakage between the three frames first (``check_roles``). Returns a dict:

    * ``metrics``: ``metrics(test labels, test scores, threshold)``
    * ``per_kind_auc``: ``{fake kind: AUC of that kind's test fakes against ALL test reals}``
    * ``model``: the fitted pipeline; ``threshold``: the validation Youden threshold
    * ``val_auc``: validation AUC (for choosing between models without looking at test)
    * ``test_scores``: test scores in ``test_df`` row order (shape ``(n_test,)``)
    """
    check_roles(train_df, val_df, test_df)
    fitted = make_model(kind, columns).fit(train_df[columns], train_df["label"].to_numpy(int))
    val_scores = scores_of(fitted, val_df, columns)
    threshold = youden_threshold(val_df["label"], val_scores)
    test_scores = scores_of(fitted, test_df, columns)
    y_test = test_df["label"].to_numpy(int)
    reals = y_test == 0
    kinds = test_df["kind"].to_numpy()
    per_kind = {
        str(k): _auc(y_test[reals | (kinds == k)], test_scores[reals | (kinds == k)])
        for k in sorted(set(kinds[~reals]))
    }
    return {
        "metrics": metrics(y_test, test_scores, threshold),
        "per_kind_auc": per_kind,
        "model": fitted,
        "threshold": threshold,
        "val_auc": _auc(val_df["label"], val_scores),
        "test_scores": test_scores,
    }


def _permutation_drops(
    fitted: Pipeline,
    df: pd.DataFrame,
    columns: list[str],
    blocks: dict[str, list[str]],
    n_repeats: int,
    seed: int,
) -> pd.DataFrame:
    """AUC drop when each block's columns are shuffled together (one row permutation per repeat,
    applied to every column of the block, so the block's inner structure is kept)."""
    y = df["label"].to_numpy(int)
    X = df[columns].reset_index(drop=True)
    base = _auc(y, fitted.predict_proba(X)[:, 1])
    rng = np.random.default_rng(seed)
    rows = []
    for name, block in blocks.items():
        drops = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            order = rng.permutation(len(X))
            shuffled[block] = X[block].to_numpy()[order]
            drops.append(base - _auc(y, fitted.predict_proba(shuffled)[:, 1]))
        rows.append({"feature": name, "auc_drop_mean": float(np.mean(drops)),
                     "auc_drop_std": float(np.std(drops))})
    out = pd.DataFrame(rows, columns=["feature", "auc_drop_mean", "auc_drop_std"])
    return out.sort_values("auc_drop_mean", ascending=False, kind="stable").reset_index(drop=True)


def permutation_importances(
    fitted: Pipeline, test_df: pd.DataFrame, columns: list[str], n_repeats: int = 10, seed: int = 0
) -> pd.DataFrame:
    """Per input column: mean / std (over ``n_repeats`` shuffles) of the drop in test AUC.

    Columns ``feature, auc_drop_mean, auc_drop_std``, sorted by the mean drop, largest first.
    Only input columns can be shuffled: ``c_amp_mahal`` / ``c_lag_mahal`` are computed inside
    the model from the ``c_amp_*`` / ``c_lag_*`` inputs, so their effect shows up there.
    """
    blocks = {c: [c] for c in columns}
    return _permutation_drops(fitted, test_df, columns, blocks, n_repeats, seed)


def group_permutation_importances(
    fitted: Pipeline, test_df: pd.DataFrame, columns: list[str], n_repeats: int = 10, seed: int = 0
) -> pd.DataFrame:
    """Like ``permutation_importances`` but per feature group: all of a group's columns are
    shuffled together. The ``feature`` column holds the group letter."""
    blocks = {
        g: [c for c in columns if c[0] == g.lower()]
        for g in FEATURE_GROUPS
        if any(c[0] == g.lower() for c in columns)
    }
    return _permutation_drops(fitted, test_df, columns, blocks, n_repeats, seed)
