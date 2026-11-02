"""Detector models and their leak-free evaluation (``pulsecheck.model``)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from pulsecheck import model
from pulsecheck.reference import RealFaceReference
from synth import synth_ffpp_table

MANIFEST = ["dataset", "path", "rel_path", "label", "group", "kind", "compression", "split"]


def _split(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return tuple(table[table["split"] == s] for s in ("train", "val", "test"))


# --- feature columns and model construction ---------------------------------------------------


def test_feature_columns_never_manifest_fields():
    df = pd.DataFrame(columns=[*MANIFEST, "a_snr_face", "b_coherence", "c_amp_nose",
                               "c_lag_chin", "d_motion", "e_other", "abc", "A_upper"])
    assert model.feature_columns(df, "ABCD") == [
        "a_snr_face", "b_coherence", "c_amp_nose", "c_lag_chin", "d_motion"
    ]
    assert model.feature_columns(df, "A") == ["a_snr_face"]
    assert model.feature_columns(df, "AC") == ["a_snr_face", "c_amp_nose", "c_lag_chin"]
    # "compression" and "dataset" start with c / d but are never features.
    assert not set(model.feature_columns(df, "ABCD")) & set(MANIFEST)
    with pytest.raises(ValueError, match="unknown"):
        model.feature_columns(df, "AX")


def test_make_model_kinds_and_unknown():
    with_c = ["a_snr_face", "c_amp_nose", "c_lag_chin"]
    logreg = model.make_model("logreg", with_c)
    assert [name for name, _ in logreg.steps] == ["reference", "impute", "scale", "clf"]
    assert isinstance(logreg["reference"], RealFaceReference)
    clf = logreg["clf"]
    assert isinstance(clf, LogisticRegression)
    assert clf.class_weight == "balanced" and clf.max_iter == 2000

    hgb = model.make_model("hgb", with_c)
    assert [name for name, _ in hgb.steps] == ["reference", "impute", "clf"]
    assert isinstance(hgb["clf"], HistGradientBoostingClassifier)
    assert hgb["clf"].class_weight == "balanced" and hgb["clf"].random_state == 0

    # No group-C map column: no reference step.
    no_c = model.make_model("logreg", ["a_snr_face", "b_coherence", "c_harmonic_face"])
    assert [name for name, _ in no_c.steps] == ["impute", "scale", "clf"]

    imputer = logreg["impute"].transformers[0][1]
    assert imputer.strategy == "median" and imputer.keep_empty_features
    with pytest.raises(ValueError, match="unknown model kind"):
        model.make_model("svm", with_c)


def test_make_model_imputes_and_outputs_only_requested_columns():
    rng = np.random.default_rng(0)
    cols = ["a_snr_face", "c_amp_nose", "c_lag_chin", "c_amp_chin", "d_motion"]
    X = pd.DataFrame(rng.normal(size=(30, len(cols))), columns=cols)
    X.loc[0, "a_snr_face"] = np.nan
    X["d_motion"] = np.nan  # an all-NaN column is kept (filled with 0), not dropped
    y = np.r_[np.zeros(15), np.ones(15)]
    pipe = model.make_model("logreg", cols).fit(X, y)
    seen = pipe[:-1].transform(X)
    assert list(seen.columns) == [*cols, "c_amp_mahal", "c_lag_mahal"]
    assert seen.notna().all().all()
    assert pipe.predict_proba(X).shape == (30, 2)


# --- thresholds and metrics -------------------------------------------------------------------

# Hand-checked: reals score [.1 .2 .5 .6], fakes [.4 .7 .8 .9]. Youden's J = TPR - FPR per
# threshold (predict fake when score >= t): .9 -> .25, .8 -> .50, .7 -> .75, .6 -> .50,
# .5 -> .25, .4 -> .50, .2 -> .25, .1 -> 0. The unique maximum is at t = 0.7.
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
S = np.array([0.1, 0.2, 0.5, 0.6, 0.4, 0.7, 0.8, 0.9])


def test_youden_and_metrics_exact():
    assert model.youden_threshold(Y, S) == pytest.approx(0.7)

    m = model.metrics(Y, S, 0.7)
    assert set(m) == {"auc", "balanced_acc", "precision", "recall", "n_real", "n_fake",
                      "tn", "fp", "fn", "tp"}
    # AUC: 14 of the 16 (fake, real) pairs are ranked correctly (.4 is below .5 and .6).
    assert m["auc"] == pytest.approx(14 / 16)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (4, 0, 1, 3)
    assert (m["n_real"], m["n_fake"]) == (4, 4)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(0.75)
    assert m["balanced_acc"] == pytest.approx((1.0 + 0.75) / 2)

    lower = model.metrics(Y, S, 0.45)  # .5 and .6 now count as fake
    assert (lower["tn"], lower["fp"], lower["fn"], lower["tp"]) == (2, 2, 1, 3)
    assert lower["precision"] == pytest.approx(3 / 5)
    assert lower["balanced_acc"] == pytest.approx((0.5 + 0.75) / 2)
    assert lower["auc"] == m["auc"]  # AUC does not depend on the threshold


# --- grouped bootstrap confidence intervals ---------------------------------------------------

# Y / S above, paired into four leakage groups of one real and one fake each.
PAIR_GROUPS = np.array(["g0", "g1", "g2", "g3", "g0", "g1", "g2", "g3"])


def test_grouped_bootstrap_auc_contains_point_and_is_seeded():
    # Hand-checked: the pairs are g0 (.1, .4), g1 (.2, .7), g2 (.5, .8), g3 (.6, .9). Only g0's
    # fake (.4) is ever ranked below a real (.5 of g2, .6 of g3). The worst resample draws g0
    # twice and two of g2/g3: 4 of 16 pairs wrong, AUC 0.75 (24 of the 256 equally likely draws,
    # 9.4% > 2.5%, so it is the lower bound). No g0 at all gives AUC 1 (31.6% > 2.5%).
    ci = model.grouped_bootstrap_auc(Y, S, PAIR_GROUPS, n_boot=200, seed=0)
    lo, hi = ci  # a plain (lo, hi) pair ...
    assert ci.n_used == 200  # ... that also counts the usable resamples: every group has both
    assert (lo, hi) == (0.75, 1.0)
    assert lo < 14 / 16 < hi  # the point AUC is inside
    assert model.grouped_bootstrap_auc(Y, S, PAIR_GROUPS, n_boot=200, seed=0) == ci
    # The seed drives the resamples: the same seed repeats them, another changes them.
    def draws(seed):
        return [tuple(i) for i in model.grouped_bootstrap_indices(PAIR_GROUPS, 5, seed)]

    assert draws(0) == draws(0) != draws(1)


def test_grouped_bootstrap_auc_resamples_whole_groups_exact_case():
    # Hand-checked: all reals in one group, all fakes in the other. A resample has both classes
    # only when it draws both groups, and then it holds each group exactly once, i.e. the
    # original rows: every usable resample's AUC is the point AUC 14/16, so the CI collapses onto
    # it. Resampling rows instead would almost never miss a class and would give a wide CI.
    groups = np.where(Y == 0, "reals", "fakes")
    ci = model.grouped_bootstrap_auc(Y, S, groups, n_boot=1000, seed=0)
    assert tuple(ci) == pytest.approx((14 / 16, 14 / 16))
    # About half the draws of two groups out of two hold both; the rest are skipped and counted.
    assert 400 < ci.n_used < 600


def test_grouped_bootstrap_indices_are_unions_of_whole_groups():
    groups = np.array(["a", "a", "a", "b", "c", "c"])
    sizes = {"a": 3, "b": 1, "c": 2}
    for idx in model.grouped_bootstrap_indices(groups, n_boot=50, seed=0):
        drawn = pd.Series(groups[idx]).value_counts()
        for g, count in drawn.items():
            assert count % sizes[g] == 0  # a group's rows always come together
        counts = np.bincount(idx, minlength=len(groups))
        for g in sizes:  # ... and every row of a drawn group appears equally often
            assert len(set(counts[groups == g])) == 1
        assert sum(drawn[g] // sizes[g] for g in drawn.index) == len(sizes)  # 3 group draws


def test_grouped_bootstrap_auc_uses_grouped_indices(monkeypatch):
    calls = []
    original = model.grouped_bootstrap_indices

    def spy(groups, n_boot, seed):
        calls.append((tuple(groups), n_boot, seed))
        return original(groups, n_boot, seed)

    monkeypatch.setattr(model, "grouped_bootstrap_indices", spy)
    model.grouped_bootstrap_auc(Y, S, PAIR_GROUPS, n_boot=20, seed=3)
    model.grouped_bootstrap_auc_diff(Y, S, S[::-1], PAIR_GROUPS, n_boot=20, seed=3)
    assert calls == [(tuple(PAIR_GROUPS), 20, 3)] * 2


def test_grouped_bootstrap_auc_diff_paired():
    same = model.grouped_bootstrap_auc_diff(Y, S, S, PAIR_GROUPS, n_boot=100, seed=0)
    assert tuple(same) == (0.0, 0.0, 0.0)
    assert same.n_used == 100
    # A perfect scorer against S: the point difference is 1 - 14/16 and the paired CI holds it.
    perfect = Y.astype(float)
    delta, lo, hi = model.grouped_bootstrap_auc_diff(Y, perfect, S, PAIR_GROUPS, n_boot=200)
    assert delta == pytest.approx(2 / 16)
    assert 0.0 <= lo <= delta <= hi
    # Paired: each resample scores both models on the same rows, so the CI of A - B is the
    # negative of the CI of B - A.
    rev = model.grouped_bootstrap_auc_diff(Y, S, perfect, PAIR_GROUPS, n_boot=200)
    assert tuple(rev) == pytest.approx((-delta, -hi, -lo))


def test_grouped_bootstrap_without_usable_resample_is_nan():
    one_class = np.zeros(4, int)
    ci = model.grouped_bootstrap_auc(one_class, S[:4], ["a", "b", "c", "d"], n_boot=10)
    assert ci.n_used == 0 and np.isnan(ci[0]) and np.isnan(ci[1])
    diff = model.grouped_bootstrap_auc_diff(one_class, S[:4], S[:4], ["a", "b", "c", "d"], 10)
    assert diff.n_used == 0 and all(np.isnan(v) for v in diff)


def test_metrics_undefined_cases_are_nan_not_warnings():
    # Nothing predicted fake: precision undefined. One class only: AUC undefined.
    none = model.metrics(Y, S, 2.0)
    assert np.isnan(none["precision"]) and none["recall"] == 0.0 and none["tp"] == 0
    one = model.metrics(np.zeros(4), S[:4], 0.5)
    assert np.isnan(one["auc"]) and np.isnan(one["recall"])
    assert one["balanced_acc"] == pytest.approx(0.5)  # specificity only: 2 of 4 below 0.5
    with pytest.raises(ValueError):
        model.youden_threshold(np.zeros(4), S[:4])


# --- grouped evaluation on synthetic feature tables -------------------------------------------


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return synth_ffpp_table()


def test_grouped_cv_reports_mean_std_on_synthetic(table):
    train = table[table["split"] == "train"]
    abc = model.grouped_cv(train, model.feature_columns(train, "ABC"), "logreg")
    assert set(abc) == {"auc_mean", "auc_std", "n_folds_used"}
    assert abc["n_folds_used"] == 5
    assert abc["auc_mean"] > 0.9 and 0.0 <= abc["auc_std"] < 0.2
    # Same folds, same numbers: the CV is reproducible for a seed.
    assert model.grouped_cv(train, model.feature_columns(train, "ABC"), "logreg") == abc


def test_grouped_cv_skips_one_class_test_folds(table, monkeypatch):
    train = table[table["split"] == "train"].reset_index(drop=True)
    fakes = np.flatnonzero(train["label"] == 1)
    one_class = (np.setdiff1d(np.arange(len(train)), fakes[:3]), fakes[:3])  # test: fakes only

    original = model.grouped_folds_df

    def folds(df, n_splits=5, seed=0):  # the first fold's test part replaced by fakes only
        return [one_class, *original(df, n_splits, seed)[1:]]

    monkeypatch.setattr(model, "grouped_folds_df", folds)
    out = model.grouped_cv(train, model.feature_columns(train, "A"), "logreg")
    assert out["n_folds_used"] == 4


def test_pipeline_fit_does_not_see_test_rows(table, monkeypatch):
    train, val, test = _split(table)
    seen: list[tuple[pd.Index, np.ndarray]] = []
    original_fit = RealFaceReference.fit

    def spy(self, X, y):
        seen.append((X.index, np.asarray(y)))
        return original_fit(self, X, y)

    monkeypatch.setattr(RealFaceReference, "fit", spy)
    cols = model.feature_columns(table, "ABC")
    out = model.fit_and_test(train, val, test, cols, "logreg")

    assert len(seen) == 1  # fitted once, on the training rows only
    index, y = seen[0]
    assert set(index) == set(train.index)
    assert not set(index) & (set(val.index) | set(test.index))
    reals = set(index[y == 0])
    assert reals == set(train.index[train["label"] == 0])  # the reference's rows: train reals

    # The imputer's medians are the training medians, not those of all rows.
    fitted = out["model"]
    medians = fitted["impute"].named_transformers_["impute"].statistics_[: len(cols)]
    np.testing.assert_allclose(medians, train[cols].median().to_numpy())
    assert not np.allclose(medians, table[cols].median().to_numpy())


def test_fit_and_test_results_and_per_kind_auc(table):
    train, val, test = _split(table)
    out = model.fit_and_test(train, val, test, model.feature_columns(table, "ABC"), "logreg")
    assert set(out["per_kind_auc"]) == {"Deepfakes", "FaceSwap", "NeuralTextures"}
    assert out["metrics"]["n_real"] == 3 and out["metrics"]["n_fake"] == 9
    assert out["metrics"]["auc"] > 0.9
    # Per-kind AUC: that kind's fakes against ALL test reals.
    scores = pd.Series(out["test_scores"], index=test.index)
    reals = test["label"] == 0
    for kind, auc in out["per_kind_auc"].items():
        subset = reals | (test["kind"] == kind)
        expected = model.metrics(test.loc[subset, "label"], scores[subset], 0.5)["auc"]
        assert auc == pytest.approx(expected)
    # The threshold comes from validation scores.
    val_scores = out["model"].predict_proba(val[model.feature_columns(table, "ABC")])[:, 1]
    assert out["threshold"] == pytest.approx(model.youden_threshold(val["label"], val_scores))


def test_fit_and_test_rejects_leakage(table):
    train, val, test = _split(table)
    cols = model.feature_columns(table, "A")
    leaky_test = pd.concat([test, train.iloc[:2]])  # a training group also in test
    with pytest.raises(ValueError, match="more than one split"):
        model.fit_and_test(train, val, leaky_test, cols, "logreg")
    # The rows' role counts, not their split column: the same frame as train and test leaks.
    with pytest.raises(ValueError, match="more than one split"):
        model.fit_and_test(train, val, train, cols, "logreg")


def test_permutation_importances_sorted_and_grouped(table):
    train, val, test = _split(table)
    cols = model.feature_columns(table, "ABC")
    fitted = model.fit_and_test(train, val, test, cols, "logreg")["model"]
    imp = model.permutation_importances(fitted, test, cols, n_repeats=3)
    assert list(imp.columns) == ["feature", "auc_drop_mean", "auc_drop_std"]
    assert sorted(imp["feature"]) == sorted(cols)
    assert imp["auc_drop_mean"].is_monotonic_decreasing
    again = model.permutation_importances(fitted, test, cols, n_repeats=3)
    pd.testing.assert_frame_equal(imp, again)  # seeded

    by_group = model.group_permutation_importances(fitted, test, cols, n_repeats=3)
    assert sorted(by_group["feature"]) == ["A", "B", "C"]
    assert by_group["auc_drop_mean"].is_monotonic_decreasing
