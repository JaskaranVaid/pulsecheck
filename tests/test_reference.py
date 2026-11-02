"""The leak-free real-face reference (``RealFaceReference``) and the synthetic showcase."""

import functools
import time
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pulsecheck import features
from pulsecheck.reference import RealFaceReference
from pulsecheck.regions import SKIN_REGIONS
from synth import synth_region_traces

AMP = [f"c_amp_{r}" for r in SKIN_REGIONS]
LAG = [f"c_lag_{r}" for r in SKIN_REGIONS if r != features.REF_REGION]
OUT = ["c_amp_mahal", "c_lag_mahal"]


def _table(n: int, rng: np.random.Generator, shift: float = 0.0) -> pd.DataFrame:
    """Random rows with an unrelated feature, both blocks, and ``c_lag_stability`` (which shares
    the lag prefix but is not a region lag, so it must stay out of the lag block)."""
    cols = ["a_snr_face", *AMP, *LAG, "c_lag_stability"]
    return pd.DataFrame(rng.normal(shift, 1.0, (n, len(cols))), columns=cols)


# --- the transformer on hand-made tables -------------------------------------------------------


def test_reference_fits_on_reals_only():
    rng = np.random.default_rng(0)
    reals = _table(40, rng)
    fakes = _table(20, rng, shift=1e3)  # wild values: any leak into the fit would show
    X = pd.concat([reals, fakes], ignore_index=True)
    y = np.r_[np.zeros(40), np.ones(20)]

    mixed = RealFaceReference().fit(X, y)
    alone = RealFaceReference().fit(reals, np.zeros(40))
    assert mixed.amp_columns_ == AMP and mixed.lag_columns_ == LAG
    np.testing.assert_allclose(mixed.amp_mean_, alone.amp_mean_)
    np.testing.assert_allclose(mixed.lag_mean_, alone.lag_mean_)
    np.testing.assert_allclose(mixed.amp_precision_, alone.amp_precision_)
    np.testing.assert_allclose(mixed.lag_precision_, alone.lag_precision_)
    out = mixed.transform(X)
    assert out[OUT].iloc[40:].min().min() > 100 * out[OUT].iloc[:40].max().max()


def test_reference_output_columns_order_and_set_output():
    rng = np.random.default_rng(1)
    X = _table(30, rng)
    y = np.zeros(30)
    ref = RealFaceReference().fit(X, y)
    out = ref.transform(X)
    assert list(out.columns) == [*X.columns, *OUT]
    pd.testing.assert_frame_equal(out[X.columns], X)  # inputs passed through untouched
    assert list(ref.get_feature_names_out()) == [*X.columns, *OUT]
    assert (out[OUT] >= 0).all().all()

    # Mahalanobis distance, written out: sqrt((x - mean)^T P (x - mean)).
    diff = X[AMP].to_numpy()[0] - ref.amp_mean_
    assert out["c_amp_mahal"].iloc[0] == pytest.approx(np.sqrt(diff @ ref.amp_precision_ @ diff))

    # Works as the first step of a pandas-output pipeline.
    pipe = Pipeline([("ref", RealFaceReference()), ("scale", StandardScaler())])
    piped = pipe.set_output(transform="pandas").fit_transform(X, y)
    assert list(piped.columns) == [*X.columns, *OUT]

    with pytest.raises(TypeError):
        RealFaceReference().fit(X.to_numpy(), y)
    with pytest.raises(ValueError):
        ref.transform(X[X.columns[::-1]])  # same columns, different order


def test_reference_nan_rows_and_too_few_rows():
    rng = np.random.default_rng(2)
    X = _table(30, rng)
    X.loc[3, "c_amp_nose"] = np.nan  # this row is dropped from the amplitude fit only
    X.loc[4, "c_lag_chin"] = np.nan
    y = np.zeros(30)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = RealFaceReference().fit(X, y).transform(X)
    assert np.isnan(out.loc[3, "c_amp_mahal"]) and np.isfinite(out.loc[3, "c_lag_mahal"])
    assert np.isnan(out.loc[4, "c_lag_mahal"]) and np.isfinite(out.loc[4, "c_amp_mahal"])
    assert out[OUT].notna().sum().tolist() == [29, 29]

    # Fewer than 5 complete real rows: NaN statistics, all-NaN output columns, no exception.
    y_few = np.r_[np.zeros(4), np.ones(26)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ref = RealFaceReference().fit(X, y_few)
        few = ref.transform(X)
    assert np.isnan(ref.amp_mean_).all() and np.isnan(ref.lag_precision_).all()
    assert few[OUT].isna().all().all()


# --- the transformer on synthetic videos ------------------------------------------------------


@functools.cache
def _row(kind: str, seed: int) -> dict[str, float]:
    """Groups A, B and C of one synthetic video (cached across the tests of this module)."""
    return features.video_features(synth_region_traces(kind, seed=seed), groups="ABC")


def _frame(kind: str, seeds: range) -> pd.DataFrame:
    return pd.DataFrame([_row(kind, s) for s in seeds])


REAL_SEEDS = range(30)  # training reals
HELD_OUT_SEEDS = range(200, 230)  # reals never seen by the fit
FAKE_SEEDS = range(100, 130)  # inherited-pulse fakes (seeds disjoint from the reals: unpaired)


def test_reference_distance_separates_inherited():
    reals = _frame("real", REAL_SEEDS)
    ref = RealFaceReference().fit(reals, np.zeros(len(reals)))
    held_out = ref.transform(_frame("real", HELD_OUT_SEEDS))
    inherited = ref.transform(_frame("inherited", FAKE_SEEDS))
    y = np.r_[np.zeros(len(held_out)), np.ones(len(inherited))]
    for col in OUT:
        fake, real = inherited[col].median(), held_out[col].median()
        auc = roc_auc_score(y, np.r_[held_out[col], inherited[col]])
        print(f"{col}: median inherited {fake:.2f}, held-out real {real:.2f}, AUC {auc:.3f}")
        # Each distance on its own ranks inherited fakes above unseen reals. The median ratio is
        # modest because a held-out real already sits about sqrt(p) (~2.5 for 6-7 columns) from
        # the mean in Mahalanobis units: that floor is the typical spread of real faces, not error.
        # Measured (30 train reals, 30 held-out, 30 inherited): amp 5.13 vs 2.71 (AUC 0.96),
        # lag 4.34 vs 2.34 (AUC 0.96).
        assert auc > 0.9, (col, auc)
        assert fake > 1.5 * real, (col, fake, real)


# Keeps only the group-A columns and the two reference distances (drops the raw C maps).
A_AND_DISTANCES = r"^a_|_mahal$"


def _pipeline(with_reference: bool, keep: str | None = None) -> Pipeline:
    """[reference] -> [keep] -> impute -> scale -> logistic regression, pandas output.

    ``keep``: a column-name regex applied right after the reference (``remainder="drop"``), so
    the classifier sees only the matching columns, e.g. the distances but not the maps they were
    computed from.
    """
    steps = [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
    ]
    if keep is not None:
        selector = make_column_selector(pattern=keep)
        steps.insert(0, ("keep", ColumnTransformer(
            [("keep", "passthrough", selector)], remainder="drop", verbose_feature_names_out=False
        )))
    if with_reference:
        steps.insert(0, ("ref", RealFaceReference()))
    return Pipeline(steps).set_output(transform="pandas")


def _cv_auc(
    X: pd.DataFrame, y: np.ndarray, with_reference: bool, keep: str | None = None
) -> float:
    """Mean ROC-AUC over 5 stratified folds; every fitted statistic is fitted on training folds."""
    pipe = _pipeline(with_reference, keep)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    return float(cross_val_score(pipe, X, y, cv=folds, scoring="roc_auc").mean())


def test_showcase_auc_inherited_vs_real():
    # The headline claim on synthetic data: a pulse copied uniformly into a fake face fools the
    # "is there a pulse?" baseline (group A), but not the physiological-consistency features.
    #
    # Measured mean fold AUC on these 60 videos (CV seed 0; ruling P2-R13):
    #   A 0.483 | B only 0.906 | AB 0.861 | ABC 1.000 | ABC without reference 1.000
    #   A + only c_amp_mahal/c_lag_mahal (raw maps dropped) 1.000 | C only (with reference) 1.000
    # A-only over CV seeds 0-4: 0.467-0.544.
    #
    # What each number shows:
    # * "ABC > 0.9" alone would not prove much: group B by itself already reaches about 0.9
    #   (0.906-0.956 over CV seeds 0-4), so the two isolating checks below are the real test of
    #   the contribution. (a) The two reference distances alone, with every raw lag/amplitude
    #   column dropped, carry the separation: the real-face reference works on its own.
    #   (b) Group C alone carries it without help from A or B.
    # * Why B separates the synthetic kinds, honestly: NOT because the inherited fake's regions
    #   fail to share a heartbeat (they share it perfectly). Inherited regions are, if anything,
    #   MORE alike (b_corr_min about 0.59 vs 0.50 for real), mainly because the real synth gives
    #   some regions a weaker pulse (chin 0.7, nose 0.8), which lowers their SNR and so their
    #   correlation with the others; the real lags add a little. B is a test of "is there one
    #   shared pulse?" (it rejects the "noise" kind); its separation of real from inherited here
    #   is a by-product of the synth's amplitude heterogeneity and may not carry over to real data.
    # * On synthetic data a linear model can also use the raw lags/amplitudes directly (ABC
    #   without reference 1.0); the reference's distinct value (absorbing an offset that
    #   training reals share, such as rolling-shutter readout time) can only show on real data.
    start = time.perf_counter()
    table = pd.concat([_frame("real", REAL_SEEDS), _frame("inherited", FAKE_SEEDS)],
                      ignore_index=True)
    y = np.r_[np.zeros(len(REAL_SEEDS)), np.ones(len(FAKE_SEEDS))]

    def cols(groups: str) -> list[str]:
        return [c for c in table.columns if c[0] in groups.lower() and c[1] == "_"]

    auc = {g: _cv_auc(table[cols(g)], y, with_reference="C" in g) for g in ("A", "AB", "ABC")}
    # The A+mahal model really sees only group A and the two distances.
    seen = _pipeline(True, A_AND_DISTANCES)[:2].fit_transform(table[cols("AC")], y).columns
    assert list(seen) == [*cols("A"), *OUT]
    auc["A+mahal"] = _cv_auc(table[cols("AC")], y, with_reference=True, keep=A_AND_DISTANCES)
    auc["C"] = _cv_auc(table[cols("C")], y, with_reference=True)
    print(f"showcase AUC: {auc}  ({time.perf_counter() - start:.1f} s)")
    assert auc["A"] < 0.7, auc
    assert auc["ABC"] > 0.9, auc
    assert auc["A+mahal"] > 0.9, auc  # (a) the reference distances alone
    assert auc["C"] > 0.9, auc  # (b) group C alone
