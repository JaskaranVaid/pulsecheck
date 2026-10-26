"""Leak-free real-face reference: how far is a video's blood-flow pattern from real faces?

Group C gives each video a map of pulse amplitude shares (``c_amp_<region>``, 7 numbers) and of
phase lags behind the forehead (``c_lag_<region>``, 6 numbers). Single numbers are hard to judge
("is a chin lag of 30 ms odd?"), but a whole map can be compared with the maps of known real
faces. ``RealFaceReference`` learns, per map (block), the mean and covariance of real faces and
scores every video by its Mahalanobis distance to that mean:

    d(x) = sqrt((x - mean)^T  P  (x - mean)),   P = inverse covariance ("precision")

Mahalanobis rather than Euclidean distance because the map entries are correlated (amplitude
shares sum to 1; the two cheeks move together) and have different natural spreads; the
covariance accounts for both. The covariance is estimated with Ledoit-Wolf shrinkage: with only a
few dozen real videos per training fold the plain sample covariance of 6-7 columns is noisy and
can be near-singular, and shrinking it towards a scaled identity keeps its inverse stable.

**No leakage:** the statistics come from the *training reals only* (rows with ``y == 0`` passed
to ``fit``). Used as the first step of a scikit-learn ``Pipeline``, cross-validation refits it on
each training fold, so a held-out video never influences the reference it is scored against.
That is also why the two distance columns are added here, at model time, and not stored in the
per-video feature table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import LedoitWolf
from sklearn.utils.validation import check_is_fitted

from pulsecheck.regions import SKIN_REGIONS

MIN_REFERENCE_ROWS = 5  # fewer complete real rows than this: no reference (NaN statistics)
DISTANCE_SUFFIX = "mahal"  # output column = prefix + suffix, e.g. "c_amp_mahal"


def _require_frame(X) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"RealFaceReference needs a pandas DataFrame, got {type(X).__name__}")
    return X


def _block_columns(columns: pd.Index, prefix: str) -> list[str]:
    """The columns ``prefix + <skin region>``, in table order.

    Matching region names (not just the prefix) keeps per-video summaries that share the prefix,
    such as ``c_lag_stability``, and earlier outputs such as ``c_lag_mahal``, out of the map.
    """
    return [c for c in columns if c.startswith(prefix) and c[len(prefix) :] in SKIN_REGIONS]


def _fit_block(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(mean, precision)`` of the complete rows of one block; NaN when there are too few.

    ``rows`` is ``(n, p)``; rows containing NaN are dropped first.
    """
    p = rows.shape[1]
    complete = rows[np.isfinite(rows).all(axis=1)]
    if p == 0 or len(complete) < MIN_REFERENCE_ROWS:
        return np.full(p, np.nan), np.full((p, p), np.nan)
    lw = LedoitWolf().fit(complete)
    return lw.location_, lw.precision_


def _mahalanobis(rows: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Distance of each row of ``rows`` ``(n, p)`` to ``mean``; NaN for rows with NaN, and for
    every row when the statistics are NaN or the block is empty."""
    if mean.size == 0 or not np.isfinite(mean).all():
        return np.full(len(rows), np.nan)
    diff = rows - mean  # (n, p); NaN rows stay NaN through the arithmetic below
    squared = np.einsum("ij,jk,ik->i", diff, precision, diff)  # diff_i^T P diff_i per row
    return np.sqrt(np.maximum(squared, 0.0))  # clip float round-off below 0


class RealFaceReference(TransformerMixin, BaseEstimator):
    """Append ``<amp_prefix>mahal`` and ``<lag_prefix>mahal``: Mahalanobis distances of each
    row's amplitude map and lag map to the real-face reference fitted on training reals.

    Parameters select the two blocks by prefix; a block holds the columns
    ``prefix + <skin region>`` present in the table.

    (scikit-learn requires mixins to come before ``BaseEstimator`` in the bases, otherwise the
    transformer tags are lost; the class is still a ``BaseEstimator`` and ``TransformerMixin``.)

    Fitted attributes: ``feature_names_in_``, ``n_features_in_``, ``amp_columns_``,
    ``lag_columns_``, ``amp_mean_``, ``amp_precision_``, ``lag_mean_``, ``lag_precision_``.
    """

    def __init__(self, amp_prefix: str = "c_amp_", lag_prefix: str = "c_lag_"):
        self.amp_prefix = amp_prefix
        self.lag_prefix = lag_prefix

    def _output_columns(self) -> list[str]:
        return [self.amp_prefix + DISTANCE_SUFFIX, self.lag_prefix + DISTANCE_SUFFIX]

    def fit(self, X: pd.DataFrame, y) -> RealFaceReference:
        """Fit one Ledoit-Wolf mean/covariance per block on the rows with ``y == 0`` (reals).

        Within a block, real rows with any NaN are dropped; with fewer than
        ``MIN_REFERENCE_ROWS`` left, that block's statistics are NaN (its distances will be NaN).
        Raises ``TypeError`` if ``X`` is not a DataFrame and ``ValueError`` if ``y`` does not
        have one label per row or ``X`` already holds an output column.
        """
        X = _require_frame(X)
        y = np.asarray(y)
        if y.shape != (len(X),):
            raise ValueError(f"y must have one label per row: {y.shape} vs {len(X)} rows")
        clash = [c for c in self._output_columns() if c in X.columns]
        if clash:
            raise ValueError(f"input already has output column(s) {clash}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.amp_columns_ = _block_columns(X.columns, self.amp_prefix)
        self.lag_columns_ = _block_columns(X.columns, self.lag_prefix)

        reals = X.loc[y == 0]  # the fakes never touch the reference
        self.amp_mean_, self.amp_precision_ = _fit_block(reals[self.amp_columns_].to_numpy(float))
        self.lag_mean_, self.lag_precision_ = _fit_block(reals[self.lag_columns_].to_numpy(float))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """``X`` unchanged, with the two distance columns appended (input order kept).

        Raises ``TypeError`` if ``X`` is not a DataFrame and ``ValueError`` if its columns differ
        from (or are ordered differently than) those seen in ``fit``.
        """
        check_is_fitted(self)
        X = _require_frame(X)
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("X must have the same columns, in the same order, as in fit")
        out = X.copy()
        amp_out, lag_out = self._output_columns()
        out[amp_out] = _mahalanobis(
            X[self.amp_columns_].to_numpy(float), self.amp_mean_, self.amp_precision_
        )
        out[lag_out] = _mahalanobis(
            X[self.lag_columns_].to_numpy(float), self.lag_mean_, self.lag_precision_
        )
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Input column names followed by the two distance columns."""
        check_is_fitted(self)
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError("input_features do not match the columns seen in fit")
        return np.asarray([*self.feature_names_in_, *self._output_columns()], dtype=object)
