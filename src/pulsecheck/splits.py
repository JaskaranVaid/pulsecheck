"""Grouped splits and the leakage guard.

FaceForensics++ fakes are made from the real videos, so a leakage group (``VideoItem.group``)
must never straddle two splits or two CV folds. ``check_no_leakage`` / ``check_no_leakage_df``
enforce the split side, ``grouped_folds`` builds group-aware CV folds, and ``split_report``
tabulates the counts per split, label and kind.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from pulsecheck.datasets.manifest import SPLITS, VideoItem

MAX_LISTED_GROUPS = 10  # groups named in a leakage error
LABEL_NAMES = {0: "real", 1: "fake"}  # labels as shown in the split report


def _check_pairs(pairs: Iterable[tuple[str, str]]) -> None:
    """Raise ``ValueError`` if any group appears under more than one split.

    ``pairs`` holds ``(group, split)`` tuples. The error names up to ``MAX_LISTED_GROUPS``
    leaking groups, each with the splits it appears in.
    """
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for group, split in pairs:
        splits_by_group[group].add(split)
    leaking = sorted(g for g, found in splits_by_group.items() if len(found) > 1)
    if not leaking:
        return
    shown = ", ".join(f"{g!r} in {sorted(splits_by_group[g])}" for g in leaking[:MAX_LISTED_GROUPS])
    raise ValueError(
        f"{len(leaking)} group(s) appear in more than one split "
        f"(showing {min(len(leaking), MAX_LISTED_GROUPS)}): {shown}"
    )


def check_no_leakage(items: list[VideoItem]) -> None:
    """Raise ``ValueError`` if any leakage group appears in more than one split."""
    _check_pairs((item.group, item.split) for item in items)


def check_no_leakage_df(df: pd.DataFrame) -> None:
    """Same check as ``check_no_leakage``, for a table with ``group`` and ``split`` columns."""
    _check_pairs(zip(df["group"], df["split"], strict=True))


def _check_enough_groups(items: list[VideoItem], n_splits: int) -> None:
    """Raise ``ValueError`` unless there are at least ``n_splits`` groups, overall and per label.

    Checked before sklearn so the message counts groups; sklearn's own errors count members.
    """
    total = len({item.group for item in items})
    if total < n_splits:
        raise ValueError(f"only {total} groups; need at least n_splits={n_splits}")
    for label in sorted({item.label for item in items}):
        count = len({item.group for item in items if item.label == label})
        if count < n_splits:
            name = LABEL_NAMES.get(label, str(label))
            raise ValueError(f"only {count} {name} groups; need at least n_splits={n_splits}")


def grouped_folds(
    items: list[VideoItem], n_splits: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified-by-label, group-aware CV folds as ``(train_idx, test_idx)`` index arrays.

    Indices refer to positions in ``items``. No group appears in both the train and the test
    part of a fold, and each item is in exactly one test fold. Raises ``ValueError`` if there are
    fewer than ``n_splits`` distinct groups overall, or fewer than ``n_splits`` for any label.
    """
    labels = np.array([item.label for item in items])
    groups = np.array([item.group for item in items])
    _check_enough_groups(items, n_splits)
    features = np.zeros((len(items), 1))  # the splitter ignores X; it only needs its length
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(train, test) for train, test in cv.split(features, labels, groups)]


def split_report(items: list[VideoItem]) -> str:
    """Plain-text table of item counts per split, label and kind, with a total line."""
    counts = Counter((item.split, item.label, item.kind) for item in items)

    def sort_key(key: tuple[str, int, str]) -> tuple[int, str, int, str]:
        split, label, kind = key
        position = SPLITS.index(split) if split in SPLITS else len(SPLITS)
        return (position, split, label, kind)

    rows = [
        (split, LABEL_NAMES.get(label, str(label)), kind, str(count))
        for (split, label, kind), count in sorted(counts.items(), key=lambda kv: sort_key(kv[0]))
    ]
    header = ("split", "label", "kind", "count")
    total = ("total", "", "", str(len(items)))
    table = [header, *rows, total]
    widths = [max(len(row[c]) for row in table) for c in range(len(header))]

    def fmt(row: tuple[str, str, str, str]) -> str:
        left = "  ".join(cell.ljust(width) for cell, width in zip(row[:3], widths[:3], strict=True))
        return f"{left}  {row[3].rjust(widths[3])}".rstrip()

    return "\n".join(fmt(row) for row in table) + "\n"
