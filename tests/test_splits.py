"""Grouped splits: the leakage guard (``check_no_leakage`` / ``check_no_leakage_df``), grouped CV
folds and the split report. Items are built directly as ``VideoItem`` rows; no media is read.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pulsecheck.datasets.manifest import VideoItem
from pulsecheck.splits import check_no_leakage, check_no_leakage_df, grouped_folds, split_report


def _item(index: int, group: str, label: int = 0, kind: str = "original", split: str = "train"):
    name = f"{index:03d}.mp4"
    return VideoItem(
        dataset="ffpp",
        path=Path("/data/ffpp") / name,
        rel_path=name,
        label=label,
        group=group,
        kind=kind,
        compression="c23",
        split=split,
    )


def _leaky_items() -> list[VideoItem]:
    return [
        _item(0, "ffpp:1_2", label=0, split="train"),
        _item(1, "ffpp:1_2", label=1, kind="Deepfakes", split="test"),
        _item(2, "ffpp:3_4", label=0, split="train"),
    ]


def _cv_items() -> list[VideoItem]:
    """40 items in 10 single-label groups: 8 real (5 groups) and 32 fake (5 groups), i.e. 1:4."""
    real_sizes = [2, 2, 2, 1, 1]
    fake_sizes = [7, 7, 6, 6, 6]
    items: list[VideoItem] = []
    index = 0
    for g, size in enumerate(real_sizes):
        for _ in range(size):
            items.append(_item(index, f"real_{g}", label=0, kind="original"))
            index += 1
    for g, size in enumerate(fake_sizes):
        for _ in range(size):
            items.append(_item(index, f"fake_{g}", label=1, kind="Deepfakes"))
            index += 1
    assert len(items) == 40
    return items


def test_leakage_detected_and_named():
    with pytest.raises(ValueError, match="ffpp:1_2"):
        check_no_leakage(_leaky_items())


def test_no_leakage_passes_for_disjoint_groups():
    items = [
        _item(0, "g_a", split="train"),
        _item(1, "g_a", split="train"),
        _item(2, "g_b", split="val"),
        _item(3, "g_c", split="test"),
    ]
    check_no_leakage(items)  # must not raise


def test_leakage_message_lists_at_most_ten_groups():
    items = []
    for g in range(12):
        items.append(_item(2 * g, f"g{g:02d}", split="train"))
        items.append(_item(2 * g + 1, f"g{g:02d}", split="test"))
    with pytest.raises(ValueError) as info:
        check_no_leakage(items)
    message = str(info.value)
    assert "12" in message
    assert "g09" in message
    assert "g10" not in message
    assert "g11" not in message


def test_check_no_leakage_df_detects_and_names_group():
    df = pd.DataFrame(
        {
            "group": [i.group for i in _leaky_items()],
            "split": [i.split for i in _leaky_items()],
        }
    )
    with pytest.raises(ValueError, match="ffpp:1_2"):
        check_no_leakage_df(df)


def test_check_no_leakage_df_passes_for_disjoint_groups():
    df = pd.DataFrame(
        {"group": ["g_a", "g_a", "g_b", "g_c"], "split": ["train", "train", "val", "test"]}
    )
    check_no_leakage_df(df)  # must not raise


def test_check_no_leakage_df_message_matches_items_message():
    items = _leaky_items()
    df = pd.DataFrame({"group": [i.group for i in items], "split": [i.split for i in items]})
    with pytest.raises(ValueError) as from_items:
        check_no_leakage(items)
    with pytest.raises(ValueError) as from_df:
        check_no_leakage_df(df)
    assert str(from_df.value) == str(from_items.value)


def test_grouped_folds_never_share_groups_and_cover_all():
    items = _cv_items()
    folds = grouped_folds(items)  # default n_splits=5
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        assert isinstance(train_idx, np.ndarray) and np.issubdtype(train_idx.dtype, np.integer)
        assert isinstance(test_idx, np.ndarray) and np.issubdtype(test_idx.dtype, np.integer)
        train_groups = {items[i].group for i in train_idx}
        test_groups = {items[i].group for i in test_idx}
        assert not (train_groups & test_groups)
    test_union = np.concatenate([test_idx for _, test_idx in folds])
    assert sorted(test_union.tolist()) == list(range(len(items)))


def test_grouped_folds_reproducible_for_a_seed():
    items = _cv_items()
    first = grouped_folds(items, seed=7)
    second = grouped_folds(items, seed=7)
    for (tr_a, te_a), (tr_b, te_b) in zip(first, second, strict=True):
        assert np.array_equal(tr_a, tr_b)
        assert np.array_equal(te_a, te_b)


def test_split_report_counts():
    items = [
        _item(0, "a", label=0, kind="original", split="train"),
        _item(1, "a", label=0, kind="original", split="train"),
        _item(2, "b", label=1, kind="Deepfakes", split="train"),
        _item(3, "c", label=1, kind="Face2Face", split="test"),
        _item(4, "c", label=1, kind="Face2Face", split="test"),
        _item(5, "d", label=0, kind="original", split="val"),
    ]
    report = split_report(items)
    rows = {}
    total = None
    for line in report.splitlines():
        parts = line.split()
        if parts and parts[0] == "total":
            total = int(parts[-1])
        elif len(parts) == 4 and parts[0] in ("train", "val", "test"):
            rows[(parts[0], parts[1], parts[2])] = int(parts[3])
    assert rows == {
        ("train", "real", "original"): 2,
        ("train", "fake", "Deepfakes"): 1,
        ("val", "real", "original"): 1,
        ("test", "fake", "Face2Face"): 2,
    }
    assert total == 6
