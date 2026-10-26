"""FaceForensics++: originals and four manipulations, with the official pair-based splits.

Originals are renamed to three-digit ids (``000``-``999``); a manipulated video is named
``<target>_<source>.mp4``. The official splits (``train.json``, ``val.json``, ``test.json``) list
pairs of ids; both ids of a pair, their originals, and every fake made from them (either
direction, every method) form one leakage group ``ffpp:{min}_{max}``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pulsecheck.datasets.manifest import SPLITS, VideoItem

FFPP_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
COMPRESSIONS = ("c23", "c40")

# Path templates under the dataset root. The ``videos/`` sub-folder is an assumption taken from
# the FF++ download script; confirm on the first real download and change only these two lines.
ORIGINAL_TEMPLATE = "original_sequences/youtube/{c}/videos/{id}.mp4"
FAKE_TEMPLATE = "manipulated_sequences/{m}/{c}/videos/{name}.mp4"

_ID_RE = re.compile(r"[0-9]{3}")

Splits = dict[str, list[tuple[str, str]]]


def load_splits(splits_dir: Path) -> Splits:
    """Read ``train.json``, ``val.json`` and ``test.json`` from ``splits_dir``.

    Each file must be a JSON array of 2-element arrays of 3-digit id strings. Raises
    ``ValueError`` (naming the file) on any other shape, and if an id occurs more than once
    across all three files (in two splits, or twice in one).
    """
    splits_dir = Path(splits_dir)
    splits: Splits = {}
    seen: dict[str, str] = {}
    for split in SPLITS:
        file = splits_dir / f"{split}.json"
        data = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{file}: expected a JSON array of id pairs")  # noqa: TRY004
        pairs: list[tuple[str, str]] = []
        for k, pair in enumerate(data):
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(v, str) and _ID_RE.fullmatch(v) for v in pair)
            ):
                raise ValueError(
                    f"{file}: entry {k} is {pair!r}, expected a pair of 3-digit id strings"
                )
            for vid in pair:
                if vid in seen:
                    where = "also in " + seen[vid] if seen[vid] != split else "twice in it"
                    raise ValueError(f"{file}: id {vid} appears in {split} and {where}")
                seen[vid] = split
            pairs.append((pair[0], pair[1]))
        splits[split] = pairs
    return splits


def list_ffpp(
    root: Path,
    compression: str,
    splits: Splits,
    methods: tuple[str, ...] = FFPP_METHODS,
    max_pairs: dict[str, int] | None = None,
) -> tuple[list[VideoItem], list[Path]]:
    """Every original and fake video of the split pairs, as ``(items, missing)``.

    For each pair ``(a, b)``: the originals ``a`` and ``b`` (label 0, kind ``original``) and, for
    each method, the fakes ``a_b`` and ``b_a`` (label 1, kind = method). Files that do not exist
    are not returned as items but listed (absolute expected paths) in ``missing``.
    ``max_pairs`` maps a split name to the number of its pairs to keep (the first N, in JSON
    order); splits not named are kept whole.

    Raises ``ValueError`` for an unknown compression, method, or ``max_pairs`` key.
    """
    root = Path(root).absolute()
    if compression not in COMPRESSIONS:
        raise ValueError(f"unknown compression {compression!r}; choose from {COMPRESSIONS}")
    unknown = [m for m in methods if m not in FFPP_METHODS]
    if unknown:
        raise ValueError(f"unknown method(s) {unknown}; choose from {FFPP_METHODS}")
    limits = max_pairs or {}
    bad = [k for k in limits if k not in SPLITS]
    if bad:
        raise ValueError(f"max_pairs has unknown split(s) {bad}; choose from {SPLITS}")

    items: list[VideoItem] = []
    missing: list[Path] = []

    def add(rel: str, label: int, group: str, kind: str, split: str) -> None:
        path = root / rel
        if not path.is_file():
            missing.append(path)
            return
        items.append(VideoItem("ffpp", path, rel, label, group, kind, compression, split))

    for split in SPLITS:
        pairs = splits.get(split, [])
        if split in limits:
            pairs = pairs[: limits[split]]
        for a, b in pairs:
            group = f"ffpp:{min(a, b)}_{max(a, b)}"
            for vid in (a, b):
                rel = ORIGINAL_TEMPLATE.format(c=compression, id=vid)
                add(rel, 0, group, "original", split)
            for method in methods:
                for name in (f"{a}_{b}", f"{b}_{a}"):
                    rel = FAKE_TEMPLATE.format(m=method, c=compression, name=name)
                    add(rel, 1, group, method, split)
    return items, missing
