"""Video manifests: one row per dataset video (label, leakage group, split, compression).

A manifest is a CSV with a header in ``VideoItem`` field order. Labels are 0 = real, 1 = fake.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath

SPLITS = ("train", "val", "test")  # allowed values of VideoItem.split
LABELS = (0, 1)  # 0 = real, 1 = fake


@dataclass(frozen=True)
class VideoItem:
    dataset: str  # "ffpp" or "celebdf"
    path: Path  # absolute path of the video file
    rel_path: str  # POSIX path relative to the dataset root
    label: int  # 0 real, 1 fake
    group: str  # leakage group: videos sharing it must never straddle splits or CV folds
    kind: str  # "original", a manipulation method, "Celeb-real", "Celeb-synthesis", "YouTube-real"
    compression: str  # "c23", "c40", or "" when the dataset has no such variants
    split: str  # "train", "val" or "test"

    @property
    def root(self) -> Path:
        """The dataset root: ``path`` with the ``rel_path`` components removed.

        Raises ``ValueError`` if ``path`` does not end with ``rel_path``.
        """
        parts = PurePosixPath(self.rel_path).parts
        if not parts or self.path.parts[-len(parts) :] != parts:
            raise ValueError(f"{self.path}: does not end with rel_path {self.rel_path!r}")
        return Path(*self.path.parts[: -len(parts)])


COLUMNS = tuple(f.name for f in fields(VideoItem))


def write_manifest(items: list[VideoItem], path: Path) -> None:
    """Write ``items`` as CSV (header in field order); parent directories are created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for item in items:
            writer.writerow(
                [
                    item.path.as_posix() if name == "path" else getattr(item, name)
                    for name in COLUMNS
                ]
            )


def read_manifest(path: Path) -> list[VideoItem]:
    """Read a manifest written by ``write_manifest``.

    Raises ``ValueError`` (naming the file and line) on a missing column, a label other than
    0/1, or a split other than train/val/test.
    """
    path = Path(path)
    items: list[VideoItem] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s) {missing}")
        for rec in reader:
            where = f"{path}:{reader.line_num}"
            try:
                label = int(rec["label"])
            except ValueError:
                label = -1
            if label not in LABELS:
                raise ValueError(f"{where}: bad label {rec['label']!r} (expected 0 or 1)")
            if rec["split"] not in SPLITS:
                raise ValueError(f"{where}: bad split {rec['split']!r} (expected one of {SPLITS})")
            items.append(
                VideoItem(
                    dataset=rec["dataset"],
                    path=Path(rec["path"]),
                    rel_path=rec["rel_path"],
                    label=label,
                    group=rec["group"],
                    kind=rec["kind"],
                    compression=rec["compression"],
                    split=rec["split"],
                )
            )
    return items
