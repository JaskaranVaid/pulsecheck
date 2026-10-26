"""Celeb-DF v2: used as a test-only set, via the official ``List_of_testing_videos.txt``.

Layout: ``root/Celeb-real``, ``root/Celeb-synthesis``, ``root/YouTube-real`` and the list file,
whose lines are ``<label> <relative path>``. The label convention is not taken on trust: the parser
checks every line against its folder (1 with ``*-real``, 0 with ``Celeb-synthesis``) and raises
if a real file list says otherwise.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from pulsecheck.datasets.manifest import VideoItem

LIST_NAME = "List_of_testing_videos.txt"
KINDS = ("Celeb-real", "Celeb-synthesis", "YouTube-real")
# Label written in the list file for each folder: 1 = real, 0 = synthesis (self-checked).
_LIST_LABEL = {"Celeb-real": 1, "YouTube-real": 1, "Celeb-synthesis": 0}
_IDENTITY_RE = re.compile(r"^(id\d+)_")  # idA_0000 (real) or idA_idB_0000 (synthesis, target idA)


def list_celebdf_test(root: Path) -> tuple[list[VideoItem], list[Path]]:
    """Videos named in ``root/List_of_testing_videos.txt``, as ``(items, missing)``.

    Labels become 0 real / 1 fake. Group: the identity token ``idN`` (the target for synthesis
    files) for Celeb-real/Celeb-synthesis, ``yt:{stem}`` for YouTube-real. ``split`` is ``test``
    and ``compression`` is empty. Listed files that do not exist are returned (absolute paths)
    in ``missing``. Blank lines are ignored.

    Raises ``FileNotFoundError`` if the list is absent, and ``ValueError`` (``file:line``) for a
    malformed line, an unknown folder, a path that leaves the root, or a label that disagrees
    with the folder.
    """
    root = Path(root).absolute()
    list_file = root / LIST_NAME
    items: list[VideoItem] = []
    missing: list[Path] = []
    for n, raw in enumerate(list_file.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        where = f"{list_file}:{n}"
        fields = raw.strip().split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"{where}: expected '<label> <path>', got {raw!r}")
        label_text, rel_text = fields
        rel = PurePosixPath(rel_text.replace("\\", "/"))
        if rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 2:
            raise ValueError(f"{where}: expected '<folder>/<file>' inside the dataset, got {raw!r}")
        kind = rel.parts[0]
        if kind not in KINDS:
            raise ValueError(f"{where}: unknown folder {kind!r} (expected one of {KINDS}): {raw!r}")
        if label_text != str(_LIST_LABEL[kind]):
            raise ValueError(
                f"{where}: label {label_text!r} contradicts folder {kind!r} "
                f"(expected {_LIST_LABEL[kind]}); the list's label convention is not the assumed "
                f"one (1 = real, 0 = synthesis): {raw!r}"
            )
        stem = rel.stem
        if kind == "YouTube-real":
            group = f"yt:{stem}"
        else:
            m = _IDENTITY_RE.match(stem)
            if m is None:
                raise ValueError(f"{where}: cannot find an 'idN_' identity in {stem!r}: {raw!r}")
            group = m.group(1)
        path = root / rel.as_posix()
        if not path.is_file():
            missing.append(path)
            continue
        label = 1 if kind == "Celeb-synthesis" else 0
        items.append(VideoItem("celebdf", path, rel.as_posix(), label, group, kind, "", "test"))
    return items, missing
