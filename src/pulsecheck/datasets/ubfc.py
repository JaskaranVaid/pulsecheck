"""UBFC-rPPG (DATASET_2): videos with pulse-oximeter ground truth.

Layout: ``root/subject<N>/vid.avi`` and ``root/subject<N>/ground_truth.txt``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VIDEO_NAME = "vid.avi"
GROUND_TRUTH_NAME = "ground_truth.txt"
_SUBJECT_RE = re.compile(r"^subject(\d+)$")


@dataclass
class UbfcSubject:
    subject_id: str
    video: Path
    ground_truth: Path


def list_subjects(root: Path) -> list[UbfcSubject]:
    """Subjects under ``root`` that have both a video and a ground-truth file, sorted by number
    (``subject3`` before ``subject10``). Other entries are ignored.

    Raises ``FileNotFoundError`` if ``root`` is not a directory.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"UBFC root not found: {root}")
    found: list[tuple[int, UbfcSubject]] = []
    for d in root.iterdir():
        m = _SUBJECT_RE.match(d.name)
        if m is None or not d.is_dir():
            continue
        video, gt = d / VIDEO_NAME, d / GROUND_TRUTH_NAME
        if video.is_file() and gt.is_file():
            found.append((int(m.group(1)), UbfcSubject(d.name, video, gt)))
    return [s for _, s in sorted(found, key=lambda item: item[0])]


def load_ground_truth(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a ground-truth file as ``(t_s, ppg, hr)``.

    The file holds exactly three whitespace-separated numeric lines, in the order PPG, HR,
    time (s), all of the same length (at least 2). Blank lines are ignored. Anything else raises
    ``ValueError`` naming the file, so a wrong assumption about the format fails loudly.
    """
    path = Path(path)
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) != 3:
        raise ValueError(
            f"{path}: expected 3 numeric lines (PPG, HR, time), found {len(lines)} non-empty lines"
        )
    try:
        ppg, hr, t = (np.array(ln.split(), dtype=float) for ln in lines)
    except ValueError as e:
        raise ValueError(f"{path}: non-numeric value in ground truth ({e})") from e
    if not (len(ppg) == len(hr) == len(t)):
        raise ValueError(
            f"{path}: lines differ in length (PPG {len(ppg)}, HR {len(hr)}, time {len(t)})"
        )
    if len(t) < 2:
        raise ValueError(f"{path}: need at least 2 samples per line, found {len(t)}")
    if not (np.isfinite(ppg).all() and np.isfinite(hr).all() and np.isfinite(t).all()):
        raise ValueError(f"{path}: ground truth contains NaN or infinite values")
    return t, ppg, hr
