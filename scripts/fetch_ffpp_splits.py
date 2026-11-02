"""Download the official FaceForensics++ train/val/test split files.

Fetches the three JSON files from the FaceForensics repository, checks them with
``load_splits`` (shape, 3-digit ids, no id in two splits) and only then writes them to
``<data dir>/ffpp/splits/`` (or ``--out``).

    python scripts/fetch_ffpp_splits.py
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from pulsecheck.config import data_dir
from pulsecheck.datasets.ffpp import load_splits
from pulsecheck.datasets.manifest import SPLITS

BASE_URL = "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits"
TIMEOUT_S = 30.0  # per request


def download(url: str) -> bytes:
    """GET ``url`` and return the body (raises ``urllib.error.URLError`` on failure)."""
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
        return response.read()


def fetch_splits(out_dir: Path, fetch: Callable[[str], bytes] = download) -> None:
    """Download, validate and write the three split files into ``out_dir``.

    Nothing is written to ``out_dir`` unless all three files validate.
    """
    bodies = {split: fetch(f"{BASE_URL}/{split}.json") for split in SPLITS}
    with tempfile.TemporaryDirectory() as tmp:
        for split, body in bodies.items():
            (Path(tmp) / f"{split}.json").write_bytes(body)
        load_splits(Path(tmp))  # raises ValueError when the files are not what we expect
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, body in bodies.items():
        (out_dir / f"{split}.json").write_bytes(body)


def main(argv: list[str] | None = None, fetch: Callable[[str], bytes] = download) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, help="default: <data dir>/ffpp/splits")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    out = args.out or data_dir() / "ffpp" / "splits"
    try:
        fetch_splits(out, fetch)
    except (OSError, ValueError) as e:  # URLError is an OSError; bad JSON is a ValueError
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {', '.join(f'{s}.json' for s in SPLITS)} to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
