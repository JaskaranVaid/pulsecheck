"""Describe every dataset video as one manifest row and print a split report.

Writes ``ffpp_<compression>.csv`` (FaceForensics++, official pair splits) and
``celebdf_test.csv`` (Celeb-DF v2, test only) to ``<data dir>/manifests/`` (or ``--out-dir``).
FF++ needs the split files from ``scripts/fetch_ffpp_splits.py``.

    python scripts/build_manifest.py --ffpp-root C:\\dev\\pulsecheck-data\\ffpp \\
        --celebdf-root C:\\dev\\pulsecheck-data\\celebdf --compression c23 c40 --max-train-pairs 200
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from pulsecheck.config import data_dir
from pulsecheck.datasets.celebdf import list_celebdf_test
from pulsecheck.datasets.ffpp import COMPRESSIONS, FFPP_METHODS, list_ffpp, load_splits
from pulsecheck.datasets.manifest import SPLITS, VideoItem, write_manifest


def _missing_kind(root: Path, path: Path, dataset: str) -> str:
    """Kind of an expected-but-absent file, from its place in the dataset tree."""
    parts = path.relative_to(root.absolute()).parts
    if dataset == "celebdf":
        return parts[0]
    return "original" if parts[0] == "original_sequences" else parts[1]


def split_report(name: str, items: list[VideoItem], missing_kinds: Counter) -> str:
    """Counts per split x label x kind, then the missing count per kind."""
    counts = Counter((i.split, i.label, i.kind) for i in items)
    order = {s: k for k, s in enumerate(SPLITS)}
    lines = [f"{name}: {len(items)} videos", f"{'split':<6} {'label':<5} {'kind':<16} count"]
    for split, label, kind in sorted(counts, key=lambda k: (order[k[0]], k[1], k[2])):
        lines.append(f"{split:<6} {label:<5} {kind:<16} {counts[(split, label, kind)]}")
    if missing_kinds:
        detail = ", ".join(f"{k} {n}" for k, n in sorted(missing_kinds.items()))
    else:
        detail = "none"
    lines.append(f"missing: {detail}")
    return "\n".join(lines)


def _methods(text: str) -> list[str]:
    names = [m.strip() for m in text.split(",") if m.strip()]
    unknown = [m for m in names if m not in FFPP_METHODS]
    if unknown or not names:
        raise argparse.ArgumentTypeError(
            f"unknown method(s) {unknown}; choose from {list(FFPP_METHODS)}"
        )
    return names


def _emit(out_dir: Path, name: str, items, missing, root: Path, dataset: str) -> bool:
    """Write one manifest and print its report; False (nothing written) if there are no items."""
    kinds = Counter(_missing_kind(root, p, dataset) for p in missing)
    if not items:
        print(
            f"{name}: no videos found under {root} ({len(missing)} expected files missing); "
            "skipping",
            file=sys.stderr,
        )
        return False
    write_manifest(items, out_dir / f"{name}.csv")
    print(split_report(name, items, kinds))
    print(f"wrote {out_dir / (name + '.csv')}\n")
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ffpp-root", type=Path, help="FaceForensics++ root")
    p.add_argument("--celebdf-root", type=Path, help="Celeb-DF v2 root")
    p.add_argument("--compression", nargs="+", default=list(COMPRESSIONS), choices=COMPRESSIONS)
    p.add_argument("--max-train-pairs", type=int, help="keep only the first N train pairs (FF++)")
    p.add_argument(
        "--methods",
        type=_methods,
        default=list(FFPP_METHODS),
        help="comma-separated FF++ manipulation methods (default: all four)",
    )
    p.add_argument("--splits-dir", type=Path, help="default: <data dir>/ffpp/splits")
    p.add_argument("--out-dir", type=Path, help="default: <data dir>/manifests")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    if args.ffpp_root is None and args.celebdf_root is None:
        p.error("give --ffpp-root and/or --celebdf-root")
    if args.max_train_pairs is not None and args.max_train_pairs < 1:
        p.error("--max-train-pairs must be at least 1")

    out_dir = args.out_dir or data_dir() / "manifests"
    written = 0

    if args.ffpp_root is not None:
        if not args.ffpp_root.is_dir():
            print(f"skipping FF++: root not found: {args.ffpp_root}", file=sys.stderr)
        else:
            splits_dir = args.splits_dir or data_dir() / "ffpp" / "splits"
            try:
                splits = load_splits(splits_dir)
            except (OSError, ValueError) as e:
                print(
                    f"error: cannot read FF++ splits from {splits_dir}: {e}\n"
                    "run scripts/fetch_ffpp_splits.py first",
                    file=sys.stderr,
                )
                return 2
            limits = None if args.max_train_pairs is None else {"train": args.max_train_pairs}
            for c in args.compression:
                items, missing = list_ffpp(args.ffpp_root, c, splits, tuple(args.methods), limits)
                written += _emit(out_dir, f"ffpp_{c}", items, missing, args.ffpp_root, "ffpp")

    if args.celebdf_root is not None:
        if not args.celebdf_root.is_dir():
            print(f"skipping Celeb-DF: root not found: {args.celebdf_root}", file=sys.stderr)
        else:
            try:
                items, missing = list_celebdf_test(args.celebdf_root)
            except (OSError, ValueError) as e:
                print(f"error: Celeb-DF: {e}", file=sys.stderr)
                return 2
            written += _emit(out_dir, "celebdf_test", items, missing, args.celebdf_root, "celebdf")

    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
