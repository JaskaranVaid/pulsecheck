"""Build the per-video feature table of one or more manifests from cached traces.

    python scripts/build_features.py --manifest C:/dev/pulsecheck-data/manifests/ffpp_c23.csv

Only cached traces are read (run ``extract_all.py`` first); nothing is ever extracted here. Each
video becomes one row: the manifest fields followed by the features of the requested groups.
Videos without a usable cache entry are skipped and counted by reason (uncached, load error,
analyze error), and the share of NaN values per feature is printed. The default output is
``<data dir>/features/<manifest stem>.csv``.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

from pulsecheck.batch import unique_items
from pulsecheck.config import SETTINGS, data_dir
from pulsecheck.datasets.manifest import COLUMNS, read_manifest
from pulsecheck.extract import RegionTraces, cache_path_for
from pulsecheck.features import FEATURE_GROUPS, video_features

# Errors that mean "this cache entry cannot be used": truncated/corrupt archive, wrong version
# (ValueError), missing array (KeyError), unreadable file (OSError).
LOAD_ERRORS = (ValueError, KeyError, OSError, EOFError, zipfile.BadZipFile)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--manifest", type=Path, action="append", required=True, metavar="CSV")
    p.add_argument("--out", type=Path, help="default: <data dir>/features/<manifest stem>.csv")
    p.add_argument("--groups", default="ABCD", help="feature groups to compute; default ABCD")
    p.add_argument("--max-seconds", type=float, default=SETTINGS.max_seconds)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    unknown = sorted(set(args.groups) - set(FEATURE_GROUPS))
    if unknown:
        p.error(f"unknown group(s) {unknown}; choose from {list(FEATURE_GROUPS)}")
    if args.out is None:
        if len(args.manifest) != 1:
            p.error("--out is required with more than one --manifest")
        args.out = data_dir() / "features" / f"{args.manifest[0].stem}.csv"
    return args


def _load_cached(item, max_seconds: float) -> RegionTraces | None:
    """Cached traces of ``item``, or ``None`` when its video or cache entry does not exist."""
    try:
        cache = cache_path_for(item.path, item.root, max_seconds)
    except FileNotFoundError:
        return None
    if not cache.exists():
        return None
    return RegionTraces.load(cache)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    items = []
    for manifest in args.manifest:
        items.extend(read_manifest(manifest))
    n_listed = len(items)
    items = unique_items(items)
    if not items:
        print("no videos listed", file=sys.stderr)
        return 1

    feature_names = [n for g in "ABCD" if g in args.groups for n in FEATURE_GROUPS[g]]
    rows: list[dict] = []
    skipped: Counter[str] = Counter()
    for item in items:
        try:
            traces = _load_cached(item, args.max_seconds)
        except LOAD_ERRORS as e:
            skipped["load error"] += 1
            print(f"load error: {item.rel_path}: {e}", file=sys.stderr)
            continue
        if traces is None:
            skipped["uncached"] += 1
            continue
        try:
            feats = video_features(traces, groups=args.groups)
        except ValueError as e:  # face unusable or clip too short
            skipped["analyze error"] += 1
            print(f"analyze error: {item.rel_path}: {e}", file=sys.stderr)
            continue
        row = {c: getattr(item, c) for c in COLUMNS}
        row["path"] = item.path.as_posix()
        rows.append({**row, **feats})

    total_skipped = sum(skipped.values())
    print(
        f"{len(items)} videos ({n_listed - len(items)} duplicate listing(s) dropped): "
        f"{len(rows)} written, {total_skipped} skipped",
        flush=True,
    )
    for reason in ("uncached", "load error", "analyze error"):
        print(f"  skipped, {reason}: {skipped[reason]}")
    if not rows:
        print("no features written: every video was skipped (uncached?)", file=sys.stderr)
        return 1

    table = pd.DataFrame(rows, columns=[*COLUMNS, *feature_names])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    print("NaN rate per feature:")
    for name in feature_names:
        print(f"  {name:<24s} {table[name].isna().mean():6.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
