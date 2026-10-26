"""Extract region traces for every video of one or more manifests, in parallel and resumable.

    python scripts/extract_all.py --manifest C:/dev/pulsecheck-data/manifests/ffpp_c23.csv
        --manifest C:/dev/pulsecheck-data/manifests/celebdf_test.csv --workers 4

Finished videos live in the trace cache, so re-running the same command skips them (status
"cached") and continues where an interrupted run stopped. One CSV row per video goes to the log
(default ``<data dir>/results/extract_log.csv``); the printed summary (videos per hour, tracking
frames per second per worker) is the throughput measurement. A video listed twice is extracted once.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pulsecheck.batch import run_batch, unique_items
from pulsecheck.config import SETTINGS, data_dir
from pulsecheck.datasets.manifest import read_manifest


def default_workers() -> int:
    """Half the logical cores minus one (leaves the machine usable); at least 1."""
    return max(1, (os.cpu_count() or 2) // 2 - 1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--manifest", type=Path, action="append", required=True, metavar="CSV")
    p.add_argument("--workers", type=int, default=default_workers(), help="default: %(default)s")
    p.add_argument("--max-seconds", type=float, default=SETTINGS.max_seconds)
    p.add_argument("--limit", type=int, help="only the first N unique videos")
    p.add_argument("--log", type=Path, default=data_dir() / "results" / "extract_log.csv")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    if args.workers < 1:
        p.error("--workers must be at least 1")
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    items = []
    for manifest in args.manifest:
        items.extend(read_manifest(manifest))
    n_listed = len(items)
    items = unique_items(items)
    n_duplicates = n_listed - len(items)
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        print("no videos to extract", file=sys.stderr)
        return 1
    print(
        f"{len(items)} videos ({n_duplicates} duplicate listing(s) dropped), "
        f"{args.workers} worker(s), max {args.max_seconds:g} s each; log {args.log}",
        flush=True,
    )
    summary = run_batch(items, args.workers, args.max_seconds, args.log)
    print(
        "summary: "
        f"total {summary['total']:.0f} ok {summary['ok']:.0f} cached {summary['cached']:.0f} "
        f"error {summary['error']:.0f} | wall_s {summary['wall_s']:.1f} "
        f"videos_per_hour {summary['videos_per_hour']:.1f} "
        f"frames_per_s {summary['frames_per_s']:.1f} (per worker, fresh extractions only)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
