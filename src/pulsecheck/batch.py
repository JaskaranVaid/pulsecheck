"""Resumable, crash-tolerant batch trace extraction with a run log.

``extract_one`` turns one manifest video into one result row and never raises for a per-video
failure. ``run_batch`` runs many of them (in-process or in a spawn ``Pool``), appends a CSV row per
finished video as it completes, and returns throughput figures. Resuming is free: finished videos
are already in the trace cache, so a re-run reports them as ``cached`` without building a tracker.
"""

from __future__ import annotations

import csv
import functools
import multiprocessing
import time
import zipfile
import zlib
from collections.abc import Iterator
from pathlib import Path

import cv2

from pulsecheck.config import SETTINGS
from pulsecheck.datasets.manifest import VideoItem
from pulsecheck.extract import RegionTraces, cache_path_for, cached_traces

LOG_COLUMNS = ("rel_path", "dataset", "status", "seconds", "frames", "message")
_SECONDS_PER_HOUR = 3600.0
# what a cache entry that cannot be loaded raises (same set as ``cached_traces`` treats as a miss)
_UNREADABLE_CACHE = (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, zlib.error)


def extract_one(item: VideoItem, max_seconds: float = SETTINGS.max_seconds) -> dict[str, object]:
    """Extract (or load) the traces of one video and describe the outcome as a log row.

    Returns ``{"rel_path", "dataset", "status", "seconds", "frames", "message"}``. ``status`` is
    ``"cached"`` when a valid cache entry already existed (no tracker is built), ``"ok"`` after a
    fresh extraction, ``"error"`` otherwise. ``seconds`` is the wall time spent on this video;
    ``frames`` is the number of frames in its traces (0 on error); ``message`` is ``""`` on
    success, else ``"<ExceptionType>: <text>"``.

    Per-video failures never propagate: the expected ones (no face, unreadable or missing video,
    OS errors, OpenCV errors) and any other ``Exception``, so one odd file cannot end an overnight
    run. ``KeyboardInterrupt`` still propagates.
    """
    start = time.perf_counter()
    status, frames, message = "error", 0, ""
    try:
        root = item.root
        cache_file = cache_path_for(item.path, root, max_seconds)
        traces = None
        if cache_file.exists():
            try:
                traces = RegionTraces.load(cache_file)
                status = "cached"
            except _UNREADABLE_CACHE:  # unreadable or old-version entry: re-extract below
                traces = None
        if traces is None:
            traces = cached_traces(item.path, root=root, max_seconds=max_seconds)
            status = "ok"
        frames = len(traces.t)
    except Exception as e:  # noqa: BLE001 - see docstring
        # Expected: NoFaceError, ValueError, FileNotFoundError, OSError, cv2.error. Anything else
        # (e.g. a tracker crash) is caught too: it must cost one video, not the run.
        status, frames, message = "error", 0, f"{type(e).__name__}: {e}"
    return {
        "rel_path": item.rel_path,
        "dataset": item.dataset,
        "status": status,
        "seconds": time.perf_counter() - start,
        "frames": frames,
        "message": message,
    }


def init_worker() -> None:
    """Pool initializer: one OpenCV thread per worker, so N workers do not oversubscribe cores."""
    cv2.setNumThreads(1)


def unique_items(items: list[VideoItem]) -> list[VideoItem]:
    """``items`` without repeats of the same video path (first one wins, order kept).

    Two workers extracting the same video would race on one cache file.
    """
    seen: set[Path] = set()
    out = []
    for item in items:
        if item.path not in seen:
            seen.add(item.path)
            out.append(item)
    return out


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def run_batch(
    items: list[VideoItem],
    workers: int,
    max_seconds: float,
    log_path: Path,
    progress_every: int = 25,
) -> dict[str, float]:
    """Extract every video, appending one row per finished video to the CSV at ``log_path``.

    Repeats of the same path in ``items`` are dropped first. ``workers == 1`` runs in this process;
    otherwise a spawn ``Pool`` of ``min(workers, len(items))`` processes is used, and rows arrive
    in completion order. The log is created (with header) if absent, appended to otherwise, and
    flushed after every row, so a crash loses nothing already finished. One progress line goes to
    stdout every ``progress_every`` finished videos and at the end. On ``KeyboardInterrupt`` the
    pool is terminated, a resume hint is printed, and the interrupt is re-raised.

    Returns ``total, ok, cached, error`` (counts), ``wall_s``, ``videos_per_hour``
    ((ok + cached) / wall time) and ``frames_per_s`` (frames of freshly extracted videos / the
    summed per-video time of those: tracking speed of one worker; 0.0 if none were extracted).
    """
    items = unique_items(items)
    total = len(items)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "cached": 0, "error": 0}
    ok_frames, ok_seconds = 0, 0.0
    start = time.perf_counter()
    pool = None

    def rows() -> Iterator[dict[str, object]]:
        nonlocal pool
        if workers <= 1 or total <= 1:
            for item in items:
                yield extract_one(item, max_seconds=max_seconds)
            return
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(min(workers, total), initializer=init_worker)
        work = functools.partial(extract_one, max_seconds=max_seconds)
        yield from pool.imap_unordered(work, items, chunksize=1)

    new_file = not log_path.exists() or log_path.stat().st_size == 0
    try:
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(LOG_COLUMNS)
                f.flush()
            for done, row in enumerate(rows(), start=1):
                writer.writerow(
                    [f"{row[c]:.3f}" if c == "seconds" else row[c] for c in LOG_COLUMNS]
                )
                f.flush()
                counts[str(row["status"])] += 1
                if row["status"] == "ok":
                    ok_frames += int(row["frames"])  # type: ignore[call-overload]
                    ok_seconds += float(row["seconds"])  # type: ignore[arg-type]
                if done % progress_every == 0 or done == total:
                    elapsed = time.perf_counter() - start
                    eta = elapsed / done * (total - done)
                    print(
                        f"[{done}/{total}] ok {counts['ok']} cached {counts['cached']} "
                        f"error {counts['error']} | elapsed {_fmt_duration(elapsed)} "
                        f"ETA {_fmt_duration(eta)}",
                        flush=True,
                    )
        if pool is not None:
            pool.close()
            pool.join()
    except KeyboardInterrupt:
        if pool is not None:
            pool.terminate()
            pool.join()
        print(
            f"interrupted: finished videos are in the cache and in {log_path}; "
            "re-run the same command to resume",
            flush=True,
        )
        raise
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise

    wall = time.perf_counter() - start
    return {
        "total": float(total),
        "ok": float(counts["ok"]),
        "cached": float(counts["cached"]),
        "error": float(counts["error"]),
        "wall_s": wall,
        "videos_per_hour": (counts["ok"] + counts["cached"]) / wall * _SECONDS_PER_HOUR
        if wall > 0
        else 0.0,
        "frames_per_s": ok_frames / ok_seconds if ok_seconds > 0 else 0.0,
    }
