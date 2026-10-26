"""Command-line interface: ``pulsecheck extract | hr | debug-video``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from pulsecheck.config import SETTINGS, data_dir
from pulsecheck.extract import NoFaceError, RegionTraces, Tracker, extract_traces
from pulsecheck.pipeline import Analysis, RegionResult, analyze
from pulsecheck.regions import ALL_REGIONS
from pulsecheck.rppg import METHODS

_ERRORS = (FileNotFoundError, ValueError, NoFaceError)


def _make_tracker() -> Tracker:
    """The single place a real MediaPipe face tracker is built (tests replace this)."""
    from pulsecheck.face import FaceTracker  # lazy: importing mediapipe takes seconds

    return FaceTracker()


def _require_file(path: Path) -> None:
    """Fail on a missing input before any tracker (and its model download) is created."""
    if not path.is_file():
        raise FileNotFoundError(f"video not found: {path}")


def _timed_extract(
    video: Path, max_seconds: float, debug_video: Path | None = None
) -> tuple[RegionTraces, float]:
    """Run ``extract_traces`` (no cache) with a fresh tracker; returns ``(traces, seconds)``."""
    _require_file(video)
    tracker = _make_tracker()
    start = time.perf_counter()
    try:
        traces = extract_traces(
            video, max_seconds=max_seconds, tracker=tracker, debug_video=debug_video
        )
    finally:
        close = getattr(tracker, "close", None)
        if close is not None:
            close()
    return traces, time.perf_counter() - start


def _speed_line(traces: RegionTraces, seconds: float) -> str:
    n = len(traces.t)
    return f"processed {n} frames in {seconds:.1f} s ({n / max(seconds, 1e-9):.1f} frames/s)"


def _fmt(value: float, spec: str) -> str:
    return "--" if not np.isfinite(value) else format(value, spec)


def _row(r: RegionResult) -> str:
    return f"{r.name:<18}{_fmt(r.hr.bpm, '.1f'):>8}{_fmt(r.hr.snr_db, '.1f'):>10}"


def _table(a: Analysis) -> str:
    lines = [f"method: {a.method}", f"{'region':<18}{'bpm':>8}{'snr_db':>10}"]
    lines += [_row(a.regions[n]) for n in ALL_REGIONS if n in a.regions]
    lines += [_row(a.face)]
    return "\n".join(lines)


def _cmd_extract(args: argparse.Namespace) -> int:
    traces, seconds = _timed_extract(args.video, args.max_seconds)
    out = args.out or data_dir() / "traces" / f"{args.video.stem}.npz"
    traces.save(out)
    print(f"saved traces to {out}")
    print(_speed_line(traces, seconds))
    return 0


def _cmd_hr(args: argparse.Namespace) -> int:
    traces, seconds = _timed_extract(args.video, args.max_seconds)
    methods = sorted(METHODS) if args.method == "all" else [args.method]
    q = traces.quality_summary()
    print(
        f"{args.video.name}: {q['duration_s']:.1f} s, "
        f"face found in {100 * q['face_found_ratio']:.0f}% of frames"
    )
    for method in methods:
        print()
        print(_table(analyze(traces, method=method)))
    print()
    print(_speed_line(traces, seconds))
    return 0


def _cmd_debug_video(args: argparse.Namespace) -> int:
    traces, seconds = _timed_extract(args.video, args.max_seconds, debug_video=args.out)
    print(f"wrote debug video to {args.out}")
    print(_speed_line(traces, seconds))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pulsecheck", description="Heart rate from facial video.")
    sub = p.add_subparsers(dest="command", required=True)

    def add_video(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("video", type=Path, help="input video file")
        sp.add_argument(
            "--max-seconds",
            type=float,
            default=SETTINGS.max_seconds,
            help="stop after this much video time (default: %(default)s)",
        )

    sp = sub.add_parser("extract", help="save per-region colour traces to an .npz file")
    add_video(sp)
    sp.add_argument(
        "--out", type=Path, help="output .npz (default: <data dir>/traces/<video name>.npz)"
    )
    sp.set_defaults(func=_cmd_extract)

    sp = sub.add_parser("hr", help="estimate heart rate per face region")
    add_video(sp)
    sp.add_argument(
        "--method",
        choices=[*sorted(METHODS), "all"],
        default="pos",
        help="rPPG method (default: %(default)s)",
    )
    sp.set_defaults(func=_cmd_hr)

    sp = sub.add_parser("debug-video", help="write the video with the face regions outlined")
    add_video(sp)
    sp.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output video; .avi (MJPG) is native, .mp4 works via an mp4v fallback",
    )
    sp.set_defaults(func=_cmd_debug_video)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return args.func(args)
    except _ERRORS as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
