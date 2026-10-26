"""Evaluate GREEN / CHROM / POS heart-rate estimates.

* UBFC-rPPG: windowed HR from the video vs HR from the pulse-oximeter PPG (8 s windows, 1 s step).
* Own clips: whole-clip HR vs the pulse counted by hand (``data_dir()/own/clips.csv``).

Writes a per-subject CSV and a Markdown summary table.

    python scripts/eval_hr.py --ubfc-root C:\\dev\\pulsecheck-data\\ubfc\\DATASET_2 --max-seconds 60
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

import numpy as np

from pulsecheck.config import data_dir
from pulsecheck.datasets.ubfc import list_subjects, load_ground_truth
from pulsecheck.evaluation import align_series, hr_metrics, reference_series
from pulsecheck.extract import NoFaceError, cached_traces
from pulsecheck.pipeline import analyze
from pulsecheck.rppg import METHODS

DEFAULT_MD_OUT = Path("docs/results/hr_eval.md")
COLUMNS = (
    "dataset",
    "subject",
    "method",
    "n",
    "mae",
    "rmse",
    "pearson",
    "pred_bpm",
    "ref_bpm",
    "error",
)
_FAILURES = (NoFaceError, ValueError, FileNotFoundError)

Pairs = dict[str, tuple[list[np.ndarray], list[np.ndarray]]]


def _warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def _num(value: float) -> str:
    return "" if value is None or not np.isfinite(value) else f"{value:.4f}"


def _row(dataset: str, subject: str, method: str, **fields) -> dict[str, str]:
    row = dict.fromkeys(COLUMNS, "")
    row.update(dataset=dataset, subject=subject, method=method)
    for key, value in fields.items():
        row[key] = value if isinstance(value, str) else _num(value)
    return row


def _eval_ubfc(subjects, methods, max_seconds, rows, pairs: Pairs) -> None:
    for subject in subjects:
        print(f"[ubfc] {subject.subject_id}", flush=True)
        try:
            t_gt, ppg, _ = load_ground_truth(subject.ground_truth)
            ref = reference_series(t_gt, ppg)
            traces = cached_traces(subject.video, max_seconds=max_seconds)
        except _FAILURES as e:
            _warn(f"{subject.subject_id}: {e}")
            rows.extend(
                _row("ubfc", subject.subject_id, m, error=str(e) or type(e).__name__)
                for m in methods
            )
            continue
        for method in methods:
            try:
                series = analyze(traces, method).face.series
                pred, truth = align_series(series, ref)
            except _FAILURES as e:
                _warn(f"{subject.subject_id}/{method}: {e}")
                rows.append(
                    _row("ubfc", subject.subject_id, method, error=str(e) or type(e).__name__)
                )
                continue
            m = hr_metrics(pred, truth)
            pairs[method][0].append(pred)
            pairs[method][1].append(truth)
            rows.append(
                _row(
                    "ubfc",
                    subject.subject_id,
                    method,
                    n=str(m["n"]),
                    mae=m["mae"],
                    rmse=m["rmse"],
                    pearson=m["pearson"],
                    pred_bpm=float(np.nanmean(pred)) if m["n"] else np.nan,
                    ref_bpm=float(np.nanmean(truth)) if m["n"] else np.nan,
                )
            )


def _read_own_clips(csv_path: Path) -> list[tuple[Path, float]]:
    clips: list[tuple[Path, float]] = []
    with open(csv_path, newline="") as f:
        for i, rec in enumerate(csv.DictReader(f), start=2):
            raw = (rec.get("path") or "").strip()
            try:
                counted = float(rec.get("counted_bpm") or "")
            except ValueError:
                _warn(f"{csv_path}:{i}: bad counted_bpm {rec.get('counted_bpm')!r}; skipped")
                continue
            candidates = (
                [Path(raw)] if Path(raw).is_absolute() else [csv_path.parent / raw, Path(raw)]
            )
            path = next((p for p in candidates if raw and p.is_file()), None)
            if path is None:
                _warn(f"{csv_path}:{i}: clip not found: {raw}; skipped")
                continue
            clips.append((path, counted))
    return clips


def _eval_own(clips, methods, max_seconds, rows, pairs: Pairs) -> None:
    for path, counted in clips:
        print(f"[own] {path.name}", flush=True)
        try:
            traces = cached_traces(path, max_seconds=max_seconds)
        except _FAILURES as e:
            _warn(f"{path}: {e}")
            rows.extend(
                _row("own", str(path), m, ref_bpm=counted, error=str(e) or type(e).__name__)
                for m in methods
            )
            continue
        for method in methods:
            try:
                bpm = analyze(traces, method).face.hr.bpm
            except _FAILURES as e:
                _warn(f"{path}/{method}: {e}")
                rows.append(
                    _row(
                        "own", str(path), method, ref_bpm=counted, error=str(e) or type(e).__name__
                    )
                )
                continue
            m = hr_metrics(np.array([bpm]), np.array([counted]))
            pairs[method][0].append(np.array([bpm]))
            pairs[method][1].append(np.array([counted]))
            rows.append(
                _row(
                    "own",
                    str(path),
                    method,
                    n=str(m["n"]),
                    mae=m["mae"],
                    rmse=m["rmse"],
                    pred_bpm=bpm,
                    ref_bpm=counted,
                )
            )


def _fmt(value: float) -> str:
    return "--" if not np.isfinite(value) else f"{value:.2f}"


def _summary_table(dataset: str, rows, pairs: Pairs, methods, unit: str) -> list[str]:
    failed = len({r["subject"] for r in rows if r["dataset"] == dataset and r["error"]})
    total = len({r["subject"] for r in rows if r["dataset"] == dataset})
    lines = [
        f"{total - failed} of {total} {unit}s evaluated; {failed} failed (excluded below).",
        "",
        f"| Method | MAE (bpm) | RMSE (bpm) | Pearson r | {unit.capitalize()}s | Pairs |",
        "|---|---|---|---|---|---|",
    ]
    for method in methods:
        preds, refs = pairs[method]
        if preds:
            m = hr_metrics(np.concatenate(preds), np.concatenate(refs))
        else:
            m = {"mae": np.nan, "rmse": np.nan, "pearson": np.nan, "n": 0}
        ok = len(
            {
                r["subject"]
                for r in rows
                if r["dataset"] == dataset and r["method"] == method and not r["error"]
            }
        )
        lines.append(
            f"| {method} | {_fmt(m['mae'])} | {_fmt(m['rmse'])} | {_fmt(m['pearson'])} | {ok} | {m['n']} |"
        )
    return lines


def _write_markdown(path: Path, command: str, rows, ubfc_pairs, own_pairs, args) -> None:
    out = [
        "# Heart-rate evaluation",
        "",
        f"Run date: {date.today().isoformat()}",
        "",
        "Command:",
        "",
        "```",
        command,
        "```",
        "",
        f"Face estimate (nan-mean of the 7 skin regions), extraction capped at {args.max_seconds:g} s per video.",
        "",
    ]
    if args.ubfc_root is not None:
        out += [
            "## UBFC-rPPG",
            "",
            (
                "Windowed HR (8 s windows, 1 s step) against HR from the pulse-oximeter PPG. "
                "Metrics pool every window of every subject."
            ),
            "",
            *_summary_table("ubfc", rows, ubfc_pairs, args.methods, "subject"),
            "",
        ]
    if args.own_clips is not None:
        out += [
            "## Own clips",
            "",
            "Whole-clip HR against the pulse counted by hand; one pair per clip.",
            "",
            *_summary_table("own", rows, own_pairs, args.methods, "clip"),
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")


def _write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _methods(text: str) -> list[str]:
    names = [m.strip() for m in text.split(",") if m.strip()]
    unknown = [m for m in names if m not in METHODS]
    if unknown or not names:
        raise argparse.ArgumentTypeError(
            f"unknown method(s) {unknown}; choose from {sorted(METHODS)}"
        )
    return names


def _quote(arg: str) -> str:
    return f'"{arg}"' if not arg or any(c.isspace() for c in arg) else arg


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ubfc-root", type=Path, help="UBFC DATASET_2 directory (subject*/vid.avi)")
    p.add_argument("--own-clips", type=Path, help="CSV written by `pulsecheck record`")
    p.add_argument("--methods", type=_methods, default=["green", "chrom", "pos"])
    p.add_argument("--max-seconds", type=float, default=60.0, help="seconds of each video to use")
    p.add_argument("--csv-out", type=Path, help="default: <data dir>/results/hr_eval.csv")
    p.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = p.parse_args(argv)
    if args.ubfc_root is None and args.own_clips is None:
        p.error("give --ubfc-root and/or --own-clips")

    # Validate inputs before anything slow (or a tracker) is created.
    try:
        subjects = list_subjects(args.ubfc_root) if args.ubfc_root is not None else []
        if args.ubfc_root is not None and not subjects:
            raise FileNotFoundError(
                f"no subject*/vid.avi + ground_truth.txt found under {args.ubfc_root}"
            )
        if args.own_clips is not None and not args.own_clips.is_file():
            raise FileNotFoundError(f"own-clips CSV not found: {args.own_clips}")
        clips = _read_own_clips(args.own_clips) if args.own_clips is not None else []
    except (FileNotFoundError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    rows: list[dict[str, str]] = []
    ubfc_pairs: Pairs = {m: ([], []) for m in args.methods}
    own_pairs: Pairs = {m: ([], []) for m in args.methods}
    if args.ubfc_root is not None:
        _eval_ubfc(subjects, args.methods, args.max_seconds, rows, ubfc_pairs)
    if args.own_clips is not None:
        _eval_own(clips, args.methods, args.max_seconds, rows, own_pairs)

    csv_out = args.csv_out or data_dir() / "results" / "hr_eval.csv"
    command = " ".join(["python", "scripts/eval_hr.py", *map(_quote, argv)])
    _write_csv(csv_out, rows)
    _write_markdown(args.md_out, command, rows, ubfc_pairs, own_pairs, args)
    print(f"wrote {csv_out}\nwrote {args.md_out}")

    # Fail if a requested dataset produced no usable result at all.
    ok = {d: any(r["dataset"] == d and not r["error"] for r in rows) for d in ("ubfc", "own")}
    wanted = [
        d for d, asked in (("ubfc", args.ubfc_root), ("own", args.own_clips)) if asked is not None
    ]
    return 0 if all(ok[d] or (d == "own" and not clips) for d in wanted) else 1


if __name__ == "__main__":
    sys.exit(main())
