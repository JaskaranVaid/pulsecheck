"""Per-region color trace extraction from video, with an on-disk cache.

Every frame is passed through a face tracker; the mean RGB of each facial region becomes one row
of ``RegionTraces.rgb``. Frames without a face leave NaN rows.
"""

from __future__ import annotations

import hashlib
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from pulsecheck.config import SETTINGS, cache_dir
from pulsecheck.regions import ALL_REGIONS, region_means, region_polygons
from pulsecheck.viz import draw_regions

_DEFAULT_FPS = 30.0  # assumed when the source does not report a frame rate


class Tracker(Protocol):
    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray | None: ...


class NoFaceError(RuntimeError):
    """No face was found in any frame of the video."""


@dataclass
class RegionTraces:
    rgb: np.ndarray  # (T, R, 3) float32; NaN where the face or region is missing
    t: np.ndarray  # (T,) float64 seconds
    region_names: tuple[str, ...]
    source: str
    native_fps: float
    face_found: np.ndarray  # (T,) bool
    face_width_px: np.ndarray  # (T,) float32; NaN where no face
    motion: np.ndarray  # (T,) float32; mean landmark displacement / face width, NaN on the
    # first frame and on the first frame after a loss

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        try:
            with open(tmp, "wb") as f:  # a file object stops numpy appending ".npz"
                np.savez_compressed(
                    f,
                    rgb=self.rgb,
                    t=self.t,
                    region_names=np.array(self.region_names),
                    source=np.array(self.source),
                    native_fps=np.array(self.native_fps, np.float64),
                    face_found=self.face_found,
                    face_width_px=self.face_width_px,
                    motion=self.motion,
                )
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> RegionTraces:
        with np.load(Path(path), allow_pickle=False) as z:
            return cls(
                rgb=z["rgb"],
                t=z["t"],
                region_names=tuple(str(n) for n in z["region_names"]),
                source=str(z["source"]),
                native_fps=float(z["native_fps"]),
                face_found=z["face_found"],
                face_width_px=z["face_width_px"],
                motion=z["motion"],
            )

    def quality_summary(self) -> dict[str, float]:
        n = len(self.t)
        if n >= 2:
            duration = float(self.t[-1] - self.t[0] + np.median(np.diff(self.t)))
        else:
            duration = n / self.native_fps if self.native_fps > 0 else 0.0
        return {
            "duration_s": round(duration, 6),
            "face_found_ratio": float(self.face_found.mean()) if n else 0.0,
            "median_face_width_px": _nanmedian(self.face_width_px),
            "median_motion": _nanmedian(self.motion),
        }


def _nanmedian(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    return float(np.median(values)) if values.size else float("nan")


def _open_capture(source: str | Path | int) -> cv2.VideoCapture:
    if not isinstance(source, int):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"video not found: {path}")
        source = str(path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"cannot open video: {source}")
    return cap


def _timestamps(raw_ms: list[float], fps: float) -> np.ndarray:
    """Seconds per frame from the container; frame_index / fps if they are unusable."""
    ms = np.asarray(raw_ms, np.float64)
    if len(ms) > 1 and np.all(np.diff(ms) > 0):
        return ms / 1000.0
    return np.arange(len(ms), dtype=np.float64) / fps


def _frame_seconds(ms: float, index: int, fps: float, prev_ms: float | None) -> float:
    """Running estimate of a frame's time: the container timestamp while it keeps increasing
    (frame 0 may legitimately be 0), otherwise ``index / fps``."""
    if index == 0 or (prev_ms is not None and ms > prev_ms):
        return ms / 1000.0
    return index / fps


def extract_traces(
    source: str | Path | int,
    max_seconds: float | None = SETTINGS.max_seconds,
    tracker: Tracker | None = None,
    debug_video: Path | None = None,
) -> RegionTraces:
    """Mean RGB of every facial region for each frame of a video file or camera index.

    Reads frames until ``max_seconds`` of video time has elapsed (all of it when ``None``); the
    cut-off follows frame timestamps, not the container's reported fps. With ``debug_video`` set,
    each frame is written there with the regions outlined. When ``tracker`` is None a MediaPipe
    ``FaceTracker`` is created (after the video has opened) and closed again.

    Raises ``FileNotFoundError`` for a missing path, ``ValueError`` if the video cannot be opened
    or has no frames, and ``NoFaceError`` if no frame contains a face.
    """
    cap = _open_capture(source)
    own_tracker = tracker is None
    writer: cv2.VideoWriter | None = None
    try:
        if own_tracker:
            from pulsecheck.face import FaceTracker  # lazy: importing mediapipe takes seconds

            tracker = FaceTracker()
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 and math.isfinite(fps) else _DEFAULT_FPS

        means: list[np.ndarray] = []
        raw_ms: list[float] = []
        found: list[bool] = []
        widths: list[float] = []
        motion: list[float] = []
        prev_lm: np.ndarray | None = None

        t0 = 0.0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            index = len(means)
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            now = _frame_seconds(ms, index, fps, raw_ms[-1] if raw_ms else None)
            if index == 0:
                t0 = now
            elif max_seconds is not None and now - t0 >= max_seconds - 1e-6:
                break
            raw_ms.append(ms)
            landmarks = tracker.detect(frame, round(now * 1000.0))

            polys = None
            if landmarks is None:
                means.append(np.full((len(ALL_REGIONS), 3), np.nan, np.float32))
                found.append(False)
                widths.append(np.nan)
                motion.append(np.nan)
                prev_lm = None
            else:
                landmarks = np.asarray(landmarks, np.float32)
                polys = region_polygons(landmarks, frame.shape[:2])
                means.append(region_means(frame, polys))
                width = float(np.ptp(landmarks[:, 0]))
                found.append(True)
                widths.append(width)
                if prev_lm is None or width <= 0:
                    motion.append(np.nan)
                else:
                    shift = np.linalg.norm(landmarks - prev_lm, axis=1).mean()
                    motion.append(float(shift) / width)
                prev_lm = landmarks

            if debug_video is not None:
                if writer is None:
                    debug_video = Path(debug_video)
                    debug_video.parent.mkdir(parents=True, exist_ok=True)
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(debug_video), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h)
                    )
                    if not writer.isOpened():
                        raise ValueError(f"cannot write debug video: {debug_video}")
                writer.write(frame if polys is None else draw_regions(frame, polys))

        if not means:
            raise ValueError(f"cannot open video: no frames could be read from {source}")
        if not any(found):
            raise NoFaceError(f"no face found in {source}")
        return RegionTraces(
            rgb=np.stack(means).astype(np.float32),
            t=_timestamps(raw_ms, fps),
            region_names=ALL_REGIONS,
            source=str(source),
            native_fps=float(fps),
            face_found=np.array(found, bool),
            face_width_px=np.array(widths, np.float32),
            motion=np.array(motion, np.float32),
        )
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if own_tracker and tracker is not None:
            tracker.close()


def _cache_path(video_path: Path, max_seconds: float | None) -> Path:
    video_path = Path(video_path)
    stat = video_path.stat()  # FileNotFoundError for a missing video
    key = f"{video_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{max_seconds}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return cache_dir() / "traces" / f"{digest}.npz"


def cached_traces(video_path: Path, **kwargs) -> RegionTraces:
    """``extract_traces`` with results cached on disk, keyed by file identity and ``max_seconds``.

    The tracker is not part of the key. When ``debug_video`` is requested the cache is bypassed
    for reading (the video must be rendered) but still refreshed.
    """
    max_seconds = kwargs.get("max_seconds", SETTINGS.max_seconds)
    path = _cache_path(video_path, max_seconds)
    if path.exists() and kwargs.get("debug_video") is None:
        try:
            return RegionTraces.load(path)
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            pass  # unreadable cache entry: recompute and overwrite
    traces = extract_traces(video_path, **kwargs)
    traces.save(path)
    return traces
