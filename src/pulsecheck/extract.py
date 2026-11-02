"""Per-region color trace extraction from video, with an on-disk cache.

Every frame is passed through a face tracker; the mean RGB of each facial region becomes one row
of ``RegionTraces.rgb``. Frames without a face leave NaN rows.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import sys
import uuid
import warnings
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from pulsecheck.config import SETTINGS, cache_dir
from pulsecheck.regions import ALL_REGIONS, region_means, region_polygons
from pulsecheck.viz import draw_regions

_DEFAULT_FPS = 30.0  # assumed when the source does not report a frame rate
# Schema/behaviour version of saved traces. Bump it whenever extraction changes what a trace
# contains (regions, landmarks, colour averaging): it is part of the cache key and stored in every
# file, and ``RegionTraces.load`` refuses any other version. 2: deterministic control-side rule.
# 3: face landmarks stored (float16) so regions can be recomputed without re-tracking.
TRACES_VERSION = 3
_N_LANDMARKS = 478  # MediaPipe face mesh with iris refinement


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
    landmarks: np.ndarray  # (T, 478, 2) float16 pixel x, y; NaN rows where no face. float16 keeps
    # ~0.25-1 px resolution at 512-2048 px, enough to rebuild the regions, not for sub-pixel motion

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # unique per process and call: parallel workers may write the same entry at once
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            with open(tmp, "wb") as f:  # a file object stops numpy appending ".npz"
                np.savez_compressed(
                    f,
                    traces_version=np.array(TRACES_VERSION),
                    rgb=self.rgb,
                    t=self.t,
                    region_names=np.array(self.region_names),
                    source=np.array(self.source),
                    native_fps=np.array(self.native_fps, np.float64),
                    face_found=self.face_found,
                    face_width_px=self.face_width_px,
                    motion=self.motion,
                    landmarks=self.landmarks,
                )
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> RegionTraces:
        """Read a file written by ``save``.

        Raises ``ValueError`` if it was written by another ``TRACES_VERSION`` (or has none).
        """
        with np.load(Path(path), allow_pickle=False) as z:
            version = int(z["traces_version"]) if "traces_version" in z.files else None
            if version != TRACES_VERSION:
                raise ValueError(
                    f"{path}: traces version {version}, expected {TRACES_VERSION}; re-extract"
                )
            return cls(
                rgb=z["rgb"],
                t=z["t"],
                region_names=tuple(str(n) for n in z["region_names"]),
                source=str(z["source"]),
                native_fps=float(z["native_fps"]),
                face_found=z["face_found"],
                face_width_px=z["face_width_px"],
                motion=z["motion"],
                landmarks=z["landmarks"],
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
        lms: list[np.ndarray] = []
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
                lms.append(np.full((_N_LANDMARKS, 2), np.nan, np.float16))
                found.append(False)
                widths.append(np.nan)
                motion.append(np.nan)
                prev_lm = None
            else:
                landmarks = np.asarray(landmarks, np.float32)
                polys = region_polygons(landmarks, frame.shape[:2])
                means.append(region_means(frame, polys))
                lms.append(landmarks.astype(np.float16))
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
            landmarks=np.stack(lms),
        )
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if own_tracker and tracker is not None:
            if sys.exc_info()[0] is None:
                tracker.close()  # extraction succeeded: a failing close is a real error
            else:
                with contextlib.suppress(Exception):  # never mask the error already propagating
                    tracker.close()


def cache_path_for(
    video_path: Path,
    root: Path | None = None,
    max_seconds: float | None = SETTINGS.max_seconds,
) -> Path:
    """Cache file of a video's traces (it need not exist yet).

    With ``root`` the key is ``TRACES_VERSION``, the path relative to ``root``, the file size and
    ``max_seconds``: no mtime and no absolute path, so moving or copying the data folder keeps the
    cache valid. Without ``root`` it is the resolved absolute path, size, mtime and ``max_seconds``.
    ``max_seconds`` is normalised with ``float()``, so ``1`` and ``1.0`` share an entry.
    Raises ``FileNotFoundError`` for a missing video and ``ValueError`` if it is outside ``root``.
    """
    video_path = Path(video_path)
    stat = video_path.stat()  # FileNotFoundError for a missing video
    cap = None if max_seconds is None else float(max_seconds)
    if root is not None:
        rel = video_path.relative_to(root).as_posix()
        key = f"v{TRACES_VERSION}|{rel}|{stat.st_size}|{cap}"
    else:
        key = f"v{TRACES_VERSION}|{video_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{cap}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return cache_dir() / "traces" / f"{digest}.npz"


def cached_traces(video_path: Path, root: Path | None = None, **kwargs) -> RegionTraces:
    """``extract_traces`` with results cached on disk.

    The key is described in ``cache_path_for`` (pass ``root`` for a relocatable data folder); an
    entry from another version is treated as a miss, as is any unreadable or corrupt entry. The
    tracker is not part of the key. When ``debug_video`` is requested the cache is bypassed for
    reading (the video must be rendered) but still refreshed. A cache write that fails (disk full,
    permissions) emits a ``UserWarning`` and the traces are returned anyway.
    """
    path = cache_path_for(video_path, root, kwargs.get("max_seconds", SETTINGS.max_seconds))
    if path.exists() and kwargs.get("debug_video") is None:
        try:
            return RegionTraces.load(path)
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, zlib.error):
            pass  # unreadable or other-version entry: recompute and overwrite
    traces = extract_traces(video_path, **kwargs)
    try:
        traces.save(path)
    except OSError as e:
        warnings.warn(f"could not write trace cache {path}: {e}", UserWarning, stacklevel=2)
    return traces
