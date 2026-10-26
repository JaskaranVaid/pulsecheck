"""MediaPipe Face Landmarker wrapper: 478 face landmarks in pixel coordinates."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Self

import cv2
import mediapipe as mp
import numpy as np

from pulsecheck.config import models_dir

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_NAME = "face_landmarker.task"
_MIN_MODEL_BYTES = 1_000_000


def ensure_model(dest_dir: Path | None = None) -> Path:
    """Return the path to the face landmarker model, downloading it once if absent."""
    dest_dir = Path(dest_dir) if dest_dir is not None else models_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / MODEL_NAME
    if path.exists() and path.stat().st_size > _MIN_MODEL_BYTES:
        return path
    part = path.with_name(path.name + ".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, part)
        if part.stat().st_size <= _MIN_MODEL_BYTES:
            raise RuntimeError(
                f"Downloaded face model is too small ({part.stat().st_size} bytes)"
            )
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)
    return path


class FaceTracker:
    """Single-face landmark tracker (MediaPipe VIDEO running mode)."""

    def __init__(self, model_path: Path | None = None) -> None:
        model = Path(model_path) if model_path is not None else ensure_model()
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._last_ts: int | None = None

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        """Return (478, 2) float32 pixel (x, y) landmarks, or None if no face found."""
        ts = int(timestamp_ms)
        if self._last_ts is not None and ts <= self._last_ts:
            ts = self._last_ts + 1  # MediaPipe needs strictly increasing timestamps
        self._last_ts = ts
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts)
        if not result.face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        pts = np.array(
            [(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]], dtype=np.float32
        )
        return pts

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
