"""Shared test helpers: a scripted face tracker and a synthetic-video writer.

Test modules import them directly (``tests/`` has no ``__init__.py``, so pytest puts it on
``sys.path``)::

    from conftest import FakeTracker, write_video

Fixtures ``fixture_landmarks`` / ``fixture_frame_shape`` expose the face fixture.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


class FakeTracker:
    """Stand-in for ``FaceTracker``: returns the fixture landmarks scaled to each frame.

    ``landmarks`` are in the fixture frame's pixels (``face.jpg``, 640 wide x 800 high); they are
    rescaled to the frame passed to ``detect``. Calls are counted, and calls whose index is in
    ``lost_frames`` return ``None`` (face lost).
    """

    def __init__(
        self,
        landmarks: np.ndarray,
        lost_frames: set[int] | None = None,
        fixture_size: tuple[int, int] = (640, 800),
    ) -> None:
        self.landmarks = np.asarray(landmarks, np.float32)
        self.lost_frames = set(lost_frames or ())
        self.fixture_size = fixture_size  # (width, height)
        self.calls = 0

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        index = self.calls
        self.calls += 1
        if index in self.lost_frames:
            return None
        h, w = frame_bgr.shape[:2]
        scale = np.array([w / self.fixture_size[0], h / self.fixture_size[1]], np.float32)
        return self.landmarks * scale


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> Path:
    """Write BGR uint8 frames to an MJPG ``.avi`` (parent directories are created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot write video: {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="session")
def fixture_landmarks() -> np.ndarray:
    return np.load(FIXTURES / "landmarks_face.npy")


@pytest.fixture(scope="session")
def fixture_frame_shape() -> tuple[int, int]:
    return cv2.imread(str(FIXTURES / "face.jpg")).shape[:2]
