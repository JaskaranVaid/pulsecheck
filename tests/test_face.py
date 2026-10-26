from pathlib import Path

import cv2
import numpy as np
import pytest

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def tracker():
    from pulsecheck.face import FaceTracker

    with FaceTracker() as t:
        yield t


def test_detects_478_landmarks_inside_image(tracker):
    img = cv2.imread(str(FIX / "face.jpg"))
    lm = tracker.detect(img, 0)
    assert lm.shape == (478, 2) and lm.dtype == np.float32
    h, w = img.shape[:2]
    assert (lm[:, 0] >= 0).all() and (lm[:, 0] < w).all()
    assert (lm[:, 1] >= 0).all() and (lm[:, 1] < h).all()


def test_blank_frame_returns_none(tracker):
    assert tracker.detect(np.zeros((480, 640, 3), np.uint8), 33) is None


def test_non_increasing_timestamp_does_not_raise(tracker):
    img = cv2.imread(str(FIX / "face.jpg"))
    tracker.detect(img, 1000)
    tracker.detect(img, 1000)
    tracker.detect(img, 999)
