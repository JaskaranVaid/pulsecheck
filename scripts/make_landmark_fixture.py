"""Regenerate tests/fixtures/landmarks_face.npy from tests/fixtures/face.jpg."""

from pathlib import Path

import cv2
import numpy as np

from pulsecheck.face import FaceTracker

FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def main() -> None:
    img = cv2.imread(str(FIX / "face.jpg"))
    if img is None:
        raise SystemExit(f"Cannot read {FIX / 'face.jpg'}")
    with FaceTracker() as tracker:
        lm = tracker.detect(img, 0)
    if lm is None:
        raise SystemExit("No face detected in face.jpg")
    np.save(FIX / "landmarks_face.npy", lm.astype(np.float32))
    h, w = img.shape[:2]
    print(f"saved landmarks {lm.shape} {lm.dtype} for image (height, width)=({h}, {w})")


if __name__ == "__main__":
    main()
