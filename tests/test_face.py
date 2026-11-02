import threading
import time
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


def _fake_download(size: int = 1_500_000):
    """A ``urlretrieve`` stand-in that writes ``size`` bytes slowly, so callers overlap."""

    def urlretrieve(url, filename):
        with open(filename, "wb") as f:
            for _ in range(5):
                f.write(b"x" * (size // 5))
                f.flush()
                time.sleep(0.05)

    return urlretrieve


def test_ensure_model_concurrent_calls_share_one_download(tmp_path, monkeypatch):
    from pulsecheck.face import MODEL_NAME, ensure_model

    monkeypatch.setattr("urllib.request.urlretrieve", _fake_download())
    results, errors = [], []

    def call():
        try:
            results.append(ensure_model(tmp_path))
        except Exception as e:  # noqa: BLE001 - the test asserts there are none
            errors.append(e)

    threads = [threading.Thread(target=call) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert results == [tmp_path / MODEL_NAME] * 2
    assert list(tmp_path.iterdir()) == [tmp_path / MODEL_NAME]  # no leftover .part files


def test_ensure_model_returns_the_winners_model_when_replace_fails(tmp_path, monkeypatch):
    from pulsecheck.face import MODEL_NAME, ensure_model

    monkeypatch.setattr("urllib.request.urlretrieve", _fake_download())
    final = tmp_path / MODEL_NAME

    def lose_the_race(self, target):
        final.write_bytes(b"y" * 1_200_000)  # another process installed a valid model first
        raise PermissionError("[WinError 32] in use")

    monkeypatch.setattr(Path, "replace", lose_the_race)
    assert ensure_model(tmp_path) == final
    assert final.stat().st_size == 1_200_000
    assert list(tmp_path.iterdir()) == [final]


def test_ensure_model_replace_failure_without_a_model_still_raises(tmp_path, monkeypatch):
    from pulsecheck.face import ensure_model

    monkeypatch.setattr("urllib.request.urlretrieve", _fake_download())

    def broken(self, target):
        raise PermissionError("no")

    monkeypatch.setattr(Path, "replace", broken)
    with pytest.raises(PermissionError):
        ensure_model(tmp_path)
    assert list(tmp_path.iterdir()) == []
