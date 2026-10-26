def test_imports():
    import cv2
    import mediapipe
    import numpy
    import scipy
    from mediapipe.tasks.python import vision

    assert all(m is not None for m in (cv2, mediapipe, numpy, scipy))
    assert hasattr(vision, "FaceLandmarker")
