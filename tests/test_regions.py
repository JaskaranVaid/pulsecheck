from pathlib import Path

import cv2
import numpy as np

from pulsecheck.regions import (
    ALL_REGIONS,
    FORBIDDEN_LANDMARKS,
    SKIN_REGIONS,
    region_means,
    region_polygons,
)

FIX = Path(__file__).parent / "fixtures"
LM = np.load(FIX / "landmarks_face.npy")
SHAPE = cv2.imread(str(FIX / "face.jpg")).shape[:2]


def test_every_region_present_and_big_enough():
    polys = region_polygons(LM, SHAPE)
    assert set(polys) == set(ALL_REGIONS)
    bbox_area = np.ptp(LM[:, 0]) * np.ptp(LM[:, 1])
    for name in SKIN_REGIONS:
        assert cv2.contourArea(polys[name].astype(np.float32)) >= 0.004 * bbox_area, name


def test_no_eye_brow_or_lip_landmark_inside_skin_regions():
    polys = region_polygons(LM, SHAPE)
    for name in SKIN_REGIONS:
        for part, idx in FORBIDDEN_LANDMARKS.items():
            for i in idx:
                pt = (float(LM[i, 0]), float(LM[i, 1]))
                inside = cv2.pointPolygonTest(polys[name].astype(np.float32), pt, False)
                assert inside < 0, (name, part, i)


def test_anatomical_order():
    polys = region_polygons(LM, SHAPE)
    c = {n: polys[n].mean(0) for n in SKIN_REGIONS}
    brow_y = LM[[*FORBIDDEN_LANDMARKS["brow_left"], *FORBIDDEN_LANDMARKS["brow_right"]], 1].min()
    lips_y = LM[list(FORBIDDEN_LANDMARKS["lips"]), 1].max()
    for f in ("forehead_left", "forehead_center", "forehead_right"):
        assert c[f][1] < brow_y
    assert c["chin"][1] > lips_y
    nose_x = LM[1, 0]
    assert c["cheek_left"][0] < nose_x < c["cheek_right"][0]
    assert c["forehead_left"][0] < c["forehead_center"][0] < c["forehead_right"][0]


def test_control_region_outside_face_bbox():
    poly = region_polygons(LM, SHAPE)["control"]
    x0, x1 = LM[:, 0].min(), LM[:, 0].max()
    assert len(poly) == 0 or (poly[:, 0].max() <= x0 or poly[:, 0].min() >= x1)


def test_region_means_rgb_order_and_values():
    frame = np.zeros((*SHAPE, 3), np.uint8)
    frame[:] = (10, 20, 30)  # BGR
    m = region_means(frame, region_polygons(LM, SHAPE))
    assert m.shape == (len(ALL_REGIONS), 3)
    np.testing.assert_allclose(m[ALL_REGIONS.index("nose")], [30, 20, 10])


def test_face_at_frame_edge_gives_nan_not_crash():
    shifted = LM.copy()
    shifted[:, 0] -= LM[:, 0].min() + 0.6 * np.ptp(LM[:, 0])  # left 60% of face off-frame
    m = region_means(np.full((*SHAPE, 3), 100, np.uint8), region_polygons(shifted, SHAPE))
    assert np.isnan(m[ALL_REGIONS.index("cheek_left")]).all()
    assert not np.isnan(m[ALL_REGIONS.index("cheek_right")]).any()


def _control_side(poly: np.ndarray, landmarks: np.ndarray) -> str:
    assert len(poly) >= 3
    centre = poly[:, 0].mean()
    if centre < landmarks[:, 0].min():
        return "left"
    assert centre > landmarks[:, 0].max()
    return "right"


def test_control_side_is_stable_under_landmark_jitter():
    # Both candidate patches are fully in frame on the fixture; tiny tracker jitter must not
    # flip the control between the two sides (that would turn its trace into a square wave).
    rng = np.random.default_rng(0)
    sides = set()
    for _ in range(200):
        jittered = (LM + rng.uniform(-0.3, 0.3, LM.shape)).astype(np.float32)
        sides.add(_control_side(region_polygons(jittered, SHAPE)["control"], jittered))
    assert sides == {"left"}


def test_control_moves_right_when_face_is_at_the_left_edge():
    shifted = LM.copy()
    shifted[:, 0] -= LM[:, 0].min() - 0.2 * np.ptp(LM[:, 0])  # left patch mostly off-frame
    poly = region_polygons(shifted, SHAPE)["control"]
    assert _control_side(poly, shifted) == "right"


def test_control_stays_left_when_mostly_covered():
    # Left patch 0.45w..0.15w beside the face; push it 5% of w past the frame edge, which keeps
    # 25/30 = 83% of it: below 90%, the right side (fully covered) wins.
    w = np.ptp(LM[:, 0])
    shifted = LM.copy()
    shifted[:, 0] -= LM[:, 0].min() - 0.40 * w
    assert _control_side(region_polygons(shifted, SHAPE)["control"], shifted) == "right"
    # only 1% of w past the edge: 29/30 = 97% covered, so left is kept even though right is 100%
    shifted = LM.copy()
    shifted[:, 0] -= LM[:, 0].min() - 0.44 * w
    assert _control_side(region_polygons(shifted, SHAPE)["control"], shifted) == "left"
