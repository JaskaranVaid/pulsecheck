"""Facial skin regions, a non-skin control region and per-region mean color.

All region polygons are convex hulls of MediaPipe face-mesh landmarks (pixel coordinates),
clipped to the frame. "left"/"right" refer to the image, not the subject.
"""

from __future__ import annotations

import cv2
import numpy as np

from pulsecheck.config import SETTINGS

SKIN_REGIONS: tuple[str, ...] = (
    "forehead_left",
    "forehead_center",
    "forehead_right",
    "cheek_left",
    "cheek_right",
    "nose",
    "chin",
)
CONTROL = "control"
ALL_REGIONS: tuple[str, ...] = SKIN_REGIONS + (CONTROL,)

# Convex hull makes the order of the indices irrelevant.
REGION_LANDMARKS: dict[str, tuple[int, ...]] = {
    "forehead_left": (109, 67, 103, 104, 69, 108),
    "forehead_center": (10, 109, 108, 151, 337, 338),
    "forehead_right": (338, 297, 332, 333, 299, 337),
    "cheek_left": (118, 117, 123, 147, 187, 205, 36, 101, 50),
    "cheek_right": (347, 346, 352, 376, 411, 425, 266, 330, 280),
    "nose": (195, 5, 4, 51, 281, 45, 275),
    "chin": (200, 199, 175, 152, 148, 176, 377, 400),
}

# Landmarks that must never fall inside a skin region (eyes, brows, lips move/shade independently).
FORBIDDEN_LANDMARKS: dict[str, tuple[int, ...]] = {
    "eye_left": (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
    "eye_right": (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398),
    "brow_left": (70, 63, 105, 66, 107, 55, 65, 52, 53, 46),
    "brow_right": (336, 296, 334, 293, 300, 276, 283, 282, 295, 285),
    "lips": (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40,
             185),
}

_EMPTY = np.zeros((0, 2), np.int32)
CONTROL_KEEP_LEFT = 0.9  # keep the left control patch while this share of it is in frame


def _clip(points: np.ndarray, frame_shape: tuple[int, int]) -> tuple[np.ndarray, float, float]:
    """Convex hull of ``points`` intersected with the frame rectangle.

    Returns ``(polygon, clipped_area, hull_area)``; the polygon is (0, 2) if nothing is left.
    Areas are exact (before the polygon is rounded to whole pixels).
    """
    h, w = frame_shape
    hull = cv2.convexHull(np.round(points).astype(np.float32).reshape(-1, 1, 2))
    hull_area = float(cv2.contourArea(hull))
    frame_rect = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    area, inter = cv2.intersectConvexConvex(hull, frame_rect.reshape(-1, 1, 2))
    if area <= 0 or inter is None or len(inter) < 3:
        return _EMPTY.copy(), 0.0, hull_area
    return np.round(inter.reshape(-1, 2)).astype(np.int32), float(area), hull_area


def _clip_hull(points: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
    """Convex hull of ``points`` intersected with the frame rectangle; (0, 2) if nothing is left."""
    return _clip(points, frame_shape)[0]


def _control_rect(landmarks: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
    """Background patch beside the head at forehead height.

    Two candidates: image-left and image-right of the face. The side is chosen by how much of
    each patch lies inside the frame (clipped area / unclipped area): left whenever at least
    ``CONTROL_KEEP_LEFT`` of it is in frame, otherwise the better-covered side (ties go left).
    Comparing raw pixel areas instead would flip sides on sub-pixel landmark jitter whenever
    both patches are (nearly) whole, turning the control trace into a two-level square wave.
    """
    x0, y0 = landmarks.min(axis=0)
    x1, y1 = landmarks.max(axis=0)
    w = x1 - x0
    ya, yb = y0, y0 + 0.25 * (y1 - y0)
    sides = []
    for xa, xb in ((x0 - 0.45 * w, x0 - 0.15 * w), (x1 + 0.15 * w, x1 + 0.45 * w)):
        rect = np.array([[xa, ya], [xb, ya], [xb, yb], [xa, yb]], np.float32)
        poly, area, full = _clip(rect, frame_shape)
        sides.append((poly, area / full if full > 0 and len(poly) else 0.0))
    (left, left_cover), (right, right_cover) = sides
    if left_cover >= CONTROL_KEEP_LEFT or left_cover >= right_cover:
        return left  # all-empty (0, 2) when neither side is in frame
    return right


def region_polygons(landmarks: np.ndarray, frame_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Per-region polygons, each (k, 2) int32, clipped to the frame ((0, 2) when off-frame).

    ``landmarks`` is (478, 2) pixel coordinates; ``frame_shape`` is (height, width).
    """
    polys = {
        name: _clip_hull(landmarks[list(idx)], frame_shape)
        for name, idx in REGION_LANDMARKS.items()
    }
    polys[CONTROL] = _control_rect(landmarks, frame_shape)
    return {name: polys[name] for name in ALL_REGIONS}


def region_means(
    frame_bgr: np.ndarray,
    polygons: dict[str, np.ndarray],
    min_pixels: int = SETTINGS.min_region_pixels,
) -> np.ndarray:
    """Mean RGB color of each region, shape (len(ALL_REGIONS), 3) float32.

    A region with fewer than ``min_pixels`` pixels yields a NaN row.
    """
    out = np.full((len(ALL_REGIONS), 3), np.nan, np.float32)
    for row, name in enumerate(ALL_REGIONS):
        poly = polygons.get(name)
        if poly is None or len(poly) < 3:
            continue
        x, y, w, h = cv2.boundingRect(poly.reshape(-1, 1, 2))
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [poly - (x, y)], 255)
        selected = mask.astype(bool)
        if selected.sum() < min_pixels:
            continue
        crop = frame_bgr[y : y + h, x : x + w]
        bgr = crop[selected].mean(axis=0)
        out[row] = bgr[::-1]
    return out
