"""Live webcam heart-rate monitor and ground-truth clip recorder.

The OpenCV loops (``run_live``, ``record_clip``) are thin; everything testable lives in pure
helpers: ``RollingBuffer``, ``analyze_buffer`` / ``compute_bpm``, ``render_waveform``,
``status_text`` and ``compose_view``.

Timestamps are always *seconds since start* (``time.perf_counter() - t0``), never epoch time:
``resample_uniform`` has an absolute span tolerance that epoch-sized floats would defeat.
"""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pulsecheck.config import SETTINGS
from pulsecheck.dsp import estimate_hr, resample_uniform
from pulsecheck.pipeline import face_signal, pulse_from_rgb
from pulsecheck.regions import SKIN_REGIONS, region_means, region_polygons
from pulsecheck.rppg import METHODS
from pulsecheck.viz import draw_regions

MIN_BPM_SECONDS = 5.0  # buffered time needed before a BPM is shown
MIN_WAVE_SECONDS = 2.5  # buffered time needed before the waveform is drawn
UPDATE_EVERY_S = 0.5  # BPM / waveform refresh interval
MIN_VALID_FRACTION = 0.6  # share of the buffer's time that needs a face to report a BPM
PANEL_HEIGHT = 140
WINDOW = "PulseCheck"
_QUIT_KEYS = (ord("q"), 27)  # q / Esc


class LiveError(RuntimeError):
    """A live/record problem the user should see as ``error: ...`` (the CLI maps it to exit 2)."""


class CameraError(LiveError):
    """The camera cannot be opened or stopped delivering frames."""


# --------------------------------------------------------------------------- pure helpers


class RollingBuffer:
    """The last ``seconds`` of ``(timestamp, (n_regions, 3) rgb)`` samples."""

    def __init__(self, seconds: float, n_regions: int) -> None:
        self.seconds = float(seconds)
        self.n_regions = int(n_regions)
        self._t: deque[float] = deque()
        self._rgb: deque[np.ndarray] = deque()

    def append(self, t: float, rgb: np.ndarray) -> None:
        rgb = np.asarray(rgb, np.float32)
        if rgb.shape != (self.n_regions, 3):
            raise ValueError(f"rgb must have shape ({self.n_regions}, 3), got {rgb.shape}")
        self._t.append(float(t))
        self._rgb.append(rgb)
        cutoff = float(t) - self.seconds
        while self._t[0] < cutoff:
            self._t.popleft()
            self._rgb.popleft()

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """``(t, rgb)`` with shapes ``(N,)`` and ``(N, n_regions, 3)``."""
        if not self._t:
            return np.empty(0), np.empty((0, self.n_regions, 3), np.float32)
        return np.array(self._t), np.stack(self._rgb)

    @property
    def duration(self) -> float:
        return self._t[-1] - self._t[0] if self._t else 0.0


def analyze_buffer(
    t: np.ndarray,
    rgb_regions: np.ndarray,
    method: str = "pos",
    fs: float = SETTINGS.fs,
) -> tuple[np.ndarray, float] | None:
    """Pulse waveform and heart rate from a buffer, or ``None`` if it is not usable yet.

    ``rgb_regions`` is ``(T, R, 3)`` RGB per skin region (NaN rows where a region or the whole
    face is missing). The signal is ``pipeline.face_signal`` (each region normalised by its own
    mean over the buffer, then averaged), resampled to ``fs``, run through the rPPG ``method``,
    band-passed (except CHROM, which filters itself) and read off the spectrum. Returns
    ``(pulse, bpm)``; ``bpm`` is NaN when the spectrum is flat but the pulse is still returned.
    """
    t = np.asarray(t, float)
    rgb = np.asarray(rgb_regions, float)
    if len(t) < 2:
        return None
    face = face_signal(rgb)
    try:
        _, x_u, valid = resample_uniform(t - t[0], face, fs=fs)
        # the grid starts at the first face sample, so measure coverage against the whole buffer
        covered = valid.sum() / fs
        if covered < MIN_VALID_FRACTION * (t[-1] - t[0]):
            return None
        pulse = pulse_from_rgb(x_u, method, fs)
        if pulse is None:
            return None
    except ValueError:  # too few valid samples for resampling / POS / the filter
        return None
    return pulse, estimate_hr(pulse, fs).bpm


def compute_bpm(
    t: np.ndarray, rgb_regions: np.ndarray, method: str = "pos", fs: float = SETTINGS.fs
) -> float | None:
    """Heart rate in bpm from a buffer (see ``analyze_buffer``), or ``None`` if unavailable."""
    result = analyze_buffer(t, rgb_regions, method, fs)
    if result is None or not math.isfinite(result[1]):
        return None
    return float(result[1])


def render_waveform(
    x: np.ndarray, width: int, height: int, color: tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """Draw ``x`` scaled to fit a BGR ``(height, width, 3)`` panel; empty / all-NaN is blank."""
    img = np.zeros((height, width, 3), np.uint8)
    x = np.asarray(x, float).ravel()
    ok = np.isfinite(x)
    if not ok.any():
        return img
    lo, hi = x[ok].min(), x[ok].max()
    margin = 6
    span = hi - lo
    if span > 0:
        ys = margin + (hi - x) / span * (height - 1 - 2 * margin)
    else:
        ys = np.full(len(x), (height - 1) / 2)
    xs = np.linspace(0, width - 1, len(x)) if len(x) > 1 else np.array([(width - 1) / 2])
    pts = np.stack([xs, ys], axis=1)
    # split at NaNs so gaps are not bridged by a line
    edges = np.flatnonzero(np.diff(np.concatenate(([0], ok.view(np.int8), [0]))))
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        run = np.round(pts[start:stop]).astype(np.int32)
        if len(run) == 1:
            cv2.circle(img, tuple(int(v) for v in run[0]), 1, color, -1)
        else:
            cv2.polylines(img, [run.reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)
    return img


def status_text(face_found: bool, duration: float, bpm: float | None) -> str:
    """The readout shown next to the waveform (ASCII: OpenCV fonts cannot draw an ellipsis)."""
    if not face_found:
        return "No face"
    if duration < MIN_BPM_SECONDS:
        return f"Hold still... {math.ceil(MIN_BPM_SECONDS - duration)} s"
    if bpm is None:
        return "-- bpm"
    return f"{bpm:.0f} bpm"


def compose_view(
    annotated: np.ndarray, pulse: np.ndarray | None, text: str, panel_height: int = PANEL_HEIGHT
) -> np.ndarray:
    """The annotated frame with a waveform panel (and the big readout) stacked underneath."""
    w = annotated.shape[1]
    panel = render_waveform(pulse if pulse is not None else np.empty(0), w, panel_height)
    scale = 1.6 if len(text) <= 8 else 1.0
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
    org = (max(w - tw - 12, 8), th + 12)
    cv2.putText(panel, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
    return np.vstack([annotated, panel])


# --------------------------------------------------------------------------- camera helpers


def _open_camera(camera: int, size: tuple[int, int] | None = None) -> Any:
    """Open camera ``camera`` and check it delivers a frame; raises ``CameraError`` otherwise."""
    cap = cv2.VideoCapture(camera)
    try:
        if not cap.isOpened():
            raise CameraError(f"cannot open camera {camera}")
        if size is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        ok, _ = cap.read()
        if not ok:
            raise CameraError(f"cannot open camera {camera}")
    except BaseException:
        cap.release()
        raise
    return cap


def _quit_requested() -> bool:
    key = cv2.waitKey(1) & 0xFF
    if key in _QUIT_KEYS:
        return True
    try:  # window closed with the X button
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return False


def _build_tracker() -> Any:
    from pulsecheck.face import FaceTracker  # lazy: importing mediapipe takes seconds

    return FaceTracker()


# --------------------------------------------------------------------------- live monitor


def run_live(
    camera: int = 0,
    method: str = "pos",
    window_s: float = 10.0,
    tracker: Any = None,
) -> int:
    """Show the webcam with the skin regions, a live pulse waveform and BPM. ``q``/Esc quits.

    The camera is opened (and checked) before any face tracker is built. ``tracker`` is for
    tests; when ``None`` a real ``FaceTracker`` is built (and closed on exit).
    """
    if method not in METHODS:
        print(f"error: unknown method {method!r}; choose from {sorted(METHODS)}", file=sys.stderr)
        return 2
    try:
        cap = _open_camera(camera)
    except CameraError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    own_tracker = tracker is None
    status = 0
    try:
        if own_tracker:
            tracker = _build_tracker()
        buffer = RollingBuffer(window_s, len(SKIN_REGIONS))
        n_skin = len(SKIN_REGIONS)
        nan_row = np.full((n_skin, 3), np.nan, np.float32)
        pulse: np.ndarray | None = None
        bpm: float | None = None
        last_update = -math.inf
        last_ms = -1
        failures = 0
        t0 = time.perf_counter()
        cv2.namedWindow(WINDOW)
        while True:
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if failures >= 30:
                    print("error: camera stopped delivering frames", file=sys.stderr)
                    status = 2
                    break
                continue
            failures = 0
            t = time.perf_counter() - t0
            last_ms = max(round(t * 1000), last_ms + 1)  # strictly increasing for MediaPipe
            landmarks = tracker.detect(frame, last_ms)
            if landmarks is None:
                buffer.append(t, nan_row)
                annotated = frame
            else:
                polys = region_polygons(landmarks, frame.shape[:2])
                buffer.append(t, region_means(frame, polys)[:n_skin])
                annotated = draw_regions(frame, polys)

            if t - last_update >= UPDATE_EVERY_S and buffer.duration >= MIN_WAVE_SECONDS:
                last_update = t
                bt, brgb = buffer.arrays()
                result = analyze_buffer(bt, brgb, method)
                pulse = None if result is None else result[0]
                bpm = result[1] if result is not None and math.isfinite(result[1]) else None
            text = status_text(landmarks is not None, buffer.duration, bpm)
            cv2.imshow(WINDOW, compose_view(annotated, pulse, text))
            if _quit_requested():
                break
    finally:
        cap.release()
        if own_tracker and tracker is not None and hasattr(tracker, "close"):
            tracker.close()
        cv2.destroyAllWindows()
    return status


# --------------------------------------------------------------------------- clip recorder


def _overlay(frame: np.ndarray, text: str, big: bool = False, color=(255, 255, 255)) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    if big:
        scale, thick = max(h / 150, 1.0), 6
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, thick, cv2.LINE_AA)
    else:
        cv2.putText(img, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return img


def record_clip(
    out_path: Path,
    seconds: float = 60.0,
    camera: int = 0,
    countdown_s: int = 3,
    fps: float = SETTINGS.fs,
) -> Path:
    """Record ``seconds`` of webcam video to ``out_path`` (MJPG ``.avi``) after a countdown.

    The file has a constant ``fps`` frame rate: frames are dropped or repeated so each lands at
    its true capture time, whatever rate the camera really delivers. ``q``/Esc cancels (the
    partial file is removed). Raises ``LiveError`` (``CameraError`` for a bad camera).
    """
    out_path = Path(out_path)
    if not seconds > 0:
        raise LiveError("clip length must be positive")
    if out_path.suffix.lower() != ".avi":
        raise LiveError(f"clips are saved as MJPG .avi, not {out_path.suffix or 'no extension'}")
    cap = _open_camera(camera, size=(1280, 720))
    writer = None
    created = False
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.namedWindow(WINDOW)

        def read() -> np.ndarray:
            ok, frame = cap.read()
            if not ok:
                raise CameraError("camera stopped delivering frames")
            return frame

        def cancelled() -> LiveError:
            return LiveError("recording cancelled")

        start = time.perf_counter()
        while (left := countdown_s - (time.perf_counter() - start)) > 0:
            number = _overlay(read(), str(math.ceil(left)), big=True, color=(0, 215, 255))
            cv2.imshow(WINDOW, number)
            if _quit_requested():
                raise cancelled()

        frame = read()
        h, w = frame.shape[:2]
        created = True  # from here on the file at out_path is ours
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
        if not writer.isOpened():
            raise LiveError(f"cannot write video: {out_path}")
        written = 0
        t0 = time.perf_counter()
        while (t := time.perf_counter() - t0) < seconds:
            target = int(t * fps) + 1
            while written < target:  # repeat the frame if the camera is slower than fps
                writer.write(frame)
                written += 1
            cv2.imshow(WINDOW, _overlay(frame, f"REC {t:4.1f} / {seconds:g} s", color=(0, 0, 255)))
            if _quit_requested():
                raise cancelled()
            frame = read()
        writer.release()
        writer = None
    except BaseException:
        if writer is not None:
            writer.release()
            writer = None
        if created:  # never delete a pre-existing file we did not touch
            out_path.unlink(missing_ok=True)
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return out_path
