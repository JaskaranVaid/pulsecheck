# Plan 1 — rPPG Engine (Phases 0–2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A tested Python package that reads a heart rate from video (file or live webcam) from 7 facial skin regions, using the GREEN, CHROM and POS methods, and is validated against UBFC-rPPG ground truth and the user's manual pulse counts.

**Architecture:** MediaPipe Face Landmarker gives 478 landmarks per frame → convex-hull skin regions → per-frame mean RGB per region (cached `.npz`) → resampled to a uniform 30 Hz → rPPG method → bandpass → spectrum → heart rate + SNR. Small focused modules, pure functions wherever possible, with the face tracker injectable so everything downstream is testable without a camera or a model.

**Tech Stack:** Python 3.13, mediapipe 1.1.x (Tasks API), OpenCV, NumPy, SciPy, pytest, ruff, argparse CLI.

**Spec:** `docs/plan.md` (approved 2026-10-26). This is plan 1 of 4:
1. **this plan:** Phases 0–2 (setup, extraction, rPPG engine, live monitor, heart-rate validation)
2. Phases 3–5: dataset pipeline, baseline detector, consistency features
3. Phases 6–7: modern AI (DF26) evaluation, verdict and quality gate
4. Phases 8–10: Streamlit demo, presentation, stretch CNN

## Before / in parallel ✋ (user)

Submit dataset access requests now. Nothing before Task 9 waits on them:
FaceForensics++ (Google form on `github.com/ondyari/FaceForensics`), Celeb-DF v2 (form on `github.com/yuezunli/celeb-deepfakeforensics`), UBFC-rPPG (`sites.google.com/view/ybenezeth/ubfcrppg`), DF26 (accept the license at `huggingface.co/datasets/DF26/DF26`). Download destination: `C:\dev\pulsecheck-data\<dataset>`.

## Global Constraints

- Project root: `C:\Users\jaska\OneDrive\Desktop\Deepfake project` (OneDrive; path contains a space).
- Virtual env **outside OneDrive**: `C:\dev\venvs\pulsecheck` (Python 3.13). Fallback to Python 3.12 only if mediapipe fails on 3.13.
- Data, models and caches **outside OneDrive**: `C:\dev\pulsecheck-data`, read from the `PULSECHECK_DATA` env var; default when unset: `Path.home() / "pulsecheck-data"`.
- **Never commit dataset footage, user video, or anything under the data dir.** Small synthetic fixtures and one public-domain test portrait are the only media in git.
- **Commits use the user's git identity only. No `Co-Authored-By` trailers, no "Generated with Claude" lines, in commits, PRs or files.**
- Heart-rate band: **0.7–3.3 Hz (42–198 bpm)**, Butterworth order 3, zero-phase (`sosfiltfilt`).
- Uniform analysis rate: **30 Hz**. HR windows: **8 s, 1 s step**. POS window: **1.6 s**. Max interpolated gap: **0.5 s**. Minimum analysable clip: **3 s**. Default extraction cap: **15 s**.
- Regions: 7 skin (`forehead_left, forehead_center, forehead_right, cheek_left, cheek_right, nose, chin`) + 1 `control`. "left/right" means **image** left/right.
- Settings live in `src/pulsecheck/config.py` as a frozen dataclass (deliberate simplification of the spec's `configs/default.yaml`; no YAML dependency).
- Model URL (verified 200 OK, 3,758,596 bytes): `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
- Lint: `ruff check` clean. Tests: `pytest -q` green before every commit.

## Review Focus

1. **Face lost mid-video** (head turn, hand over face): those frames become NaN; gaps ≤ 0.5 s are interpolated, longer gaps invalidate the affected HR windows. Never crash, never a silently wrong number. → Task 4, Task 5 tests.
2. **Odd or variable frame rates** (24, 25, 29.97 fps, jittered timestamps): the heart rate must still be correct after resampling on timestamps. → Task 5 test `test_resample_jittered_24fps_recovers_hr`.
3. **Very short clips** (DF26 is 4–6 s): clips under 8 s use one whole-clip window; under 3 s raise a clear `ValueError("clip too short")`. → Task 5, Task 7 tests.
4. **No face anywhere / camera missing / unreadable file:** clear error message and non-zero exit code from the CLI, not a stack trace. → Task 4, Task 7, Task 8 tests.
5. **Face near the frame edge:** region polygons clipped to the frame; a region with < 20 pixels gives NaN, never a ZeroDivision or a value from outside the frame. → Task 3 tests.

---

### Task 1: Repository, environment and settings

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md` (stub), `LICENSE` (MIT, "Jaskaran Vaid"), `src/pulsecheck/__init__.py`, `src/pulsecheck/config.py`
- Test: `tests/test_config.py`, `tests/test_smoke.py`

**Interfaces:**
- Produces:
  - `pulsecheck.config.Settings` (frozen dataclass): `fs=30.0, band=(0.7, 3.3), filter_order=3, window_s=8.0, step_s=1.0, pos_window_s=1.6, max_gap_s=0.5, min_seconds=3.0, max_seconds=15.0, min_region_pixels=20`
  - `pulsecheck.config.SETTINGS: Settings`
  - `pulsecheck.config.data_dir() -> Path` (env `PULSECHECK_DATA`, else `Path.home()/"pulsecheck-data"`; creates it)
  - `pulsecheck.config.models_dir() -> Path` (`data_dir()/"models"`), `cache_dir() -> Path` (`data_dir()/"cache"`)

- [ ] **Step 1: Init git and environment**

```powershell
cd "C:\Users\jaska\OneDrive\Desktop\Deepfake project"; git init -b main
py -3.13 -m venv C:\dev\venvs\pulsecheck
setx PULSECHECK_DATA C:\dev\pulsecheck-data
```
`pyproject.toml`: name `pulsecheck`, `requires-python = ">=3.12"`, src layout, deps `mediapipe>=1.1,<2`, `numpy>=2`, `scipy>=1.13`; extra `dev = ["pytest>=8", "ruff>=0.6"]`; script `pulsecheck = "pulsecheck.cli:main"`; ruff line-length 100. **Do not list opencv separately until you check:** after `pip install -e .[dev]`, run `pip show mediapipe` → if it already requires `opencv-contrib-python`, depend on nothing else for cv2; otherwise add `opencv-python>=4.10`. Two OpenCV packages installed together break `import cv2`.
`.gitignore`: Python defaults, `.venv/`, `data/`, `*.npz`, `*.mp4`, `*.avi`, `*.task`, `.ruff_cache/`, `.pytest_cache/`, but **not** `tests/fixtures/*`.

- [ ] **Step 2: Write failing tests**

```python
# tests/test_config.py
def test_data_dir_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PULSECHECK_DATA", str(tmp_path / "d"))
    from pulsecheck.config import data_dir, models_dir, cache_dir
    assert data_dir() == tmp_path / "d" and (tmp_path / "d").is_dir()
    assert models_dir() == tmp_path / "d" / "models"
    assert cache_dir() == tmp_path / "d" / "cache"

def test_settings_values():
    from pulsecheck.config import SETTINGS
    assert SETTINGS.fs == 30.0 and SETTINGS.band == (0.7, 3.3)
    assert SETTINGS.window_s == 8.0 and SETTINGS.pos_window_s == 1.6

# tests/test_smoke.py
def test_imports():
    import cv2, mediapipe, numpy, scipy  # noqa: F401
    from mediapipe.tasks.python import vision
    assert hasattr(vision, "FaceLandmarker")
```

- [ ] **Step 3: Run, expect FAIL** — `C:\dev\venvs\pulsecheck\Scripts\python -m pytest -q` → `ModuleNotFoundError: pulsecheck`

- [ ] **Step 4: Implement `config.py`**, then `pip install -e ".[dev]"`

- [ ] **Step 5: Run, expect PASS.** If `test_imports` fails on 3.13, recreate the venv with Python 3.12 (`uv python install 3.12` or python.org installer), record the reason in `README.md` under "Setup", and rerun.

- [ ] **Step 6: Commit and create the public repo**

```powershell
git add -A; git commit -m "chore: project skeleton, settings, environment"
gh repo create JaskaranVaid/pulsecheck --public --source . --push --description "Detecting deepfakes by checking for a real heartbeat (rPPG) in facial video"
```
If the name is taken, use `pulsecheck-deepfake-detection`. Verify with `git log --format="%an <%ae>%n%b"` that there is no co-author trailer.

---

### Task 2: Face tracker and test fixtures

**Files:**
- Create: `src/pulsecheck/face.py`, `tests/fixtures/face.jpg`, `tests/fixtures/README.md`, `tests/fixtures/landmarks_face.npy`, `scripts/make_landmark_fixture.py`
- Test: `tests/test_face.py`

**Interfaces:**
- Consumes: `models_dir()`
- Produces:
  - `face.MODEL_URL: str` (pinned URL from Global Constraints)
  - `face.ensure_model(dest_dir: Path | None = None) -> Path` downloads once (urllib, to `.part` then rename); verifies size > 1 MB
  - `class FaceTracker` with `__init__(model_path: Path | None = None)`, `detect(frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray | None` returning `(478, 2) float32` **pixel** coordinates (x, y), `close()`, and context-manager support. VIDEO running mode, `num_faces=1`. If `timestamp_ms` ≤ the previous one, use previous + 1 (MediaPipe requires strictly increasing timestamps).
  - Fixture `tests/fixtures/landmarks_face.npy`: `(478, 2) float32` landmarks of `face.jpg`, plus its `(height, width)` stored in `tests/fixtures/README.md`.

- [ ] **Step 1: Get the fixture portrait.** A frontal, public-domain portrait from Wikimedia Commons (a US federal government work, e.g. an official NASA astronaut portrait), resized to 640 px wide, saved as `tests/fixtures/face.jpg`. Write the source URL, author, and license in `tests/fixtures/README.md`.

- [ ] **Step 2: Write failing tests**

```python
# tests/test_face.py
import cv2, numpy as np, pytest
from pathlib import Path
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
    tracker.detect(img, 1000); tracker.detect(img, 1000); tracker.detect(img, 999)
```

- [ ] **Step 3: Run, expect FAIL** (`ImportError`).
- [ ] **Step 4: Implement `face.py`** using `mp.tasks.BaseOptions(model_asset_path=...)`, `mp.tasks.vision.FaceLandmarkerOptions(running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1)`, `mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))`, `detect_for_video(image, ts)`; convert normalized x, y to pixels.
- [ ] **Step 5: Run, expect PASS.**
- [ ] **Step 6: Generate the landmark fixture** with `scripts/make_landmark_fixture.py` (runs the tracker on `face.jpg`, saves `landmarks_face.npy`).
- [ ] **Step 7: Commit** — `feat: MediaPipe face tracker with model download and fixtures`

---

### Task 3: Skin regions, control region and per-region color

**Files:**
- Create: `src/pulsecheck/regions.py`, `src/pulsecheck/viz.py`
- Test: `tests/test_regions.py`

**Interfaces:**
- Consumes: fixture `landmarks_face.npy`
- Produces:
  - `SKIN_REGIONS: tuple[str, ...]` (7 names, order fixed as in Global Constraints), `CONTROL = "control"`, `ALL_REGIONS = SKIN_REGIONS + (CONTROL,)`
  - `REGION_LANDMARKS: dict[str, tuple[int, ...]]`, `FORBIDDEN_LANDMARKS: dict[str, tuple[int, ...]]` (keys `eye_left, eye_right, brow_left, brow_right, lips`)
  - `region_polygons(landmarks: np.ndarray, frame_shape: tuple[int, int]) -> dict[str, np.ndarray]`: each value `(k, 2) int32`, the **convex hull** of the region's landmarks, clipped to the frame; keys = `ALL_REGIONS`; an off-frame region gives shape `(0, 2)`
  - `region_means(frame_bgr: np.ndarray, polygons: dict[str, np.ndarray], min_pixels: int = 20) -> np.ndarray`: `(len(ALL_REGIONS), 3) float32`, **RGB** order, NaN row when a region has < `min_pixels` pixels
  - `viz.draw_regions(frame_bgr: np.ndarray, polygons: dict[str, np.ndarray], labels: bool = True) -> np.ndarray` (returns a copy)

Starting landmark sets (convex hull makes order irrelevant). Adjust until the invariant tests pass **and** the overlay looks right on the fixture:

```python
REGION_LANDMARKS = {
    "forehead_left":   (109, 67, 103, 104, 69, 108),
    "forehead_center": (10, 109, 108, 151, 337, 338),
    "forehead_right":  (338, 297, 332, 333, 299, 337),
    "cheek_left":      (118, 117, 123, 147, 187, 205, 36, 101, 50),
    "cheek_right":     (347, 346, 352, 376, 411, 425, 266, 330, 280),
    "nose":            (195, 5, 4, 51, 281, 45, 275),
    "chin":            (200, 199, 175, 152, 148, 176, 377, 400),
}
FORBIDDEN_LANDMARKS = {
    "eye_left":  (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246),
    "eye_right": (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398),
    "brow_left": (70, 63, 105, 66, 107, 55, 65, 52, 53, 46),
    "brow_right": (336, 296, 334, 293, 300, 276, 283, 282, 295, 285),
    "lips": (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185),
}
```
**Control region:** a rectangle beside the head at forehead height. Let the face bbox (from all landmarks) be `(x0, y0, x1, y1)`, width `w`. Candidate rectangles are `x ∈ [x0 − 0.45w, x0 − 0.15w]` (left) and `[x1 + 0.15w, x1 + 0.45w]` (right), with `y ∈ [y0, y0 + 0.25·(y1 − y0)]`. Clip both to the frame and keep the one with the larger remaining area.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_regions.py  (load LM = np.load(fixtures/"landmarks_face.npy"), SHAPE = fixture (h, w))
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
                assert cv2.pointPolygonTest(polys[name].astype(np.float32), pt, False) < 0, (name, part, i)

def test_anatomical_order():
    c = {n: region_polygons(LM, SHAPE)[n].mean(0) for n in SKIN_REGIONS}
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
    frame = np.zeros((*SHAPE, 3), np.uint8); frame[:] = (10, 20, 30)  # BGR
    m = region_means(frame, region_polygons(LM, SHAPE))
    assert m.shape == (len(ALL_REGIONS), 3)
    np.testing.assert_allclose(m[ALL_REGIONS.index("nose")], [30, 20, 10])

def test_face_at_frame_edge_gives_nan_not_crash():
    shifted = LM.copy(); shifted[:, 0] -= LM[:, 0].min() + 0.6 * np.ptp(LM[:, 0])  # left 60% of face off-frame
    m = region_means(np.full((*SHAPE, 3), 100, np.uint8), region_polygons(shifted, SHAPE))
    assert np.isnan(m[ALL_REGIONS.index("cheek_left")]).all()
    assert not np.isnan(m[ALL_REGIONS.index("cheek_right")]).any()
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement `regions.py`** (hull via `cv2.convexHull`; clip with `np.clip` then re-hull; means via bounding-rect crop + `cv2.fillPoly` mask) **and `viz.draw_regions`** (one distinct color per region, control drawn dashed or grey).
- [ ] **Step 4: Run, expect PASS.** Write `draw_regions` output for the fixture to `data_dir()/debug/regions_fixture.png` and look at it. Regions must sit on skin.
- [ ] **Step 5: Commit** — `feat: facial skin regions, control region and per-region color means`

---

### Task 4: Trace extraction and caching

**Files:**
- Create: `src/pulsecheck/extract.py`
- Test: `tests/test_extract.py`, `tests/conftest.py` (shared `FakeTracker` and `write_video` helpers)

**Interfaces:**
- Consumes: `FaceTracker.detect`, `region_polygons`, `region_means`, `draw_regions`, `ALL_REGIONS`, `SETTINGS`, `cache_dir()`
- Produces:
  - `class Tracker(Protocol)`: `detect(frame_bgr, timestamp_ms) -> np.ndarray | None`
  - `class NoFaceError(RuntimeError)`
  - `@dataclass class RegionTraces`: `rgb: np.ndarray (T, R, 3) float32` (NaN where face/region missing), `t: np.ndarray (T,) float64 seconds`, `region_names: tuple[str, ...]`, `source: str`, `native_fps: float`, `face_found: np.ndarray (T,) bool`, `face_width_px: np.ndarray (T,) float32`, `motion: np.ndarray (T,) float32` (mean landmark displacement / face width, NaN on first frame or after loss). Methods `save(path: Path) -> None`, `classmethod load(path: Path) -> RegionTraces`, `quality_summary() -> dict[str, float]` with keys `duration_s, face_found_ratio, median_face_width_px, median_motion`.
  - `extract_traces(source: str | Path | int, max_seconds: float | None = SETTINGS.max_seconds, tracker: Tracker | None = None, debug_video: Path | None = None) -> RegionTraces`. Timestamps from `CAP_PROP_POS_MSEC`; if 0 or non-increasing, fall back to `frame_index / fps`. Raises `FileNotFoundError` (missing path), `ValueError("cannot open video: …")`, `NoFaceError` (face in 0 frames).
  - `cached_traces(video_path: Path, **kwargs) -> RegionTraces`: cache file `cache_dir()/traces/<sha1(abs path | size | mtime | max_seconds)>.npz`

- [ ] **Step 1: Write failing tests** (`conftest.FakeTracker(landmarks, lost_frames=set())` returns the fixture landmarks scaled to the frame, or `None` for frame indices in `lost_frames`. `write_video(path, frames, fps)` uses `cv2.VideoWriter` with the `MJPG` codec in `.avi`.)

```python
def test_traces_follow_color_modulation(tmp_path):
    # 3 s @30 fps, uniform frames whose green value = 100 + 5*sin(2π·1.2·t)
    path = tmp_path / "with space" / "v.avi"; ...
    tr = extract_traces(path, tracker=FakeTracker(LM))
    g = tr.rgb[:, ALL_REGIONS.index("nose"), 1]
    expected = 100 + 5 * np.sin(2 * np.pi * 1.2 * tr.t)
    assert np.corrcoef(g, expected)[0, 1] > 0.98          # MJPG is lossy; correlation not equality
    assert tr.rgb.shape == (90, len(ALL_REGIONS), 3) and abs(tr.native_fps - 30) < 0.5

def test_lost_frames_are_nan(tmp_path):
    tr = extract_traces(video, tracker=FakeTracker(LM, lost_frames={10, 11, 12}))
    assert np.isnan(tr.rgb[10:13]).all() and not tr.face_found[10:13].any()
    assert tr.quality_summary()["face_found_ratio"] == pytest.approx(87 / 90)

def test_no_face_raises(tmp_path):
    with pytest.raises(NoFaceError):
        extract_traces(video, tracker=FakeTracker(LM, lost_frames=set(range(1000))))

def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        extract_traces(Path("nope.avi"), tracker=FakeTracker(LM))

def test_max_seconds_caps_frames(tmp_path):   # 3 s video, max_seconds=1.0 → 30 frames
    ...

def test_save_load_roundtrip_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PULSECHECK_DATA", str(tmp_path / "data"))
    a = cached_traces(video, tracker=FakeTracker(LM)); b = cached_traces(video, tracker=None)  # 2nd call must hit cache
    np.testing.assert_array_equal(a.rgb, b.rgb); assert a.region_names == b.region_names
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement `extract.py`.** Create a real `FaceTracker` only when `tracker is None` (and close it after). `debug_video` writes each frame with `draw_regions` applied.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** — `feat: per-region color trace extraction with caching`

---

### Task 5: Signal processing (`dsp.py`)

*(Named `dsp.py`, not `signal.py`, to avoid confusion with Python's standard `signal` module.)*

**Files:**
- Create: `src/pulsecheck/dsp.py`
- Test: `tests/test_dsp.py`, `tests/synth.py` (synthetic signal helpers)

**Interfaces:**
- Consumes: `SETTINGS`
- Produces:
  - `resample_uniform(t: np.ndarray, x: np.ndarray, fs: float = 30.0, max_gap_s: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]` → `(t_u, x_u, valid)`. `x` has shape `(T, ...)`. NaN samples are dropped before linear interpolation. Output samples inside a gap longer than `max_gap_s` get `valid=False` (values there are still linearly filled so filters can run).
  - `bandpass(x: np.ndarray, fs: float, low_hz: float = 0.7, high_hz: float = 3.3, order: int = 3) -> np.ndarray` (`butter(..., output="sos")` + `sosfiltfilt`, `padlen=min(3*(2*order+1)*2, len(x)-1)`)
  - `power_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]` (`scipy.signal.periodogram`, Hann window, `nfft=max(4096, next_pow2(len(x)))`)
  - `@dataclass(frozen=True) class HRResult: bpm: float; peak_hz: float; snr_db: float`
  - `estimate_hr(pulse: np.ndarray, fs: float, band: tuple[float, float] = (0.7, 3.3)) -> HRResult`. Peak = max power inside `band`. SNR (de Haan & Jeanne 2013 style) = 10·log10(signal/noise): signal = power in `[f0±0.1 Hz] ∪ [2f0±0.2 Hz]`, noise = all other power in `[0.5, 5.0] Hz`.
  - `@dataclass class HRSeries: t_center: np.ndarray; bpm: np.ndarray; snr_db: np.ndarray` (NaN for invalid windows)
  - `hr_series(pulse: np.ndarray, fs: float, window_s: float = 8.0, step_s: float = 1.0, valid: np.ndarray | None = None) -> HRSeries`. A window is invalid when > 20% of its samples are `valid=False`. Signal shorter than `window_s` but ≥ `min_seconds` gives one whole-signal window. Shorter than `min_seconds` raises `ValueError("clip too short")`.
  - `tests/synth.py`: `synth_rgb(bpm: float, seconds: float, fs: float = 30.0, pulse_amp: float = 0.002, intensity_noise: tuple[float, float] | None = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray]` → `(t, rgb (T, 3))`, defined exactly as
    `rgb = DC * (1 + pulse_amp * v * w(t)) * (1 + a_n * sin(2π f_n t))`, with `DC = [180, 130, 110]`, blood-volume vector `v = [0.33, 0.77, 0.53]`, waveform `w(t) = sin(2π f0 t) + 0.3 sin(4π f0 t)`, `f0 = bpm/60`, and `(f_n, a_n) = intensity_noise` (or no noise term when `None`). The noise term multiplies **all channels equally**, which simulates lighting flicker. Plus tiny Gaussian sensor noise (σ = 0.05, seeded).

- [ ] **Step 1: Write failing tests**

```python
def test_bandpass_keeps_72bpm_rejects_out_of_band():
    fs = 30; t = np.arange(0, 20, 1 / fs)
    keep = bandpass(np.sin(2 * np.pi * 1.2 * t), fs); assert keep[60:-60].std() > 0.6
    for f in (0.15, 5.0):
        assert bandpass(np.sin(2 * np.pi * f * t), fs)[60:-60].std() < 0.1

@pytest.mark.parametrize("bpm", [48, 72, 110, 180])
def test_estimate_hr_on_pure_pulse(bpm):
    t, rgb = synth_rgb(bpm, 20)
    r = estimate_hr(bandpass(rgb[:, 1] - rgb[:, 1].mean(), 30), 30)
    assert abs(r.bpm - bpm) < 2 and r.snr_db > 5

def test_snr_drops_with_noise():
    rng = np.random.default_rng(0); t, rgb = synth_rgb(72, 20); g = rgb[:, 1] - rgb[:, 1].mean()
    clean = estimate_hr(bandpass(g, 30), 30).snr_db
    noisy = estimate_hr(bandpass(g + rng.normal(0, 3 * g.std(), g.size), 30), 30).snr_db
    assert noisy < clean - 3

def test_resample_jittered_24fps_recovers_hr():
    rng = np.random.default_rng(1); t = np.cumsum(rng.uniform(0.8, 1.2, 24 * 15) / 24)
    x = np.sin(2 * np.pi * 1.25 * t)
    tu, xu, valid = resample_uniform(t, x, fs=30)
    assert valid.all() and np.allclose(np.diff(tu), 1 / 30)
    assert abs(estimate_hr(bandpass(xu, 30), 30).bpm - 75) < 2

def test_resample_marks_long_gap_invalid_and_fills_short_gap():
    t = np.arange(0, 10, 1 / 30); x = np.sin(t); x[30:39] = np.nan; x[150:180] = np.nan  # 0.3 s and 1.0 s gaps
    tu, xu, valid = resample_uniform(t, x)
    assert not np.isnan(xu).any()
    assert valid[(tu > 1.05) & (tu < 1.25)].all()
    assert not valid[(tu > 5.1) & (tu < 5.9)].any()

def test_hr_series_windows_and_short_clip_rules():
    t, rgb = synth_rgb(72, 20); p = bandpass(rgb[:, 1] - rgb[:, 1].mean(), 30)
    s = hr_series(p, 30); assert len(s.bpm) == 13 and np.nanmax(np.abs(s.bpm - 72)) < 3
    s5 = hr_series(p[: 5 * 30], 30); assert len(s5.bpm) == 1 and abs(s5.bpm[0] - 72) < 4
    with pytest.raises(ValueError, match="too short"):
        hr_series(p[: 2 * 30], 30)

def test_hr_series_invalid_window_is_nan():
    t, rgb = synth_rgb(72, 20); p = bandpass(rgb[:, 1] - rgb[:, 1].mean(), 30)
    valid = np.ones_like(p, bool); valid[0:150] = False
    assert np.isnan(hr_series(p, 30, valid=valid).bpm[0])
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement `dsp.py` and `tests/synth.py`.**
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** — `feat: resampling, bandpass, spectrum, heart rate and SNR`

---

### Task 6: rPPG methods (GREEN, CHROM, POS)

**Files:**
- Create: `src/pulsecheck/rppg.py`
- Test: `tests/test_rppg.py`

**Interfaces:**
- Consumes: `bandpass`, `estimate_hr`, `synth_rgb`
- Produces: `green(rgb: np.ndarray, fs: float) -> np.ndarray`, `chrom(rgb: np.ndarray, fs: float) -> np.ndarray`, `pos(rgb: np.ndarray, fs: float, window_s: float = 1.6) -> np.ndarray`, and `METHODS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {"green": green, "chrom": chrom, "pos": pos}`. Input `rgb` is `(N, 3)` in RGB order, uniformly sampled, no NaN. Output is the `(N,)` pulse signal, **zero-mean, not bandpassed** (the caller bandpasses).

Algorithms (implement exactly):

```python
# GREEN: g / mean(g) - 1
# CHROM (de Haan & Jeanne 2013), global version:
#   Cn = rgb / rgb.mean(0)
#   X = 3*Rn - 2*Gn ;  Y = 1.5*Rn + Gn - 1.5*Bn
#   Xf, Yf = bandpass(X), bandpass(Y) ;  alpha = std(Xf)/std(Yf) ;  S = Xf - alpha*Yf
#   (CHROM is the one method that bandpasses internally; returning S is fine)
# POS (Wang et al. 2017):
#   l = round(window_s * fs);  H = zeros(N)
#   P = [[0, 1, -1], [-2, 1, 1]]
#   for n in range(l, N + 1):
#       C = rgb[n-l:n].T / rgb[n-l:n].mean(0)[:, None]      # 3 x l, temporally normalized
#       S = P @ C                                           # 2 x l
#       h = S[0] + (S[0].std() / (S[1].std() + 1e-9)) * S[1]
#       H[n-l:n] += h - h.mean()                            # overlap-add
#   return H - H.mean()
```

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.parametrize("name", ["green", "chrom", "pos"])
def test_all_methods_find_clean_pulse(name):
    t, rgb = synth_rgb(72, 20)
    assert abs(estimate_hr(bandpass(METHODS[name](rgb, 30), 30), 30).bpm - 72) < 2

@pytest.mark.parametrize("name", ["chrom", "pos"])
def test_chrom_pos_cancel_in_band_lighting_flicker(name):
    t, rgb = synth_rgb(72, 20, intensity_noise=(1.8, 0.01))   # 108-bpm flicker, 5x the pulse
    assert abs(estimate_hr(bandpass(METHODS[name](rgb, 30), 30), 30).bpm - 72) < 2

def test_green_is_fooled_by_lighting_flicker():
    """Documents WHY we use POS: the green channel alone locks onto the flicker."""
    t, rgb = synth_rgb(72, 20, intensity_noise=(1.8, 0.01))
    assert abs(estimate_hr(bandpass(green(rgb, 30), 30), 30).bpm - 108) < 2

def test_pos_output_shape_and_zero_mean():
    t, rgb = synth_rgb(72, 10); h = pos(rgb, 30)
    assert h.shape == (300,) and abs(h.mean()) < 1e-9
```

- [ ] **Step 2: Run, expect FAIL.**  - [ ] **Step 3: Implement.**  - [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** — `feat: GREEN, CHROM and POS rPPG methods`

---

### Task 7: Analysis pipeline and CLI (`extract`, `hr`, `debug-video`)

**Files:**
- Create: `src/pulsecheck/pipeline.py`, `src/pulsecheck/cli.py`
- Test: `tests/test_pipeline.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `RegionTraces`, `extract_traces`, `cached_traces`, `NoFaceError`, `resample_uniform`, `bandpass`, `estimate_hr`, `hr_series`, `HRResult`, `HRSeries`, `METHODS`, `SKIN_REGIONS`, `SETTINGS`
- Produces:
  - `@dataclass class RegionResult: name: str; pulse: np.ndarray; hr: HRResult; series: HRSeries`
  - `@dataclass class Analysis: method: str; fs: float; t: np.ndarray; valid: np.ndarray; regions: dict[str, RegionResult]; face: RegionResult; quality: dict[str, float]`
  - `analyze(traces: RegionTraces, method: str = "pos", fs: float = SETTINGS.fs) -> Analysis`. Per region: resample → rPPG → bandpass → whole-clip `estimate_hr` + `hr_series`. `face` = the same on the **nan-mean of the 7 skin regions' RGB** per frame (the most robust single estimate). `control` gets analyzed like any other region (later plans compare skin vs control). Raises `ValueError("clip too short")` under `min_seconds`, and `KeyError` for an unknown method.
  - `cli.main(argv: list[str] | None = None) -> int` with subcommands:
    - `extract VIDEO [--max-seconds S] [--out FILE.npz]`
    - `hr VIDEO [--method pos|chrom|green|all] [--max-seconds S]` prints a table (region × bpm, SNR) plus the face estimate
    - `debug-video VIDEO --out FILE.mp4`
    - (`live` and `record` are added in Task 8)
  - `extract` and `hr` print a final line `processed <N> frames in <S> s (<F> frames/s)`. This is the face-tracking speed the spec's compute budget (§7) needs.
  - Errors (`FileNotFoundError`, `ValueError`, `NoFaceError`) print `error: <message>` to stderr and return **2**.
  - `cli._make_tracker()` is the single place a real `FaceTracker` is built (tests monkeypatch it).

- [ ] **Step 1: Write failing tests**

```python
def make_traces(bpm=72, seconds=20, fs=30):   # RegionTraces from synth_rgb: same pulse in all skin regions, flat noise in control
    ...

def test_analyze_finds_hr_in_every_skin_region_and_face():
    a = analyze(make_traces(72))
    assert abs(a.face.hr.bpm - 72) < 2
    for n in SKIN_REGIONS: assert abs(a.regions[n].hr.bpm - 72) < 3
    assert a.regions["control"].hr.snr_db < a.face.hr.snr_db - 6

def test_analyze_short_clip_raises():
    with pytest.raises(ValueError, match="too short"): analyze(make_traces(seconds=2))

def test_analyze_unknown_method_raises():
    with pytest.raises(KeyError): analyze(make_traces(), method="magic")

def test_cli_hr_prints_table(tmp_path, monkeypatch, capsys):   # synthetic .avi + FakeTracker via monkeypatch of cli._make_tracker
    assert main(["hr", str(video), "--method", "all"]) == 0
    out = capsys.readouterr().out
    assert "face" in out and "pos" in out and "chrom" in out

def test_cli_missing_file_exit_2(capsys):
    assert main(["hr", "does-not-exist.mp4"]) == 2
    assert "error:" in capsys.readouterr().err
```

- [ ] **Step 2: Run, expect FAIL.**  - [ ] **Step 3: Implement.**  - [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** — `feat: analysis pipeline and command-line interface`

---

### Task 8: Live webcam monitor and clip recorder

**Files:**
- Create: `src/pulsecheck/live.py`; Modify: `src/pulsecheck/cli.py` (add `live`, `record`)
- Test: `tests/test_live.py`

**Interfaces:**
- Consumes: `FaceTracker`, `region_polygons`, `region_means`, `draw_regions`, `resample_uniform`, `METHODS`, `bandpass`, `estimate_hr`, `SKIN_REGIONS`, `data_dir()`
- Produces:
  - `class RollingBuffer`: `__init__(self, seconds: float, n_regions: int)`, `append(self, t: float, rgb: np.ndarray) -> None` (`rgb` shape `(n_regions, 3)`; drops samples older than `seconds`), `arrays(self) -> tuple[np.ndarray, np.ndarray]`, property `duration -> float`
  - `render_waveform(x: np.ndarray, width: int, height: int, color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray` (BGR uint8 `(height, width, 3)`, signal scaled to fit; all-NaN or empty input gives a blank panel)
  - `run_live(camera: int = 0, method: str = "pos", window_s: float = 10.0) -> int`. Shows the frame with regions, a waveform panel and a big BPM readout. Until 5 s are buffered it shows "Hold still… n s". BPM is recomputed every 0.5 s from the face (skin nan-mean) signal. When the face is lost it shows "No face" and keeps going. `q`/Esc quits. If the camera can't be opened it prints `error: cannot open camera 0` and returns 2.
  - `record_clip(out_path: Path, seconds: float = 60.0, camera: int = 0) -> Path`: 3-2-1 countdown, then records (MJPG `.avi`) with an on-screen timer.
  - CLI `live [--camera N] [--method M]`. CLI `record [--seconds S] [--camera N] [--out PATH]` (default out: `data_dir()/own/clip_<timestamp>.avi`); after recording it prompts `Pulse you counted (bpm): ` and appends `path,counted_bpm,recorded_at` to `data_dir()/own/clips.csv`.

- [ ] **Step 1: Write failing tests**

```python
def test_rolling_buffer_drops_old_samples():
    b = RollingBuffer(seconds=2.0, n_regions=8)
    for i in range(100): b.append(i / 30, np.full((8, 3), i, np.float32))
    t, rgb = b.arrays()
    assert t[0] >= t[-1] - 2.0 and rgb.shape[1:] == (8, 3) and b.duration <= 2.0

def test_render_waveform_shape_and_blank_cases():
    img = render_waveform(np.sin(np.linspace(0, 10, 300)), 400, 120)
    assert img.shape == (120, 400, 3) and img.dtype == np.uint8 and img.any()
    assert not render_waveform(np.array([]), 400, 120).any()
    assert not render_waveform(np.full(10, np.nan), 400, 120).any()

def test_run_live_bad_camera_returns_2(capsys):
    assert run_live(camera=99) == 2 and "cannot open camera" in capsys.readouterr().err
```

- [ ] **Step 2: Run, expect FAIL.**  - [ ] **Step 3: Implement.**  - [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Manual check ✋ (needs the user):** `pulsecheck live`. The regions stick to the face while moving, and the BPM settles within ~10 s. Then record one 60 s clip and run `pulsecheck hr` on it at 720p to get the frames/s figure. Write that figure into `docs/plan.md` §7, replacing the "30–60 frames/second" estimate.
- [ ] **Step 6: Commit** — `feat: live webcam heart-rate monitor and clip recorder`

---

### Task 9: UBFC-rPPG loader and heart-rate evaluation

**Files:**
- Create: `src/pulsecheck/datasets/__init__.py`, `src/pulsecheck/datasets/ubfc.py`, `src/pulsecheck/evaluation.py`, `scripts/eval_hr.py`
- Test: `tests/test_ubfc.py`, `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `cached_traces`, `analyze`, `resample_uniform`, `bandpass`, `hr_series`, `data_dir()`
- Produces:
  - `@dataclass class UbfcSubject: subject_id: str; video: Path; ground_truth: Path`
  - `list_subjects(root: Path) -> list[UbfcSubject]` for the DATASET_2 layout `root/subject*/vid.avi` + `ground_truth.txt`, sorted numerically (`subject3` before `subject10`)
  - `load_ground_truth(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]` → `(t_s, ppg, hr)`; file = 3 whitespace-separated numeric lines in order **PPG, HR, time (s)**. ⚠ This format comes from secondary sources; confirm it on the first real file and adjust the parser + test if it differs.
  - `evaluation.hr_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]` with keys `mae, rmse, pearson, n`, ignoring pairs where either is NaN
  - `scripts/eval_hr.py --ubfc-root PATH [--own-clips CSV] [--methods green,chrom,pos] [--max-seconds 60]`:
    - UBFC: for each subject and method, compare `analyze(...).face.series.bpm` with the reference series = `hr_series(bandpass(resampled GT PPG))` on the **same** 8 s / 1 s windows.
    - Own clips: whole-clip `face.hr.bpm` vs `counted_bpm`.
    - Writes `data_dir()/results/hr_eval.csv` (per subject × method) and `docs/results/hr_eval.md` (summary table: method × MAE / RMSE / Pearson, separately for UBFC and own clips, plus run date and the exact command).

- [ ] **Step 1: Write failing tests**

```python
def test_list_subjects_sorts_numerically(tmp_path):
    for n in (10, 3, 1):
        d = tmp_path / f"subject{n}"; d.mkdir(); (d / "vid.avi").touch(); (d / "ground_truth.txt").touch()
    assert [s.subject_id for s in list_subjects(tmp_path)] == ["subject1", "subject3", "subject10"]

def test_load_ground_truth_three_lines(tmp_path):
    p = tmp_path / "ground_truth.txt"
    p.write_text("0.1 0.2 0.3\n70 71 72\n0.0 0.033 0.066\n")
    t, ppg, hr = load_ground_truth(p)
    np.testing.assert_allclose(ppg, [0.1, 0.2, 0.3]); np.testing.assert_allclose(hr, [70, 71, 72])
    np.testing.assert_allclose(t, [0.0, 0.033, 0.066])

def test_hr_metrics_ignores_nan():
    m = hr_metrics(np.array([70, 80, np.nan]), np.array([72, 78, 75]))
    assert m["n"] == 2 and m["mae"] == pytest.approx(2.0) and m["rmse"] == pytest.approx(2.0)
```

- [ ] **Step 2: Run, expect FAIL.**  - [ ] **Step 3: Implement.**  - [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Real run ✋ (needs UBFC downloaded to `C:\dev\pulsecheck-data\ubfc`):** `python scripts/eval_hr.py --ubfc-root C:\dev\pulsecheck-data\ubfc\DATASET_2 --max-seconds 60`. **Acceptance:** POS face MAE on UBFC ≤ 5 bpm, and POS beats GREEN on MAE. If it doesn't, stop and investigate (superpowers:systematic-debugging) before Plan 2. Don't tune parameters until the numbers look good.
- [ ] **Step 6: Commit** — `feat: UBFC-rPPG loader and heart-rate evaluation` (+ `docs/results/hr_eval.md` once real numbers exist)

---

### Task 10: Documentation for Plan 1

**Files:**
- Modify: `README.md`
- Create: `docs/METHOD.md`, `docs/INTERVIEW_NOTES.md`
- Test: none (docs). Verification = links resolve and the commands in the README run on a fresh venv.

**Interfaces:** consumes everything above; produces the docs that Plans 2–4 extend.

- [ ] **Step 1: `README.md`:** one-paragraph pitch; status ("Phase 2 of 9: heart-rate engine done; detector in progress"); setup (venv outside OneDrive, `PULSECHECK_DATA`); commands (`pulsecheck live`, `hr`, `debug-video`, `record`); test command; data-licence note.
- [ ] **Step 2: `docs/METHOD.md`** (rPPG section): why skin color pulses; the regions (with the fixture overlay image `docs/figures/regions.png`, which shows the public-domain portrait); GREEN vs CHROM vs POS in plain words plus the formulas; bandpass, spectrum and SNR definitions; the jitter/resampling rationale; a results table linked to `docs/results/hr_eval.md`.
- [ ] **Step 3: `docs/INTERVIEW_NOTES.md`:** Q&A for every decision in this plan (why MediaPipe Face Landmarker, why convex hulls, why 0.7–3.3 Hz, why resample to 30 Hz, why POS, how SNR is defined, why a control region, why the tracker is injectable, why data is outside OneDrive), each answered in 2–4 plain sentences.
- [ ] **Step 4: Verify** — `ruff check .`, `pytest -q` all green; README commands work.
- [ ] **Step 5: Commit and push** — `docs: method, interview notes and README for the rPPG engine`; `git push`.

---

## Done when (Plan 1)

- `pytest -q` green, `ruff check .` clean, pushed to `github.com/JaskaranVaid/pulsecheck` with the user as the only author.
- `pulsecheck live` shows a stable BPM on the user's face (✋ user check).
- `docs/results/hr_eval.md` has real UBFC numbers, POS MAE ≤ 5 bpm (✋ needs UBFC download), plus own-clip comparisons (✋ needs 3–5 recorded clips).
