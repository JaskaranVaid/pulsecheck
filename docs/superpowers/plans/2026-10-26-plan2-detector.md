# Plan 2 — Dataset Pipeline, Baseline Detector, Consistency Features (Phases 3–5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the rPPG engine into a measurable deepfake detector. That means dataset manifests and batch extraction for FaceForensics++ and Celeb-DF, a classic "is there a pulse?" baseline, and our physiological-consistency features, all evaluated without leakage. The headline is a before/after results table.

**Architecture:** manifest (one row per video, with label/group/split) → batch extraction into a versioned, batch-safe trace cache (now also storing landmarks) → `analyze()` on **fixed 4 s windows** → per-video feature row (groups A pulse presence, B spatial agreement, C physiological consistency, D quality) → scikit-learn pipelines. Real-face reference statistics for group C are fitted *inside* the pipeline on training reals only → grouped evaluation → a markdown results file plus figures. Everything is testable on synthetic videos and traces. Real numbers come from the user's downloads.

**Tech Stack:** Python 3.13, existing `pulsecheck` package, NumPy/SciPy, **scikit-learn, pandas, matplotlib (new)**, pytest, ruff.

**Spec:** `docs/plan.md` (approved 2026-10-26), §5 Data, §6.2 Features, §6.3 Evaluation rules, §6.4 tables 2–4 and 6, §7 Compute, §8 Phases 3–5. Plan 1 (`docs/superpowers/plans/2026-10-26-plan1-rppg-engine.md`) is merged. Its carried-forward rulings are binding here and appear in Global Constraints.

## Before / in parallel ✋ (user)

1. Download FaceForensics++ with the official script into `C:\dev\pulsecheck-data\ffpp`, **videos only**: `-d original` and each of `-d Deepfakes`, `-d Face2Face`, `-d FaceSwap`, `-d NeuralTextures`, with `-c c23` and again with `-c c40` and `-t videos`. That's about 12 GB in total. Do **not** use `-d all`, which also pulls DeepFakeDetection (+25 GB).
2. Download Celeb-DF v2 into `C:\dev\pulsecheck-data\celebdf` (folders `Celeb-real`, `Celeb-synthesis`, `YouTube-real`, plus `List_of_testing_videos.txt`).
3. After Task 5 lands, run the batch extraction overnight. After Task 9, run the experiments. The exact commands are in the README (Task 10).

## Global Constraints

- Everything from Plan 1's Global Constraints still holds:
  - venv `C:\dev\venvs\pulsecheck`; data dir from `PULSECHECK_DATA` = `C:\dev\pulsecheck-data`
  - no dataset media in git
  - **user's git identity only, no Co-Authored-By / Claude attribution anywhere**
  - `ruff check` clean; `pytest -q -W error` green before every commit
- New dependencies (add to `pyproject.toml` `dependencies` in the task that first needs them): `scikit-learn>=1.5`, `pandas>=2.2`, `matplotlib>=3.9`.
- Labels: **0 = real, 1 = fake** everywhere.
- **Equal-length feature windows (Plan 1 ruling R11):** all detector features are computed with `SETTINGS.feature_window_s = 4.0` and `SETTINGS.feature_step_s = 1.0` for every dataset, so DF26's 4–6 s clips (Plan 3) are comparable. Clips are capped at `SETTINGS.max_seconds = 15.0`. A clip shorter than 4 s but at least `min_seconds` gets one whole-clip window, as in `hr_series`.
- rPPG method for features: POS (`method="pos"`).
- **No leakage:**
  - FF++ groups are the official split pairs. A fake `a_b` and its originals `a` and `b` share the group `ffpp:{min(a,b)}_{max(a,b)}`.
  - Celeb-DF is test-only.
  - Any statistic fitted from data (real-face references, imputers, scalers) is fitted only on the training rows of that fit, inside the sklearn `Pipeline`.
  - Thresholds and calibration are chosen on validation data, never on test.
- Metrics: ROC-AUC (primary), balanced accuracy, precision and recall at the validation-chosen threshold (Youden's J). Report mean ± std for 5-fold grouped CV.
- One `FaceTracker` per video, never shared across videos. Worker processes call `cv2.setNumThreads(1)`.
- Results that need real data are written to `docs/results/detection.md`. Until real runs happen, that file says "pending" with the exact command. **Never write unmeasured numbers.**
- Data roots: `C:\dev\pulsecheck-data\ffpp`, `C:\dev\pulsecheck-data\celebdf`. Manifests: `data_dir()/manifests/`. Feature tables: `data_dir()/features/`. Run logs and CSVs: `data_dir()/results/`.

## Review Focus

1. **Partial or missing downloads** (some method or compression folders absent, a missing file inside a split): loaders list only files that exist, return the missing paths, and the manifest builder prints a per-kind count. Never crash and never invent rows. → Task 3 tests.
2. **Videos that fail extraction** (no face, unreadable, shorter than `min_seconds`): batch extraction logs an `error` row and continues; the feature builder skips them and reports the count. → Task 5 and Task 7 tests.
3. **Windows multiprocessing (spawn) and interruption:** workers must not depend on parent-process monkeypatching or globals. Ctrl+C must not leave partial cache files, because writes are atomic with unique temp names. Re-running resumes from the cache. → Task 1 and Task 5 tests.
4. **NaN features** (unusable region, every window invalid, control off-frame): feature functions return NaN, never raise. The model pipeline imputes. The results file reports the per-feature NaN rate. → Task 7, 8 and 9 tests.
5. **Leakage through fitted statistics:** the group-C real-face reference and all scalers and imputers are fitted only inside the pipeline on training rows (reals only for the reference). A test proves that fitting on a fold never touches held-out rows. → Task 8 and Task 9 tests.

---

### Task 1: Batch-safe trace cache v3 (landmarks stored)

**Files:** Modify `src/pulsecheck/extract.py`. Test: `tests/test_extract.py`.

**Interfaces:**
- Consumes: existing `RegionTraces`, `extract_traces`, `cached_traces`, `cache_dir()`.
- Produces:
  - `TRACES_VERSION = 3`.
  - `RegionTraces.landmarks: np.ndarray` with shape `(T, 478, 2)`, `float16` pixel coordinates, NaN rows where no face. It is saved and loaded with the other fields.
  - `cached_traces(video_path: Path, root: Path | None = None, **kwargs) -> RegionTraces`:
    - When `root` is given, the key is `v3|{video_path.relative_to(root).as_posix()}|{size}|{float(max_seconds)}`, with no mtime, so moving or copying the data folder keeps the cache valid.
    - Without `root`, the key is `v3|{resolved abs path}|{size}|{mtime_ns}|{float(max_seconds)}`.
    - `max_seconds` is normalised with `float()` (`None` stays `None`).
  - `RegionTraces.save` writes to a unique temp name `f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"` and then calls `replace`.
  - `cache_path_for(video_path: Path, root: Path | None = None, max_seconds: float | None = SETTINGS.max_seconds) -> Path` is the public form of today's `_cache_path` with the key rules above. `cached_traces` uses it, and Task 7's feature builder uses it to read the cache without extracting.
  - Cache read treats `zlib.error` like the other read errors: recompute.
  - A failed cache write (`OSError`) emits `warnings.warn(...)` and still returns the traces.
  - `extract_traces`: closing the own tracker can no longer mask an exception that is already propagating. A failing `close()` while an error propagates is suppressed and the original error is re-raised. A failing `close()` after success raises.

- [ ] **Step 1: Write failing tests**

```python
def test_landmarks_saved_and_loaded(tmp_path):            # FakeTracker video → traces.landmarks.shape == (T, 478, 2), dtype float16;
    ...                                                    # NaN rows for lost frames; save/load roundtrip equal (NaN-aware)
def test_relative_key_survives_moving_root(tmp_path, monkeypatch):
    # cached_traces(root/"a/v.avi", root=root) → copy root to root2 → cached_traces(root2/"a/v.avi", root=root2) hits cache
    # (patched pulsecheck.face.FaceTracker raises if constructed on the 2nd call)
def test_max_seconds_int_and_float_share_key(tmp_path, monkeypatch): ...   # max_seconds=1 then 1.0 → second is a hit
def test_zlib_error_entry_is_recomputed(tmp_path, monkeypatch): ...        # corrupt a byte inside a compressed member
def test_failed_cache_write_warns_and_returns(tmp_path, monkeypatch):      # monkeypatch RegionTraces.save to raise OSError
    with pytest.warns(UserWarning): tr = cached_traces(...)
    assert tr.rgb.shape[0] > 0
def test_save_uses_unique_temp_names(tmp_path, monkeypatch): ...           # two saves record distinct temp paths (spy on replace)
def test_close_failure_does_not_mask_original_error(tmp_path, monkeypatch): # fake tracker: detect never finds a face, close raises RuntimeError
    with pytest.raises(NoFaceError): extract_traces(video)                 # NoFaceError, not the close error
def test_v2_cache_file_is_a_miss(tmp_path, monkeypatch): ...               # file saved with traces_version=2 → recomputed
```

- [ ] **Step 2: Run, expect FAIL.**  - [ ] **Step 3: Implement.**  - [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit**: `feat: batch-safe trace cache v3 with landmarks and relative keys`

---

### Task 2: Configurable analysis windows and feature-window settings

**Files:** Modify `src/pulsecheck/config.py` and `src/pulsecheck/pipeline.py`. Test: `tests/test_pipeline.py`, `tests/test_config.py`.

**Interfaces:**
- Produces:
  - `Settings.feature_window_s = 4.0` and `Settings.feature_step_s = 1.0`.
  - `analyze(traces, method="pos", fs=SETTINGS.fs, window_s=SETTINGS.window_s, step_s=SETTINGS.step_s) -> Analysis`. Every `RegionResult.series` (including `_unusable`) and `face.series` use these window values. Defaults are unchanged, so all existing behaviour and tests stay the same.

- [ ] **Step 1: Write failing tests**

```python
def test_feature_window_settings():
    assert SETTINGS.feature_window_s == 4.0 and SETTINGS.feature_step_s == 1.0

def test_analyze_window_params_shape_series():
    a = analyze(make_traces(72, seconds=15), window_s=4.0, step_s=1.0)
    assert len(a.face.series.t_center) == 12 and a.face.series.t_center[0] == pytest.approx(2.0)
    for r in a.regions.values():
        assert len(r.series.t_center) == 12          # unusable regions too (same grid)

def test_analyze_short_clip_single_feature_window():  # 4.5 s clip, window_s=4.0 → 1 window centred at 2.0
    ...
```

- [ ] **Step 2–4:** RED → implement → GREEN.  - [ ] **Step 5: Commit**: `feat: configurable analysis windows; 4 s feature-window setting`

---

### Task 3: Video manifest, FaceForensics++ and Celeb-DF loaders, manifest builder

**Files:**
- Create: `src/pulsecheck/datasets/manifest.py`, `src/pulsecheck/datasets/ffpp.py`, `src/pulsecheck/datasets/celebdf.py`, `scripts/fetch_ffpp_splits.py`, `scripts/build_manifest.py`
- Test: `tests/test_datasets.py`
- Modify: `pyproject.toml` (pandas is not needed here; plain `csv`)

**Interfaces:**
- `manifest.VideoItem` (frozen dataclass):
  - `dataset: str` (`"ffpp"` or `"celebdf"`)
  - `path: Path` (absolute)
  - `rel_path: str` (POSIX, relative to the dataset root)
  - `label: int` (0 real, 1 fake)
  - `group: str`
  - `kind: str` (`"original"`, `"Deepfakes"`, `"Face2Face"`, `"FaceSwap"`, `"NeuralTextures"`, `"Celeb-real"`, `"Celeb-synthesis"`, `"YouTube-real"`)
  - `compression: str` (`"c23"`, `"c40"`, or `""`)
  - `split: str` (`"train"`, `"val"`, `"test"`)
  - Property `root -> Path`, the dataset root (`path` minus `rel_path`).
- `manifest.write_manifest(items: list[VideoItem], path: Path) -> None` and `read_manifest(path: Path) -> list[VideoItem]`: CSV with a header in field order; read raises `ValueError` on a missing column or a bad label.
- `ffpp.FFPP_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")`.
- `ffpp.load_splits(splits_dir: Path) -> dict[str, list[tuple[str, str]]]`:
  - Reads `train.json`, `val.json`, `test.json`.
  - Each file is a JSON array of 2-element arrays of 3-digit strings, e.g. `[["953","974"], ...]`.
  - Raises `ValueError` on any other shape, or if an id appears in two splits.
- `ffpp.list_ffpp(root: Path, compression: str, splits: dict, methods=FFPP_METHODS, max_pairs: dict[str, int] | None = None) -> tuple[list[VideoItem], list[Path]]` returns `(items, missing)`:
  - Originals: `root/original_sequences/youtube/{c}/videos/{id}.mp4`.
  - Fakes: `root/manipulated_sequences/{m}/{c}/videos/{a}_{b}.mp4` **and** `{b}_{a}.mp4`.
  - Group: `ffpp:{min}_{max}`.
  - `max_pairs` keeps the first N pairs of a split, in JSON order.
  - ⚠ The `videos/` sub-folder comes from the FF++ download script's conventions. Confirm it on the first real download and adjust the path template plus its test if it differs.
- `celebdf.list_celebdf_test(root: Path) -> list[VideoItem]` parses `List_of_testing_videos.txt`:
  - Each line is `"<label> <relpath>"`. **The parser verifies the label convention itself:** label `1` must go with a `*-real/` folder and `0` with `Celeb-synthesis/`. Anything else raises `ValueError` naming the line. If real files show the opposite convention, the error is loud.
  - Group is the identity token for Celeb-real/synthesis (`idN`, where synthesis files `idA_idB_xxxx` use the target `idA`), or `yt:{stem}` for YouTube-real.
  - `split = "test"`, `compression = ""`. Rows whose file is missing are skipped and counted in a returned list (same `(items, missing)` tuple shape).
- `scripts/fetch_ffpp_splits.py [--out DIR]` downloads the three JSONs from `https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/{train,val,test}.json` to `data_dir()/ffpp/splits/` (default) and validates them with `load_splits`.
- `scripts/build_manifest.py --ffpp-root P --celebdf-root P --compression c23 c40 --max-train-pairs 200 [--out-dir]`:
  - Writes `ffpp_c23.csv`, `ffpp_c40.csv` and `celebdf_test.csv` to `data_dir()/manifests/`.
  - Prints a split report: counts per split × label × kind, plus the missing count per kind.
  - Missing roots are skipped with a message.
  - `main(argv) -> int` is importable.

- [ ] **Step 1: Write failing tests** on fake trees built in `tmp_path` (empty files are fine; nothing is decoded):

```python
def test_load_splits_validates_shape_and_disjointness(tmp_path): ...       # good file ok; nested wrong / id in two splits → ValueError
def test_list_ffpp_pairs_both_directions_groups_and_missing(tmp_path):
    # splits: train [["001","002"]], test [["003","004"]]; create originals 001-004 and Deepfakes 001_002, 002_001, 003_004 (004_003 missing)
    items, missing = list_ffpp(root, "c23", splits, methods=("Deepfakes",))
    assert {i.rel_path for i in items if i.label == 1} == {...three...}
    assert [p.name for p in missing] == ["004_003.mp4"]
    assert {i.group for i in items if i.split == "test"} == {"ffpp:003_004"}
def test_list_ffpp_max_pairs(tmp_path): ...
def test_celebdf_parses_and_self_checks_labels(tmp_path):
    # lines "1 Celeb-real/id0_0000.mp4", "0 Celeb-synthesis/id0_id3_0000.mp4", "1 YouTube-real/00170.mp4"
    # → labels real/fake/real mapped to 0/1/0, groups id0, id0, yt:00170; "0 Celeb-real/..." → ValueError
def test_manifest_roundtrip_and_bad_label(tmp_path): ...
def test_build_manifest_main_writes_csvs_and_report(tmp_path, capsys): ...
```

- [ ] **Step 2–4:** RED → implement → GREEN.  - [ ] **Step 5: Commit**: `feat: video manifests with FaceForensics++ and Celeb-DF loaders`

---

### Task 4: Grouped splits and leakage guard

**Files:** Create `src/pulsecheck/splits.py`. Test: `tests/test_splits.py`. Modify `pyproject.toml` to add `scikit-learn>=1.5` and `pandas>=2.2`; reinstall the editable package.

**Interfaces:**
- `check_no_leakage(items: list[VideoItem]) -> None` raises `ValueError` listing (up to 10) groups that appear in more than one split.
- `grouped_folds(items: list[VideoItem], n_splits: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]`: index pairs into `items`, via `sklearn.model_selection.StratifiedGroupKFold(shuffle=True, random_state=seed)` on `label` with `group`.
- `split_report(items: list[VideoItem]) -> str`: a plain-text table of counts per split × label × kind.

- [ ] **Step 1: Write failing tests**

```python
def test_leakage_detected_and_named(): ...                 # same group in train and test → ValueError mentioning the group
def test_grouped_folds_never_share_groups_and_cover_all(): # 40 items, 10 groups, 1:4 real:fake
    for tr, te in grouped_folds(items): assert not ({items[i].group for i in tr} & {items[i].group for i in te})
    # union of test folds == all indices, each exactly once
def test_split_report_counts(): ...
```

- [ ] **Step 2–4:** RED → implement → GREEN.  - [ ] **Step 5: Commit**: `feat: grouped splits and leakage guard`

---

### Task 5: Resumable parallel batch extraction

**Files:** Create `scripts/extract_all.py` and `src/pulsecheck/batch.py` (the importable worker logic). Test: `tests/test_batch.py`.

**Interfaces:**
- `batch.extract_one(item: VideoItem, max_seconds: float = SETTINGS.max_seconds) -> dict[str, object]`:
  - Calls `cached_traces(item.path, root=item.root, max_seconds=max_seconds)`.
  - Returns `{"rel_path", "dataset", "status": "ok"|"cached"|"error", "seconds", "frames", "message"}`.
  - Catches `NoFaceError`, `ValueError`, `FileNotFoundError`, `OSError` and `cv2.error` into `status="error"`.
  - `"cached"` means the cache file already existed and loaded, so no tracker was built.
  - Never raises for a per-video failure.
- `batch.init_worker() -> None`: `cv2.setNumThreads(1)`. It is the Pool initializer.
- `batch.run_batch(items: list[VideoItem], workers: int, max_seconds: float, log_path: Path, progress_every: int = 25) -> dict[str, float]`:
  - `workers == 1` runs in-process (no Pool).
  - Otherwise uses `multiprocessing.get_context("spawn").Pool(workers, initializer=init_worker)` with `imap_unordered`.
  - Appends one CSV row per finished video to `log_path` immediately (header written if the file is new).
  - Prints progress lines (`done/total`, ok/cached/error counts, elapsed, ETA).
  - On `KeyboardInterrupt`: terminates the pool, prints a resume hint, and re-raises.
  - Returns a summary with keys `total, ok, cached, error, wall_s, videos_per_hour, frames_per_s`.
- `scripts/extract_all.py --manifest CSV [--manifest CSV ...] [--workers N] [--max-seconds 15] [--limit N] [--log PATH]`:
  - Default `--workers` is `max(1, (os.cpu_count() or 2) // 2 - 1)`.
  - Default log: `data_dir()/results/extract_log.csv`.
  - Prints the summary, which is the throughput measurement the spec §7 needs.
  - Has `main(argv) -> int` and an `if __name__ == "__main__"` guard (required for spawn).

- [ ] **Step 1: Write failing tests** (synthetic `.avi` files from `conftest.write_video`; FakeTracker via monkeypatched `pulsecheck.face.FaceTracker`, in-process only):

```python
def test_extract_one_ok_then_cached(tmp_path, monkeypatch): ...      # 1st "ok", 2nd "cached" (FaceTracker patched to raise on 2nd)
def test_extract_one_error_row_not_raise(tmp_path, monkeypatch): ...  # face never found → status "error", message mentions face
def test_run_batch_in_process_logs_every_row(tmp_path, monkeypatch):
    s = run_batch(items, workers=1, max_seconds=3, log_path=log)
    assert s["total"] == 3 and s["ok"] + s["error"] == 3 and len(read_log(log)) == 3
def test_run_batch_spawn_pool_resumes_from_cache(tmp_path, monkeypatch):
    # populate the cache with workers=1 (patched tracker), then run workers=2 with a real Pool:
    # children never build a tracker because every item is cached → all rows "cached"
def test_extract_all_main_parses_args_and_limit(tmp_path, monkeypatch): ...
```

- [ ] **Step 2–4:** RED → implement → GREEN.  - [ ] **Step 5: Commit**: `feat: resumable parallel batch extraction with run log`

---

### Task 6: Cross-region spectral helpers

**Files:** Modify `src/pulsecheck/dsp.py`. Test: `tests/test_dsp.py`.

**Interfaces** (all reuse `power_spectrum`'s Hann window and `nfft` rule, and the SNR half-width constants; no restated numbers):
- `cross_spectrum(x: np.ndarray, y: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]` returns `(freqs, Sxy)`, with `Sxy = X · conj(Y)` from the Hann-windowed, zero-padded rFFTs.
- `phase_lag_s(x: np.ndarray, y: np.ndarray, fs: float, f0: float) -> float`: the time by which **y lags x** at `f0`, in seconds. It is `angle(Sxy at the bin nearest f0) / (2π f0)`, wrapped to `(-1/(2 f0), 1/(2 f0)]`. It is NaN if `f0` is NaN or either input is flat.
- `band_coherence(xs: list[np.ndarray], ys: list[np.ndarray], fs: float, f0: float) -> float`: magnitude-squared coherence at `f0` averaged over windows, `|Σ_w Σ_b Sxy|² / (Σ_w Σ_b Sxx · Σ_w Σ_b Syy)`. The sum over `b` covers bins within `f0 ± SNR_FUNDAMENTAL_HALFWIDTH_HZ`. The result is in [0, 1]. It is NaN for empty input or NaN `f0`.
- `harmonic_ratio(x: np.ndarray, fs: float, f0: float) -> float`: power in `2f0 ± SNR_HARMONIC_HALFWIDTH_HZ` divided by power in `f0 ± SNR_FUNDAMENTAL_HALFWIDTH_HZ`. It is NaN for flat input or NaN `f0`.

- [ ] **Step 1: Write failing tests**

```python
def test_phase_lag_recovers_known_delay():
    fs, f0 = 30.0, 1.2; t = np.arange(0, 8, 1 / fs)
    x = np.sin(2 * np.pi * f0 * t); y = np.sin(2 * np.pi * f0 * (t - 0.05))   # y lags x by 50 ms
    assert phase_lag_s(x, y, fs, f0) == pytest.approx(0.05, abs=0.005)
    assert phase_lag_s(y, x, fs, f0) == pytest.approx(-0.05, abs=0.005)
def test_band_coherence_high_for_shared_low_for_independent():   # 6 windows of 4 s; shared pulse + small noise → > 0.9; independent noise → < 0.3
    ...
def test_harmonic_ratio_pure_vs_harmonic():                       # pure sine < 0.01; with 0.3-amplitude 2nd harmonic ≈ 0.09 (abs 0.03)
    ...
def test_helpers_nan_safe():                                      # NaN f0 / flat input → NaN, no warnings
    ...
```

- [ ] **Step 2–4:** RED → implement → GREEN.  - [ ] **Step 5: Commit**: `feat: cross-spectrum, phase lag, band coherence and harmonic ratio`

---

### Task 7: Per-video features, groups A (pulse presence) and D (quality), plus the feature-table builder

**Files:** Create `src/pulsecheck/features.py`, `scripts/build_features.py`. Modify `tests/synth.py` (add `synth_region_traces`). Test: `tests/test_features.py`.

**Interfaces:**
- `tests/synth.py`: `synth_region_traces(kind: str, seconds: float = 15.0, fs: float = 30.0, seed: int = 0, bpm: float = 72.0) -> RegionTraces`. It builds a `RegionTraces` for `ALL_REGIONS` with no face loss, landmarks all NaN-free placeholders (zeros, float16), `face_width_px=150` and `motion=0.01`. Kinds:
  - `"real"`: one shared pulse; per-region delays `REAL_LAGS_S` (forehead 0, cheeks +0.03, nose +0.02, chin +0.05 s); per-region amplitudes `REAL_AMPS` (forehead 1.0, cheeks 1.3, nose 0.8, chin 0.7, relative); control region with no pulse.
  - `"inherited"`: the same shared pulse in every skin region with **zero** lag and **equal** amplitude, i.e. a pulse copied from a driver video and pasted uniformly.
  - `"noise"`: independent band-limited noise per region, no common pulse.
  - All kinds get the same skin DC values, sensor noise σ and seed-controlled jitter. Exact constants are defined in `synth.py` and documented there.
- `features.FEATURE_GROUPS: dict[str, tuple[str, ...]]` with keys `"A"`, `"B"`, `"C"`, `"D"`. Every feature name is prefixed with its group letter (`a_…`, `b_…`, `c_…`, `d_…`).
- `features.group_a(analysis: Analysis) -> dict[str, float]`:
  - `a_snr_face`: median over valid windows of `face.series.snr_db`.
  - `a_snr_regions_median` and `a_snr_regions_max`: across skin regions, each region's median window SNR.
  - `a_hr_std_face`: std of valid face window bpm.
  - `a_hr_in_rest_range`: fraction of valid face windows with bpm in [45, 150].
  - `a_valid_frac`: `analysis.valid.mean()`.
- `features.group_d(traces: RegionTraces, analysis: Analysis) -> dict[str, float]`:
  - `d_duration_s`, `d_face_found_ratio`, `d_face_width_px`, `d_motion` (from `quality_summary`).
  - `d_brightness`: median face-signal G.
  - `d_n_windows`: number of face windows.
- `features.video_features(traces: RegionTraces, groups: str = "ABCD", method: str = "pos") -> dict[str, float]`:
  - Runs `analyze(traces, method, window_s=SETTINGS.feature_window_s, step_s=SETTINGS.feature_step_s)`.
  - Returns the requested groups merged. Groups B and C are added in Task 8; until then, requesting them raises `NotImplementedError`.
  - Missing values are NaN; it never raises for an unusable region.
  - If `analyze` raises `ValueError` (face unusable or clip too short), the error propagates. The builder records it.
- `scripts/build_features.py --manifest CSV [...] [--out CSV] [--groups ABCD]`:
  - Reads **cached traces only**: for each item, `extract.cache_path_for(item.path, item.root, max_seconds)` (Task 1) and `RegionTraces.load` if the file exists. It never extracts.
  - Uncached or failed videos are skipped and counted.
  - Writes one row per video: the manifest fields + features. Default output: `data_dir()/features/<manifest stem>.csv`.
  - Prints the skipped counts and the per-feature NaN rate.

- [ ] **Step 1: Write failing tests**

```python
def test_group_a_real_vs_noise():                    # 10 seeds each: median a_snr_face(real) > median(noise) + 3 dB
def test_group_a_cannot_separate_inherited_from_real():  # documents the limitation: |median(real) - median(inherited)| < 1.5 dB
def test_features_nan_safe_when_region_unusable():   # control all-NaN, one cheek all-NaN → finite group A values, no warnings
def test_group_d_values():                           # duration ≈ 15.0, face_found_ratio 1.0, face_width 150, n_windows 12
def test_build_features_skips_uncached_and_reports(tmp_path, capsys, monkeypatch): ...
```

- [ ] **Step 2–4:** RED → implement → GREEN.  - [ ] **Step 5: Commit**: `feat: per-video baseline features (pulse presence, quality) and feature tables`

---

### Task 8: Consistency features, groups B and C, plus the leak-free real-face reference

**Files:** Modify `src/pulsecheck/features.py`. Create `src/pulsecheck/reference.py`. Test: `tests/test_features.py`, `tests/test_reference.py`.

**Interfaces:**
- Every per-window computation uses the face's window bpm as `f0` (NaN windows are skipped), on `analysis.regions[*].pulse` sliced to the same 4 s windows as the series, with `REF_REGION = "forehead_center"`.
- `features.group_b(analysis) -> dict[str, float]`:
  - `b_corr_median`, `b_corr_min`: median over windows of the median and min pairwise Pearson correlation between skin-region pulses.
  - `b_hr_spread`: median over windows of the std of the region bpm values.
  - `b_coherence`: mean over skin-region pairs of `band_coherence` across windows at the window `f0`.
- `features.group_c(analysis) -> dict[str, float]`:
  - `c_lag_<region>` for the 6 non-reference skin regions: median over windows of `phase_lag_s(ref, region, f0)`.
  - `c_lag_stability`: mean over regions of the std over windows of those lags.
  - `c_amp_<region>` for all 7 skin regions: `sqrt` of the region pulse's power within `f0 ± SNR_FUNDAMENTAL_HALFWIDTH_HZ`, divided by the sum of that quantity over the 7 skin regions. Median over windows. `analysis.regions[r].pulse` comes from `pulse_from_rgb` on the region's own RGB, so it is already relative to that region's mean colour, and the map compares pulse strength, not brightness.
  - `c_harmonic_face`: `harmonic_ratio` of the face pulse, median over windows. `c_harmonic_spread`: std across skin regions of their harmonic ratios.
  - `c_control_ratio`: control-region power at `f0` divided by face power at `f0`, median over windows; NaN if control is unusable.
  - `c_corr_stability`: std over windows of the per-window median pairwise correlation.
  - **Rolling-shutter note:** this goes in the docstring and in METHOD. Lags partly reflect row readout time between vertical positions; the learned real-face reference absorbs any camera-consistent offset.
- `reference.RealFaceReference(BaseEstimator, TransformerMixin)` with params `amp_prefix="c_amp_"` and `lag_prefix="c_lag_"`:
  - `fit(X: pd.DataFrame, y)`: on rows with `y == 0` only, fits `sklearn.covariance.LedoitWolf` separately for the amplitude columns and the lag columns (rows with NaN in that block are dropped for that block; if fewer than 5 rows remain, it stores NaN stats).
  - `transform(X)`: returns X with **two added columns**, `c_amp_mahal` and `c_lag_mahal` (Mahalanobis distance to the fitted real-face mean; NaN for rows with NaN in the block or when stats are NaN).
  - `get_feature_names_out()` is implemented.
  - Requires a DataFrame and keeps the column order.

- [ ] **Step 1: Write failing tests**

```python
def test_group_b_separates_noise_from_real():      # median b_coherence(real) > 0.8 > 0.4 > median(noise)
def test_group_c_lags_recover_synthetic_delays():   # real: c_lag_chin ≈ +0.05 s (abs 0.015), c_lag_nose ≈ +0.02 s; inherited: |lags| < 0.01
def test_group_c_amplitude_map_differs():           # real c_amp_cheek_left > c_amp_chin; inherited: all ≈ 1/7 (abs 0.02)
def test_reference_fits_on_reals_only(): # y has reals and fakes; fakes set to wild values → fitted mean unchanged vs fit on reals alone
def test_reference_distance_separates_inherited():  # fit on 30 synthetic reals; mahal(inherited) median > 3 × mahal(held-out real) median
def test_reference_nan_rows_and_too_few_rows():     # NaN block → NaN output; <5 reals → all-NaN columns, no exception
def test_showcase_auc_inherited_vs_real():
    # 30 real + 30 inherited synthetic videos, 5-fold CV with logistic regression in a Pipeline:
    # features A only → AUC < 0.7 ; A+B+C (with RealFaceReference in the pipeline) → AUC > 0.9
```

- [ ] **Step 2–4:** RED → implement → GREEN.  - [ ] **Step 5: Commit**: `feat: spatial-agreement and physiological-consistency features with leak-free real-face reference`

---

### Task 9: Models, evaluation and the experiments script

**Files:** Create `src/pulsecheck/model.py`, `scripts/run_experiments.py`. Test: `tests/test_model.py`, `tests/test_experiments.py`. Modify `pyproject.toml` to add `matplotlib>=3.9`.

**Interfaces:**
- `model.feature_columns(df: pd.DataFrame, groups: str) -> list[str]`: columns whose prefix letter is in `groups`. It never includes manifest fields.
- `model.make_model(kind: str, columns: list[str]) -> sklearn.pipeline.Pipeline`. Steps, in order:
  - `RealFaceReference` (only if any `c_amp_`/`c_lag_` column is present)
  - `SimpleImputer(strategy="median", keep_empty_features=True)` on numeric columns (via `ColumnTransformer` passthrough → DataFrame output with `set_output(transform="pandas")`)
  - `StandardScaler` (logreg only)
  - the classifier: `LogisticRegression(class_weight="balanced", max_iter=2000)` for `"logreg"`, `HistGradientBoostingClassifier(class_weight="balanced", random_state=0)` for `"hgb"`
  - Unknown kind raises `ValueError`.
- `model.youden_threshold(y_true, scores) -> float`.
- `model.metrics(y_true, scores, threshold) -> dict[str, float]` with keys `auc, balanced_acc, precision, recall, n_real, n_fake, tn, fp, fn, tp` (the confusion matrix at the threshold; the results markdown prints it for E2's best model).
- `model.grouped_cv(df, columns, kind, n_splits=5, seed=0) -> dict[str, float]`: `auc_mean`, `auc_std` over `StratifiedGroupKFold` folds on `df.group`.
- `model.fit_and_test(train_df, val_df, test_df, columns, kind) -> dict`:
  - Fits on train.
  - Picks the threshold on val (Youden).
  - Returns test `metrics`, per-`kind` test AUC (each fake kind vs all test reals), and the fitted model.
  - It calls `check_no_leakage` on the union of train, val and test items first.
- `model.permutation_importances(fitted, test_df, columns, n_repeats=10, seed=0) -> pd.DataFrame`: mean/std drop in AUC, sorted.
- `scripts/run_experiments.py --features-dir DIR [--kind logreg hgb] [--md-out docs/results/detection.md] [--fig-dir docs/figures] [--csv-out DIR]` runs every experiment whose feature files exist, and skips the others with a "not run: missing <file>" line in the markdown:
  - **E2:** FF++ c23 official train → val → test for groups `A`, `AB`, `ABC`, `ABCD`; AUC per manipulation; plus 5-fold grouped CV on train.
  - **E3:** train on c23, test on c40 test.
  - **E4:** train on FF++ c23, test on the Celeb-DF test list.
  - **E6:** permutation importance (top 15) for the best `ABCD` model, plus a group ablation (drop each group).
  - Writes ROC figures `roc_e2.png`, `roc_e4.png`.
  - The markdown header records the run date, the exact command, row counts and the per-feature NaN rate.
  - `main(argv) -> int`.

- [ ] **Step 1: Write failing tests**

```python
def test_make_model_kinds_and_unknown(): ...
def test_pipeline_fit_does_not_see_test_rows():   # spy: RealFaceReference.fit receives only train rows (and only y==0 rows used)
def test_youden_and_metrics_exact():               # small hand-checked arrays
def test_grouped_cv_reports_mean_std_on_synthetic(): ...
def test_fit_and_test_rejects_leakage(): ...       # same group in train and test → ValueError
def test_run_experiments_end_to_end_synthetic(tmp_path):
    # write ffpp_c23.csv features from synthetic traces (real vs inherited vs noise, with fake kinds),
    # celebdf_test.csv; no c40 file → E3 "not run"; markdown contains E2 table rows for A and ABC with AUC numbers that match
    # model.fit_and_test recomputed in-test; figures written
```

- [ ] **Step 2–4:** RED → implement → GREEN.  - [ ] **Step 5: Commit**: `feat: detector models, grouped evaluation and experiments script`

---

### Task 10: Documentation for Plan 2

**Files:** Modify `README.md`, `docs/METHOD.md`, `docs/INTERVIEW_NOTES.md`, `docs/plan.md` (only the small fixes listed below). Create `docs/results/detection.md` (pending placeholder). Optionally add the synthetic showcase figure `docs/figures/consistency_synthetic.png` via `scripts/make_figures.py`.

- [ ] **Step 1: README:** status (Phases 3–5 code done, real results pending); the user's data steps as copy-paste commands in order (download → `fetch_ffpp_splits.py` → `build_manifest.py` → `extract_all.py` overnight → `build_features.py` → `run_experiments.py`), with the measured or estimated time per step; and the layout update.
- [ ] **Step 2: METHOD:**
  - Datasets and splits (the leakage rules).
  - Why 4 s feature windows.
  - Each feature group, with formulas: phase lag from the cross-spectrum, band coherence, the amplitude map, Mahalanobis with Ledoit-Wolf shrinkage, and why the reference is fitted inside the pipeline.
  - The rolling-shutter caveat.
  - Models and metrics.
  - The synthetic showcase (A-only vs A+B+C on inherited-pulse fakes, labelled synthetic).
- [ ] **Step 3: INTERVIEW_NOTES:** Q&As covering:
  - Why group by FF++ pair?
  - Why Celeb-DF test-only?
  - Why 4 s windows when 8 s gives better heart rates?
  - Why phase lags?
  - What is coherence?
  - Why Mahalanobis + Ledoit-Wolf?
  - Why fit the reference inside the pipeline?
  - Why logistic regression and gradient boosting rather than a neural net?
  - Why AUC?
  - What does the synthetic showcase prove and not prove?
  - Why spawn-safe workers and one tracker per video?
- [ ] **Step 4: `docs/plan.md`:**
  - §4 and §6.1 say Welch; note that the implementation uses a Hann periodogram (as METHOD explains).
  - §6.2 "whole clip if shorter" now applies to 4 s feature windows.
  - §7 subsets: state the FF++ train-pair default (`--max-train-pairs 200`) and that the official val/test splits are used in full.
- [ ] **Step 5: Verify** every documented command with `--help` (and the synthetic end-to-end where possible), then run `ruff` and `pytest`. Commit: `docs: detector pipeline, features and evaluation protocol`.

---

## Done when (Plan 2)

- `pytest -q -W error` green and `ruff check .` clean. The branch has a PR to `main` and the user is the only author.
- On synthetic data the showcase holds: group A alone can't separate inherited-pulse fakes from reals (AUC < 0.7); A+B+C can (AUC > 0.9). This is labelled synthetic everywhere.
- `docs/results/detection.md` is "pending" with exact commands. ✋ Real numbers arrive when the user has downloaded FF++/Celeb-DF and run extraction and experiments.
