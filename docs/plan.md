# PulseCheck — Heartbeat-Based Deepfake Detection

*Design and phase-by-phase build plan. Draft for approval, 2026-10-26.*
*Working name "PulseCheck" is a placeholder (my call) — rename freely.*

---

## 1. Intent

A portfolio project that shows an interviewer you can take a published idea
(Intel's FakeCatcher, Ciftci et al. 2019/2020), rebuild it from first
principles, test it honestly against today's AI video, and improve on it.

**The core idea.** Each heartbeat pushes blood into the face and changes skin
color very slightly. A camera can pick this up (remote photoplethysmography,
"rPPG"). In a real video, every skin patch is driven by the same heart, so
the patches agree. In a fake, the pulse may be missing, may be noise, or may
disagree between patches.

**The twist (our addition).** Recent research shows the simple version of
this idea is breaking:

- Seibold et al., *"High-quality deepfakes have a heart!"* (Frontiers in
  Imaging, April 2025): modern face-swaps *inherit* a believable pulse from
  the driving video, so "is there a pulse?" is no longer enough. The same
  paper suggests the **spatial pattern of blood flow** across the face is
  still unrealistic in fakes, and that this is a promising direction.
- FakeCatcher's own paper reports that hand-built features with an SVM
  reached only ~61% on its hardest dataset. The 96% headline came from a CNN
  on "PPG maps", tested by Intel itself.

So our project asks: **"Is this pulse *physiologically consistent* across
the face — right timing, right amplitude pattern, stable over time?"**, and
tests that on both classic deepfakes and 2025–26 AI video.

**Why it matters (your interview "why").** AI-generated video on social media
is now good enough to fool most people. Viewers in the DF26 study scored
roughly coin-flip on Veo 3.1 clips. Detectors that rely on biology rather
than visual artifacts are one of the few directions that don't depend on a
particular generator's quirks.

## 2. Goals, non-goals, anti-goals

**Goals**
1. A working rPPG engine you can validate: it reads your heart rate from a
   webcam, checked against UBFC-rPPG's pulse-oximeter readings and your own
   manual pulse count.
2. A detector with a measured **before/after**: classic "pulse present"
   features (baseline) vs. baseline + our consistency features.
3. Honest evaluation on old deepfakes (FaceForensics++, Celeb-DF), under
   heavier compression, across datasets, and on modern AI video (DF26:
   Veo 3.1, Kling 3.0, Grok Imagine, Wan, HunyuanVideo, LTX).
4. A three-way verdict: **likely real / likely AI / can't tell**. "Can't tell"
   comes from a quality check, not just a low score.
5. A clean demo: live webcam heart-rate window plus an upload-a-video web
   page showing regions, waveforms, an agreement map, and the verdict with
   reasons.
6. A README an interviewer gets in 2 minutes, plus an interview-notes
   document explaining every design choice.

**Non-goals**
- A consumer product, browser extension, accounts, or hosting at scale.
- A deep-learning project. The small CNN is a stretch goal only.
- Detecting audio deepfakes, or AI video with no visible face.

**Anti-goals (failure even if it "works")**
- Claiming it "detects any AI video" or quoting numbers we didn't measure.
- Data leakage that inflates results (see §6.3).
- Code you can't explain line by line.
- Redistributing dataset videos (licenses forbid it).

## 3. Constraints

- **Your machine:** Intel i5-1335U (10 cores), 16 GB RAM, **no NVIDIA GPU**,
  ~123 GB free disk, Windows 11, Python 3.13 and 3.14 installed. Everything
  must run on CPU. That's fine, because rPPG is lightweight. The bottleneck
  is face tracking across thousands of videos (see §7, compute budget).
- **Dataset access needs *you*.** Each dataset needs a request form or
  license acceptance in your name. I can't fill those in for you.
- **Licenses:** all datasets are research/education use only. No dataset
  footage in the public repo or README GIF. DF26's commercial-generator
  subset is evaluation-only.
- **Folder:** code lives in `Desktop\Deepfake project`, which is inside
  **OneDrive**. Datasets (tens of GB) and the Python virtual env (thousands
  of files) must **not** sync to OneDrive, so they live outside it:
  - data and cache: `C:\dev\pulsecheck-data` (set via the `PULSECHECK_DATA`
    environment variable or `configs/local.yaml`)
  - virtual env: `C:\dev\venvs\pulsecheck`
- **No heart-rate device.** UBFC-rPPG's pulse-oximeter readings are the main
  ground truth. For webcam clips, the reference is a 60-second manual pulse
  count at the wrist or neck, taken during the recording (rough, ±3–5 bpm),
  optionally cross-checked with a free phone app that reads the pulse
  through the phone camera with a finger on the lens.
- **No deadline.** Optimize for the best outcome, not speed: full dataset
  runs rather than subsets where compute allows, and the stretch CNN
  comparison is in scope.
- **Authorship:** commits use your git identity only. No Claude co-author
  lines anywhere. Public repo under `github.com/JaskaranVaid`.

## 4. Decisions (and how sure I am)

| Area | Choice | Why | Confidence |
|---|---|---|---|
| Language | Python 3.13 in a virtual env | Every rPPG/deepfake codebase is Python; 3.13 already installed | High |
| Face tracking | **MediaPipe Face Landmarker** (Tasks API, `mediapipe` 1.1.x, 478 landmarks, VIDEO mode) | Old Face Mesh API is retired and removed in recent releases | High (verify it installs on 3.13 in Phase 0; fallback: Python 3.12 venv) |
| Video I/O | OpenCV (`opencv-python`) | Standard, fast, webcam support | High |
| rPPG methods | Implement **GREEN, CHROM, POS** ourselves (~50 lines each) | You can explain them; comparing them is a results row | High |
| Reference check | **pyVHR** / **rPPG-Toolbox** used only to sanity-check our numbers | Proves our implementation is right | Medium (optional) |
| Signal processing | NumPy + SciPy (Butterworth bandpass, Welch spectrum, coherence) | Standard | High |
| Classifier | scikit-learn: logistic regression + HistGradientBoosting, with calibration | Explainable, CPU-fast, small data | High |
| Web demo | **Streamlit** (upload a video) | Fast to build, looks clean | High |
| Live demo | Plain **OpenCV window** for webcam | streamlit-webrtc is fiddly; this is reliable | High |
| Plots | Matplotlib (+ Plotly in Streamlit) | Standard | High |
| Tests | pytest, with synthetic "fake pulse" videos | Proves the math without datasets | High |
| Packaging | `pyproject.toml`, `pip install -e .`, a `pulsecheck` command-line tool | Clean, conventional | High |

## 5. Data

| Dataset | Role | Access | Notes |
|---|---|---|---|
| **UBFC-rPPG** (Dataset 2: 42 videos, ~1 min, 30 fps, with pulse-oximeter ground truth) | Prove our heart-rate engine is accurate | Download from the authors' Google Site (may need a request) | Required for Phase 2 |
| **Your own webcam clips + manual pulse count** | Second heart-rate check; real-face demo footage you own | You record ~5 short clips | Required: your clips are the only footage we can show publicly |
| **FaceForensics++** (1,000 real + 4,000 manipulated; c23 and c40 compression) | Main training/testing set; compression test | Google form → download script | Download **videos only, c23 + c40**, no raw. Start the request now. |
| **Celeb-DF v2** (590 real + 5,639 fake) | Cross-dataset test (train on FF++, test here) | Google form → link | Start the request now |
| **DF26** (271 real + 2,420 fake talking heads, 1280×720, ~5 s each; Veo 3.1, Kling 3.0, Grok Imagine, Wan 2.6, HunyuanVideo, LTX, Wan 2.2) | Modern AI test, broken down by generator | Hugging Face, gated, auto-approved after accepting the license | Commercial subset is **evaluation-only**. 3.6 GB. |
| ~~CCDF~~ | — | — | **Dropped:** it's surveillance footage (crimes/accidents), so faces are too small for rPPG |

**Caveat on DF26:** clips are only 4–6 seconds long. rPPG works best on
10 seconds or more, so short clips give a less precise heart rate. We'll
measure how much that hurts and report it, since it's itself a finding.

## 6. Technical design

### 6.1 Pipeline

```
video ─► face landmarks (MediaPipe, every frame)
      ─► skin regions (7 polygons from landmarks; eyes/mouth/brows excluded)
      ─► per-frame mean R,G,B per region  (+ 1 non-skin "control" region)
      ─► rPPG signal per region (POS by default; GREEN/CHROM selectable)
      ─► detrend + bandpass 0.7–3.3 Hz  (= 42–200 bpm)
      ─► per-region: spectrum, heart rate, signal-to-noise ratio
      ─► features (§6.2) ─► calibrated classifier ─► quality gate
      ─► verdict: likely real / likely AI / can't tell  (+ reasons)
```

Signals per video are cached to disk (`.npz`), so face tracking, the slow
step, runs once per video. Datasets mix 24, 25 and 30 fps, so every signal
is resampled to a fixed 30 Hz using frame timestamps before any analysis.

**Regions (7 skin + 1 control):** forehead-left, forehead-center,
forehead-right, left cheek, right cheek, nose bridge, chin. Control region:
a background or hair patch beside the head. A real pulse should **not** show
up there. If it does, that's lighting flicker, not a heartbeat.

### 6.2 Features (each video → one row of numbers)

Computed over sliding windows (default 8 s, 1 s step; whole clip if
shorter), then summarized per video (median, spread).

**A. Pulse presence — the classic baseline (FakeCatcher-style)**
- Signal-to-noise ratio of the pulse per region (power at the heart rate
  plus its first harmonic vs. everything else)
- Peak sharpness in the spectrum
- Is the heart rate in a plausible range, and how stable is it over time?

**B. Spatial agreement**
- Correlation between every pair of region signals
- Heart-rate spread across regions (real: nearly identical)
- Spectral coherence between regions at the heart-rate frequency

**C. Physiological consistency — our addition** (motivated by Seibold 2025)
- **Timing pattern:** phase lag between regions at the heart-rate frequency.
  Blood reaches facial regions with small, structured delays. We test
  whether this lag pattern is consistent over time and matches the pattern
  learned from real faces.
- **Amplitude map plausibility:** relative pulse strength across forehead,
  cheeks, nose and chin. We learn the "normal" pattern from real training
  videos and score each video by its distance from it (Mahalanobis
  distance).
- **Waveform shape:** ratio of the 2nd harmonic to the fundamental (a real
  pulse wave isn't a pure sine), and whether that shape matches across
  regions.
- **Control-region check:** pulse strength in the background vs. the skin.
- **Temporal stability:** how much all of the above jumps between windows.

**D. Quality (not used to classify; drives "can't tell")**
- Face size in pixels, head motion, frames where the face was lost,
  brightness, clip length, and an estimate of compression strength.

### 6.3 Evaluation rules (where interviewers poke)

- **No leakage:** FaceForensics++ fakes are made *from* its real videos, so
  splits group each original video with all its fakes. Celeb-DF splits by
  person (we use its official 518-video test list as the test set). DF26 is
  **test-only** by default: the model never trains on it, which makes it a
  true "unseen generators" test. Optional extra experiment: train on its
  open-weight subset (grouped by source clip) and test on the commercial
  subset.
- **Metrics:** ROC-AUC (main), plus balanced accuracy, precision/recall at
  the chosen threshold, and heart-rate error (MAE/RMSE/Pearson) for the
  rPPG engine.
- **Balanced classes:** FF++ is 1:4 real:fake, DF26 about 1:9. Report AUC
  and balanced accuracy, and use class weights.
- **Thresholds and calibration** are chosen on validation data only, never
  on test data.
- Five-fold grouped cross-validation, reported as mean ± std.

### 6.4 Planned results tables (filled in with real numbers)

1. **Heart-rate accuracy** on UBFC + your webcam: GREEN vs CHROM vs POS.
2. **Detection on FF++ c23:** A only → A+B → A+B+C; AUC per manipulation type.
3. **Compression:** train c23, test c40.
4. **Cross-dataset:** train FF++, test Celeb-DF.
5. **Modern AI (DF26):** AUC per generator, plus "can't tell" rate.
6. **What mattered:** feature importance (permutation) and an ablation.
7. *(Stretch)* Small CNN on PPG maps vs our features.

## 7. Compute budget

Face tracking on your CPU is the slow part. My estimate: roughly 30–60
frames/second at 720p, to be measured in Phase 1. To keep this manageable:

- Process only the **first 15 seconds** of each FF++/Celeb-DF video. That's
  enough for rPPG.
- Start with a **subset**: e.g. 400 FF++ originals + their 4 fakes each at
  c23 (2,000 videos), plus the c40 versions of the test split, all of DF26,
  and the 518-video Celeb-DF test list. Scale up if time allows.
- Batch runs are **resumable**, run in parallel worker processes, and run in
  the background overnight. Expect several hours of compute per dataset.

## 8. Phase-by-phase build plan

Each phase ends with something you can run and review. ✋ = needs you.

### Phase 0 — Setup and data requests
- Me: `git init` in `Desktop\Deepfake project`, create the public GitHub
  repo, set up the data folder and virtual env outside OneDrive.
- ✋ Submit access requests: FaceForensics++ form, Celeb-DF form, UBFC
  download, DF26 license on Hugging Face. **Do this first. Approvals can take
  days and nothing in Phases 1–2 waits on them.**
- Me: project skeleton (`pyproject.toml`, `src/pulsecheck/`, `tests/`,
  `scripts/`, `configs/`), virtual env, pinned dependencies, linting (ruff),
  pytest, `.gitignore` that keeps data out of git.
- Me: confirm MediaPipe Face Landmarker installs and runs on Python 3.13 on
  your machine (fallback: 3.12).
- **Done when:** `pytest` passes on an empty test and a smoke script detects
  your face from the webcam.

### Phase 1 — Face tracking and signal extraction
- `face.py`: Face Landmarker wrapper (VIDEO mode, model auto-download).
- `regions.py`: 7 skin regions + control region from landmark indices; skin
  masks.
- `extract.py`: video → per-frame mean RGB per region, with face-lost
  handling, timestamps and quality stats → cached `.npz`.
- Debug tool: writes an annotated video with the regions drawn on.
- Tests on synthetic video (a face image with known color modulation).
- **Done when:** the debug video shows regions sticking to your face while
  you move, and the speed number for §7 is measured.

### Phase 2 — The rPPG engine (first milestone)
- `rppg.py`: GREEN, CHROM, POS.
- `signal.py`: detrending, Butterworth bandpass, Welch spectrum, heart-rate
  peak, SNR, phase/coherence helpers.
- `pulsecheck live`: OpenCV window with your face, regions, live waveform
  and BPM.
- `scripts/eval_hr.py`: UBFC evaluation → results table 1.
- ✋ You record 3–5 clips (60 s each) and count your pulse during each one.
  I compare.
- Tests: synthetic signal at known BPM recovered within ±2 bpm; filter
  passes 72 bpm and rejects 10 bpm and 300 bpm.
- **Done when:** POS on UBFC gets a heart-rate error in the range published
  papers report for POS (a few bpm), and the live window matches your manual
  pulse count within a few bpm.

### Phase 3 — Dataset pipeline
- `datasets/`: loaders for FF++, Celeb-DF, DF26, UBFC → one manifest CSV
  (path, label, group id, generator, compression).
- `scripts/extract_all.py`: parallel, resumable batch extraction with a
  progress bar.
- `splits.py`: grouped train/val/test splits (§6.3) plus a test that fails
  if any group appears in two splits.
- **Done when:** signals are cached for the agreed subsets and a split report
  shows no leakage.

### Phase 4 — Baseline detector (second milestone)
- `features.py` groups A and D. `train.py`, `evaluate.py` with grouped
  cross-validation, ROC curves, confusion matrices.
- **Done when:** results table 2 row "A only" is filled in on FF++ c23. This
  is our honest re-creation of the classic method.

### Phase 5 — Consistency features (the contribution)
- Feature groups B and C, with real-face reference statistics learned from
  the training split only.
- Ablations and permutation importance → results tables 2, 3, 4, 6.
- **Done when:** the before/after table exists, whichever way it comes out.
  If the new features don't help, we say so and show why. That's still a
  result.

### Phase 6 — Modern AI test and "where it breaks"
- Run on DF26. Per-generator results (table 5), including the effect of
  short clips.
- `scripts/case_studies.py`: plots for interesting wins and failures (e.g. a
  fake with an inherited pulse but a wrong timing pattern; a real video that
  compression wipes out).
- **Done when:** a written "Limitations and failure modes" section backed by
  plots.

### Phase 7 — Verdict and confidence
- Calibrated probability (`CalibratedClassifierCV`) + quality gate →
  likely real / likely AI / can't tell, with plain-English reasons (e.g.
  "pulse found in 6/7 regions; timing pattern unusual").
- `pulsecheck analyze video.mp4` → JSON + summary.
- **Done when:** the command gives sensible verdicts on held-out videos and
  says "can't tell" on deliberately degraded ones.

### Phase 8 — Demo app
- Streamlit app: upload → progress → frame with regions → stacked waveforms
  per region → region-agreement heatmap → spectrum → verdict card with
  reasons and quality warnings. Includes an "About the method" tab.
- Polish the live webcam window, e.g. a side-by-side "your pulse" panel.
- ✋ Demo footage: your own real clips + an AI clip you generate yourself
  (see open questions).
- **Done when:** a fresh clone runs the app with one command.

### Phase 9 — Presentation
- README: GIF at the top, 3-line explanation, pipeline diagram, results
  tables, limitations, ethics/license note, how to run.
- `docs/INTERVIEW_NOTES.md`: every design choice, the question an
  interviewer might ask, and the answer, in plain language.
- `docs/METHOD.md`: the math (POS, SNR, phase lag), with figures.
- GitHub Actions running tests on every push.
- Resume bullet drafts based on the real numbers.

### Phase 10 — Stretch (optional)
- Small CNN on PPG maps (Intel-style) as a comparison row.
- Host the Streamlit demo on Hugging Face Spaces (upload only, no webcam).

## 9. Repository layout

```
pulsecheck/
├── pyproject.toml
├── README.md
├── configs/default.yaml        # window sizes, filter band, regions, paths
├── src/pulsecheck/
│   ├── face.py  regions.py  extract.py
│   ├── rppg.py  signal.py  features.py
│   ├── quality.py  model.py  verdict.py
│   ├── datasets/ (ffpp.py, celebdf.py, df26.py, ubfc.py)
│   ├── live.py  cli.py
│   └── viz.py
├── app/streamlit_app.py
├── scripts/ (extract_all.py, eval_hr.py, train.py, evaluate.py, case_studies.py)
├── tests/
├── docs/ (plan.md, METHOD.md, INTERVIEW_NOTES.md, figures/)
└── data/  (git-ignored: raw/, cache/, manifests/)
```

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Dataset approval slow or denied | Phases 3–6 blocked | Request on day one. Phases 0–2 need only UBFC + webcam. DF26 is auto-approved. |
| MediaPipe 1.x issue on Python 3.13 | Setup delay | Check in Phase 0; fall back to Python 3.12 venv |
| CPU too slow for full datasets | Smaller experiments | 15-second cap, subsets, overnight resumable runs |
| DF26 clips too short for good rPPG | Weaker modern-AI results | Report it as a finding; use short windows with zero-padding |
| New features don't beat baseline | Weaker headline | Honest negative result + analysis is still strong. Ablations show what *does* help. |
| Compression wipes out the pulse | Many "can't tell" | That's the correct behavior; report the rate |

## 11. Left to the builder (my calls)

- Exact landmark indices for regions, window lengths, filter order, SNR
  definition (de Haan & Jeanne 2013 style), classifier hyperparameters.
- Code style, module boundaries, plot styling, CLI flags.

## 12. Decisions from you (2026-10-26)

1. **Folder:** `Desktop\Deepfake project` (data and env kept outside
   OneDrive, see §3).
2. **Demo footage:** yes. Your own webcam clips (real) + a short AI video
   you generate yourself with a tool whose terms allow sharing (fake).
3. **Heart-rate reference:** no device. UBFC ground truth + manual pulse
   count (see §3).
4. **GitHub:** public repo under your account; you are the only
   contributor.
5. **Deadline:** none. Best possible outcome; Phase 10 is in scope.
