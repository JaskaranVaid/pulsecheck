# Method: the heart-rate engine and the deepfake detector

This document explains how PulseCheck turns a face video into a heart rate (sections 1-11), and how
it turns the pulse signals of many videos into a real-vs-fake detector (sections 12-20), and why
each step is built the way it is. The overall design is in [plan.md](plan.md). **No result on real
deepfake videos exists yet**; the only detector numbers here come from a clearly labelled synthetic
showcase (section 19).

Contents: [1 Why skin colour pulses](#1-why-skin-colour-pulses) ·
[2 Pipeline](#2-pipeline) · [3 Regions](#3-regions) · [4 From colour to pulse](#4-from-colour-to-pulse-green-chrom-pos) ·
[5 Resampling](#5-resampling-to-a-uniform-30-hz-grid) · [6 Filter, spectrum, heart rate](#6-band-pass-spectrum-and-heart-rate) ·
[7 SNR](#7-signal-to-noise-ratio-snr) · [8 Windows and validity](#8-windows-and-validity) ·
[9 Evaluation](#9-evaluation) · [10 Results](#10-results) · [11 Limitations](#11-known-limitations) ·
**Detector:** [12 Datasets and splits](#12-the-detector-datasets-splits-and-the-leakage-rules) ·
[13 Batch extraction](#13-batch-extraction-and-the-trace-cache) ·
[14 4 s windows](#14-why-fixed-4-s-feature-windows) · [15 Feature groups](#15-feature-groups) ·
[16 Real-face reference](#16-the-real-face-reference-mahalanobis-distance-with-ledoit-wolf-shrinkage) ·
[17 Rolling shutter](#17-rolling-shutter-part-of-a-measured-lag-may-not-be-blood) ·
[18 Models and metrics](#18-models-metrics-and-the-experiments) ·
[19 Synthetic showcase](#19-the-synthetic-showcase-simulated-data-not-real-video) ·
[20 Detector limitations](#20-detector-limitations-and-what-is-not-measured)

## 1. Why skin colour pulses

Every heartbeat pushes a small pulse of blood through the vessels just under the skin. Blood
absorbs light, and it absorbs green light much more than red. So when more blood is under the skin,
the skin reflects a tiny bit less green light, and the face gets very slightly darker and
redder, then recovers before the next beat. The change is a fraction of a percent of the pixel
brightness, far too small to see, but a camera records it and averaging over many pixels makes it
measurable. This is **remote photoplethysmography (rPPG)**: "photoplethysmography" is measuring
blood volume with light (the green LED on a smartwatch does it), and "remote" means doing it with an
ordinary camera from a distance.

Everything else in the pipeline exists to dig this tiny periodic signal out of noise: camera
sensor noise, head motion, and changes in room lighting.

## 2. Pipeline

```
video frame
  -> face landmarks            MediaPipe Face Landmarker, 478 points            (face.py)
  -> 7 skin regions + control  convex hulls of landmark subsets                 (regions.py)
  -> mean R, G, B per region   one row per frame, NaN if region unusable        (extract.py)
  -> cache .npz (v3)           key: version, relative path, size, cap; plus landmarks (extract.py)
  -> resample to 30 Hz         on video timestamps, gaps over 0.5 s flagged     (dsp.py)
  -> GREEN / CHROM / POS       colour -> one pulse signal                       (rppg.py)
  -> band-pass 0.7-3.3 Hz      Butterworth, order 3, zero-phase (CHROM: once,   (dsp.py)
                               inside the method)
  -> spectrum, peak, SNR       8 s windows, 1 s step                            (dsp.py)
  -> heart rate per region and for the whole face                               (pipeline.py)
```

Face tracking is the slow step (about 60-65 frames/second per worker on a 640x800 video on the
development laptop, an i5-1335U with no GPU, once MediaPipe is loaded; an earlier figure of about 20
frames/second included the start-up). The evaluation script (`scripts/eval_hr.py`) and the batch
extraction (section 13) therefore cache each video's colour traces (`extract.cached_traces`), so
re-running with another rPPG method, filter or window does not re-run the tracker. The cache key
includes `TRACES_VERSION` (now 3), a number that is raised whenever extraction changes what a trace
contains; every file stores it too, and a file with a missing or different version is treated as a
cache miss and recomputed. The key itself is described in section 13: with a dataset root it uses
the path relative to the root and no modification time, so moving the data folder keeps the cache.
The `pulsecheck` command line does **not** use this cache: `pulsecheck hr` re-extracts on every
run, and `pulsecheck extract` only saves a video's traces to an `.npz` file for inspection.

## 3. Regions

![The seven skin regions and the control region](figures/regions.png)

*The regions drawn on the public-domain NASA portrait in `tests/fixtures` (see
`tests/fixtures/README.md` for its source and licence). Regenerate with
`python scripts/make_figures.py`. "Left" and "right" are as seen in the image, not the subject's
own left and right.*

- **Seven skin regions:** `forehead_left`, `forehead_center`, `forehead_right`, `cheek_left`,
  `cheek_right`, `nose`, `chin`. Each is the convex hull (the tightest convex outline) of a small
  set of landmark points, so it is a plain polygon that moves with the face. Eyes, eyebrows and
  lips are deliberately left out: they move and cast shadows on their own, which would swamp the
  pulse. `regions.py` lists forbidden landmarks for these and tests check, on the test portrait, that no skin region
  contains any of them.
- **One control region:** a patch of background beside the head at forehead height. There is no
  skin there, so a real pulse should not appear in it. If a "pulse" does show up in the control
  region, it came from the lighting or the camera, not from the heart. The patch is
  0.15-0.45 face widths to the image-left of the face, as long as at least 90% of it is inside the
  frame; otherwise it goes on whichever side has the larger share inside the frame (ties go
  left). The rule compares *shares*, not pixel areas, on purpose: an earlier version picked the
  side with more pixels, and when both patches were whole their areas differed only by rounding,
  so sub-pixel tracker jitter kept flipping the side (on the test portrait, 200 frames with
  uniform +/-0.3 px landmark jitter flipped it 65 times; the current rule never flips, and a test
  checks that). The two sides differ in colour, so the control trace became a square wave far
  larger than any pulse.
- **Clipping and unusable regions:** polygons are clipped to the image with
  `cv2.intersectConvexConvex`, so a face half out of frame gives a smaller region instead of an
  error. A region with fewer than 20 pixels returns NaN (not a number) for that frame, which later
  steps treat as "no data".
- **The `face` signal:** the combined colour of whichever skin regions have data in each frame. It
  is less noisy than any single region, and it is the number the heart-rate evaluation uses.
  Each region's R, G and B are first divided by that region's own average over the clip, so every
  region sits around 1.0; the per-frame mean of these is then multiplied by the average of the
  region averages to keep an RGB-like level (`pipeline.face_signal`, also used by the live view).
  Without the normalisation, a bright region (say a cheek) that is missing in some frames would
  pull the plain average up and down by a few percent each time it drops in or out, many times
  the pulse. A synthetic test (regions between 15% darker and 20% brighter than each other, the
  brightest cheek missing on a random 10% of frames) checks that GREEN's 8 s windows still read
  72 bpm with a mean error of at most 0.5 bpm.

## 4. From colour to pulse: GREEN, CHROM, POS

All three take a trace of mean `(R, G, B)` per frame, sampled uniformly, and return one pulse
signal. Write `Rn`, `Gn`, `Bn` for a channel divided by its own average, so each sits around 1.0
and only the relative changes remain.

**GREEN** (the baseline). Use the green channel alone: `g / mean(g) - 1`. Green carries the
strongest pulse (see section 1), so this works in a steady room. Its weakness is that any change
in overall brightness (a flickering lamp, a passing shadow, auto-exposure) moves green just as much
as the pulse does, and GREEN cannot tell them apart.

**CHROM** (de Haan & Jeanne, 2013). Skin colour changes from the pulse and from brightness have
different *directions* in colour space: brightness scales R, G and B together, while blood changes
them by different amounts. CHROM builds two combinations of the normalised channels:

```
X = 3 Rn - 2 Gn
Y = 1.5 Rn + Gn - 1.5 Bn
```

band-passes both, and subtracts one from the other with a weight chosen so that whatever X and Y
have in common (the brightness change) cancels:

```
alpha = std(X_filtered) / std(Y_filtered)
pulse = X_filtered - alpha * Y_filtered
```

This is the "global" version: one `alpha` for the whole clip. Because CHROM's own definition
band-passes X and Y before computing `alpha`, its output is already band-limited, and the
pipeline does not band-pass it a second time (GREEN and POS are band-passed once, after the
method).

**POS** (Wang et al., 2017, "plane orthogonal to skin"). POS also works on the normalised channels
but projects them onto two axes that are *exactly* perpendicular to the brightness direction
`(1, 1, 1)`:

```
S1 =        Gn - Bn          (axis  [ 0,  1, -1])
S2 = -2 Rn + Gn + Bn         (axis  [-2,  1,  1])
h  = S1 + (std(S1) / std(S2)) * S2
```

Because both axes have coefficients that sum to zero, a pure brightness change (the same
multiplier on R, G and B) contributes nothing to `S1` or `S2`. The weight `std(S1)/std(S2)`
combines the two projections so the pulse adds up while remaining noise is reduced. This is done on
a sliding **1.6 s window** (48 samples at 30 Hz): each window gives one `h` with its mean removed,
and the overlapping windows are added together. Short windows keep each projection roughly stable
even when the skin tone or lighting drifts.

**Why POS is the default.** A test in the repository (`tests/test_rppg.py`) builds a synthetic
72 bpm pulse and adds a lighting flicker at 108 bpm that is about five times stronger than the
pulse. GREEN reads 108 bpm, the flicker; CHROM and POS both read the true 72 bpm.

![GREEN versus POS spectra under lighting flicker](figures/flicker_methods.png)

*Spectrum of the same synthetic signal after GREEN (left) and POS (right). Regenerate with
`python scripts/make_figures.py`. This is a synthetic demonstration of why the method choice
matters, not a measured result on real video.*

All three methods are implemented in `rppg.py`, each in a few lines of NumPy. The choice of default
is tested on real video by the evaluation in section 9; until those results exist, POS is the
default because of the published literature and the flicker behaviour above, not because it won on
our data.

## 5. Resampling to a uniform 30 Hz grid

Video frames do not arrive at perfectly even intervals, especially from a webcam, and datasets mix
24, 25 and 30 fps. The filter and the Fourier transform both assume evenly spaced samples, and
uneven spacing shows up as fake frequencies. So `dsp.resample_uniform` linearly interpolates the
per-frame colour onto an even 30 Hz grid *using each frame's real timestamp*.

- Frames with no data (face lost) are dropped before interpolating.
- Interpolation across a gap is allowed only up to **0.5 s**. Inside a longer gap the grid is still
  filled (the filter needs a continuous signal) but those samples are flagged `valid = False`.
- **One shared grid, anchored to the start of the video.** The pipeline builds a single time grid from the
  first to the last frame and passes it to every region (the `t_out` argument). Samples before the
  first or after the last valid frame are flagged invalid, not trimmed. Consequences: all regions
  line up sample for sample, and every window's centre time is "seconds from the start of the
  video", the same clock the ground-truth heart rate uses.

## 6. Band-pass, spectrum and heart rate

- **Band-pass 0.7-3.3 Hz** (42-198 beats per minute), Butterworth order 3, applied forwards and
  backwards (`sosfiltfilt`) so it adds no time shift. Below 0.7 Hz is slow drift and breathing-rate
  changes; above 3.3 Hz is beyond any plausible resting or exercising heart rate.
- **Detrending.** The plan's "detrend" step is done by the 0.7 Hz high-pass edge of this filter,
  plus the constant (mean) detrend that `scipy.signal.periodogram` applies to each window.
- **Spectrum:** a Hann-windowed periodogram (a squared Fourier transform with a smooth window to
  limit leakage), zero-padded to at least 4096 points. Zero-padding does not add real frequency
  resolution (that is set by the window length, `1/T` Hz) but it samples the spectrum finely enough
  that the peak position is not rounded to a coarse grid.
- **Heart rate** = the frequency of the strongest spectral peak inside 0.7-3.3 Hz, times 60.
- A flat or constant input returns NaN instead of a confident-looking number.

The project plan mentioned a Welch spectrum. A single windowed periodogram is used instead:
Welch splits an 8 s window into shorter segments and averages them, which costs frequency
resolution that 8 s cannot spare.

## 7. Signal-to-noise ratio (SNR)

The SNR says how much of the signal's power sits at the heart rate (and its harmonic) rather than
spread across everything else. It follows the style of de Haan & Jeanne (2013):

```
f0     = frequency of the spectral peak in 0.7-3.3 Hz
signal = power within  f0 +/- 0.1 Hz     plus power within  2*f0 +/- 0.2 Hz
noise  = all other power between 0.5 and 5 Hz
SNR    = 10 * log10(signal / noise)       in decibels (dB)
```

The second harmonic (`2*f0`) is counted as signal because a real pulse wave is not a pure sine; it
has a sharp rise and a slower fall, which puts energy at twice the heart rate. 0 dB means signal and
noise power are equal; higher is cleaner.

SNR is the main "is there a pulse?" quality number that the later detector will use as a feature.

## 8. Windows and validity

- **8 s windows with a 1 s step** for sliding-window heart rate. The window is a trade-off between
  frequency detail and following changes in heart rate. Its natural bin spacing is `1/T` Hz
  (0.125 Hz, 7.5 bpm, at 8 s). The position of a single peak can be located more finely than that
  (the zero-padding in section 6), but to show up as two *distinct* peaks, two tones need to be
  about `2/T` apart with a Hann window (0.25 Hz, about 15 bpm, at 8 s), because the Hann main lobe
  is `+/- 2/T` wide.
- A window is **invalid (NaN)** if more than 20% of its samples are flagged invalid (face lost, or
  inside a gap longer than 0.5 s). Validity is tracked per region, so a region the tracker could not
  see is dropped for those windows without discarding the whole frame for the other regions. Each
  region's result carries its own validity mask on the shared grid (`RegionResult.valid`; all
  False for an unusable region), and the face's mask is `Analysis.valid`.
- The **whole-clip heart rate** (`RegionResult.hr`, used for own clips and the per-video UBFC
  protocol) follows the same 20% rule over the whole grid (the same constant,
  `dsp.MAX_INVALID_FRACTION`): if more than 20% of a region's grid, or the face's, is invalid,
  its whole-clip result is NaN instead of a number computed mostly from interpolation.
- A clip shorter than **3 s** is rejected (`ValueError: clip too short`). A clip between 3 s and 8 s
  yields one window over the whole clip.
- A region that cannot be analysed at all (too few valid samples, or a non-positive mean colour)
  gives NaN results; it does not crash the analysis of the other regions.

## 9. Evaluation

Two checks, both run by `scripts/eval_hr.py`:

1. **UBFC-rPPG** (public dataset of faces filmed with a pulse oximeter). The ground truth is
   computed from the oximeter's PPG signal with the *same* resampling, band-pass and peak rule as
   the video, so any difference comes from the video and not from two different estimators.
   Metrics are MAE (mean absolute error, in bpm), RMSE (root mean squared error, which punishes
   large misses more) and Pearson r (do the two numbers move together). Two protocols are
   reported, each in its own table in `hr_eval.md`:
   - **Windowed (8 s / 1 s, pooled).** 8 s windows every 1 s on both sides
     (`evaluation.reference_series`), matched by centre time (which must agree within 0.5 s),
     with the metrics **pooled over every window of every subject**. This measures how well the
     estimate tracks the heart rate over time.
   - **Per-video (whole clip).** One pair per subject: the face's whole-clip heart rate
     (`face.hr`) against the whole-clip heart rate of the PPG over the same duration
     (`evaluation.reference_hr`: resampled, band-passed, `estimate_hr`). This is the protocol most
     published UBFC-rPPG results use, so it is the one to compare with the literature. A subject
     whose whole-clip estimate is NaN (section 8) has no pair here.

   If the methods in one table end up with different numbers of pairs (one failed on a subject the
   others handled), the table gets a warning line: the methods were compared on different subject
   sets, so their rows are not directly comparable.
2. **Own webcam clips.** `pulsecheck record` saves a clip and asks for the pulse counted by hand at
   the wrist. `eval_hr.py` compares the whole-clip heart rate with that count. A clip whose
   whole-clip estimate is NaN (face invalid for more than 20% of it) is reported as an error row
   ("no valid face signal") and not counted as evaluated. The reference is rough (a few bpm of
   error), so this is a sanity check on real-world footage, not a precise accuracy figure.

## 10. Results

No real-data results exist yet. UBFC-rPPG has not been downloaded and no own-clips have been
recorded. The table in [results/hr_eval.md](results/hr_eval.md) is generated by the command below;
nothing is entered by hand.

| Method | UBFC windowed MAE / RMSE / r | UBFC per-video MAE / RMSE / r | Own clips MAE (bpm) |
|---|---|---|---|
| GREEN | pending | pending | pending |
| CHROM | pending | pending | pending |
| POS | pending | pending | pending |

To fill it in (after downloading UBFC-rPPG into the data directory and recording clips with
`pulsecheck record`):

```powershell
$env:PULSECHECK_DATA = "C:\dev\pulsecheck-data"
C:\dev\venvs\pulsecheck\Scripts\python scripts/eval_hr.py `
    --ubfc-root C:\dev\pulsecheck-data\ubfc\DATASET_2 `
    --own-clips C:\dev\pulsecheck-data\own\clips.csv --max-seconds 60
```

The goal set in the plan is POS within about 5 bpm MAE on UBFC. That is a target, not a result.

## 11. Known limitations

- **SNR depends strongly on clip length.** With the definition in section 7, the same clean 72 bpm
  sine scores very different SNRs depending on how long it is. Measured on the development machine
  (a noise-free 1.2 Hz sine sampled at 30 Hz, band-passed with `dsp.bandpass`, read with
  `dsp.estimate_hr`):

  | Clip length | 20 s | 8 s | 5 s | 4 s |
  |---|---|---|---|---|
  | SNR of an identical clean 72 bpm sine | 32.9 dB | 6.7 dB | 1.6 dB | -0.0 dB |
  | Heart rate read from it | 72.1 bpm | 72.1 bpm | 72.1 bpm | 71.6 bpm |

  The reason is spectral leakage: a Hann window of length `T` spreads a pure tone over a main
  lobe about `+/- 2/T` Hz wide. At 20 s that is `+/- 0.1 Hz`, which fits inside the signal band; at
  8 s it is `+/- 0.25 Hz`, so much of the tone's own energy lands outside the `+/- 0.1 Hz` band and
  is counted as noise. The heart rate itself stays within half a beat per minute (the small
  offsets come from the 4096-point spectrum's 0.44 bpm spacing and, at 4 s, from the band-pass
  filter's edge effects on so short a clip); only the SNR number shifts.

  *Design consequence for the detector:* SNR is only comparable between clips of the **same
  length**. Real, fake and modern AI videos will not have the same length (the DF26 clips are
  about 4-6 s), so later stages must compute features on equal-length windows for every dataset, and
  must not compare raw SNR across datasets with different clip lengths.
- **Short clips give coarse heart rates.** The bin spacing is `1/T` (7.5 bpm at 8 s) and two
  tones need about `2/T` (about 15 bpm at 8 s) to appear as separate peaks; both get worse for
  shorter clips.
- **Accuracy on real video is not yet measured** (section 10).
- **Face tracking speed.** About 60-65 frames/second per worker on this laptop (640x800 video, once
  MediaPipe is loaded), so large batches still take hours (section 13).
- **Skin tone, motion and compression** all weaken the signal. These effects are not yet
  quantified.
- **Rolling shutter can bias the phase lags** the detector measures between regions at different
  heights; see section 17.
- **Detector limitations** are collected in section 20.

## 12. The detector: datasets, splits and the leakage rules

From here on the document covers the detector (plan phases 3-5). The heart-rate engine above turns
one video into per-region pulse signals; the detector turns each video into **one row of numbers**
(its features) and trains a classifier on those rows. Labels are 0 = real, 1 = fake.

**FaceForensics++ (FF++).** 1,000 original YouTube videos and, made from them, four kinds of fakes
(Deepfakes, Face2Face, FaceSwap, NeuralTextures), at two compression levels (c23: light, c40:
heavy). The official split files (`train.json`, `val.json`, `test.json`, fetched by
`scripts/fetch_ffpp_splits.py`) list **pairs** of video ids. A fake is named `<target>_<source>`:
its face is built from two originals, and both directions of every pair exist. So the pair is the
unit that must not be split.

- **Leakage group = pair:** `ffpp:{min(a,b)}_{max(a,b)}`. A pair's two originals and all of their
  fakes (4 methods x 2 directions) share one group, so 10 videos per pair.
- **Why:** a fake contains the real video's background, lighting and motion. If the real video were
  in training and its fake in test (or the reverse), the model could recognise the scene instead of
  the pulse and the test score would be inflated. Groups also never straddle cross-validation folds
  (`splits.grouped_folds_df` uses `StratifiedGroupKFold`).
- **Guard:** `splits.check_no_leakage_df` raises if any group occurs in two splits, and
  `model.check_roles` runs the same check on the train / val / test frames before every fit, so
  passing the same rows twice (or c23 training rows next to their c40 copies as test) is also
  caught.
- **Subset:** `build_manifest.py --max-train-pairs N` keeps the first N training pairs in the
  official file order (a pair is kept or dropped as a whole); validation and test are used in full.

**Celeb-DF v2** is **test-only**: it is never used to train or to choose a threshold. It measures
whether what was learned on FF++ transfers to a dataset with other faces, other fake-making software
and other video quality. Its test videos come from the official `List_of_testing_videos.txt`. The
parser does not trust the list's label convention: it checks every line against its folder
(the real folders versus `Celeb-synthesis`) and stops if they disagree. Group: the identity
(`idN`, the target person for a synthesis file) or `yt:<name>` for YouTube-real videos.

**Everything fitted is fitted on training rows only.** The real-face reference (training reals
only), the median imputer and the scaler live inside one scikit-learn pipeline (section 18).
Thresholds are chosen on validation scores only. "Best" models are chosen by validation AUC, never
by test AUC. Section 18 lists the experiments.

## 13. Batch extraction and the trace cache

Face tracking is far slower than everything after it, so it runs once per video and its result is
cached (`extract.cached_traces`). Trace cache **version 3** stores, per frame, the mean RGB of every
region **and the 478 face landmarks** (as float16, a resolution of about 0.25-1 px at 512-2048 px
frame size: enough to rebuild the regions, not for sub-pixel motion), so regions can be recomputed
later without tracking again.

- **Cache key.** With a dataset root (the batch scripts always pass one):
  `v3|<path relative to the root>|<file size>|<max_seconds>`, hashed. No modification time and no
  absolute path, so moving or copying the whole data folder keeps the cache valid. Without a root
  (`eval_hr.py`): the absolute path, file size, modification time and `max_seconds`. The version
  number is part of the key and stored in each file; a file with another version is treated as a
  miss and re-extracted.
- **`scripts/extract_all.py`** (`batch.run_batch`): a `multiprocessing` Pool using the **spawn**
  start method (workers start as fresh interpreters, as Windows does anyway, instead of inheriting
  MediaPipe and OpenCV state by fork), **one `FaceTracker` per video** (MediaPipe's video mode needs
  strictly increasing timestamps and follows the face from frame to frame, while every video's
  clock restarts at zero; a tracker per video keeps videos independent), and
  `cv2.setNumThreads(1)` in every worker so that N workers do not each start N OpenCV threads. The
  face model file is downloaded once in the parent before the pool starts (a race between workers
  downloading it on a fresh install was found and fixed). The scripts have `if __name__ ==
  "__main__"` guards, which spawn requires.
- **Resumable.** Finished videos are in the cache, so re-running the same command reports them as
  `cached` and continues. Cache files are written under a temporary name and moved into place, so a
  crash cannot leave a half-written entry.
- **Fails safely.** A per-video error (no face, unreadable file, any exception) is logged to
  `extract_log.csv` and the run continues. A **stall guard** stops the run, with a resume hint, when
  no video has finished for 600 s (a worker died hard, or a video hangs). Ctrl+C is responsive
  (the parent polls once a second) and prints the same hint.
- **Speed.** Measured on the development laptop: about 60-65 frames/s per worker on 640x800 video
  once MediaPipe is loaded (the earlier 20 frames/s figure included the start-up). The run prints
  the real `videos_per_hour` and `frames_per_s`. How the per-worker rate changes with more workers
  has not been measured, so a time estimate is `videos x frames per video / (workers x frames/s)`
  and should be read as a lower bound.

## 14. Why fixed 4 s feature windows

Every detector feature is computed on **4 s windows with a 1 s step, on every dataset**
(`SETTINGS.feature_window_s`, `feature_step_s`); the heart-rate engine keeps its 8 s windows.

- **Why not longer.** An 8 s window gives a better heart-rate reading (finer frequency bins: 1/T Hz),
  but the SNR of section 7 depends on window length (section 11), so features must be computed on
  windows of one length for every dataset, and the shortest clips set that length. The DF26 clips
  planned for a later stage are only 4-6 s, and a 4 s window fits in all of them.
- **What it costs.** (1) The frequency bin is 0.25 Hz (15 bpm) and a pulse spreads over a Hann main
  lobe of +/- 0.5 Hz, so a single window's heart rate is coarse. (2) With the SNR of section 7 a
  4 s window cannot score above about 0 dB even for a perfectly clean pulse (measured: -0.0 dB for
  a noise-free sine, section 11), so the top of the SNR scale is cut off: **group A's SNR features
  are weak by construction.** They can still tell a noisy window from a cleaner one, but not a
  clean one from a very clean one. This weakens the baseline; it is a cost of the protocol, to be
  remembered when reading any "baseline vs. our features" comparison. (3) Sliding windows overlap
  by 3 s, so consecutive windows are strongly correlated: the median over windows is a smoother,
  not an average of independent measurements.
- **Clips shorter than 4 s** (but at least 3 s) get one window over the whole clip. A clip under 3 s
  is rejected, and that video gets no feature row.
- **Clip cap.** Only the first 15 s of each video are extracted (`SETTINGS.max_seconds`).

## 15. Feature groups

Every feature name starts with its group letter. A feature that cannot be computed (no valid window,
region unusable) is NaN; the models impute NaN with the training median, and
`build_features.py` prints the NaN rate of every feature, which `run_experiments.py` repeats in
`detection.md`. A window is **valid** when its face heart rate is not NaN; `f0` below is that
window's face heart rate in Hz (bpm / 60), and every per-window group B/C quantity is read at it.
A "usable" region is one whose pulse is finite and has at most 20% invalid samples in the window.

### Group A: pulse presence (the classic baseline)

How strong and plausible is a heartbeat in the face signal and in each skin region?
`a_snr_face`, `a_snr_regions_median`, `a_snr_regions_max` (median window SNR, in dB, of the face,
and the median and the maximum over skin regions of each region's median SNR), `a_hr_std_face` (std
of the face's window bpm), `a_hr_in_rest_range` (fraction of valid windows with 45-150 bpm) and
`a_valid_frac` (fraction of the clip with face data). SNR is weak at 4 s (section 14).

### Group B: spatial agreement

Do the seven skin regions carry one shared heartbeat?

- `b_corr_median`, `b_corr_min`: per valid window, the Pearson correlation of every pair of usable
  skin-region pulses; the median and the minimum over pairs; then the median over windows.
- `b_hr_spread`: per valid window the std of the regions' bpm values; median over windows.
- `b_coherence`: the **debiased band coherence**, below.

**Coherence.** Correlation compares two signals sample by sample. Coherence asks the same in the
frequency domain at the heart rate: does region `y` follow region `x` with a *consistent* amplitude
ratio and timing from window to window? With `X`, `Y` the Hann-windowed Fourier transforms of the
two regions' pulses, `Sxy = X conj(Y)` the cross-spectrum, `Sxx = |X|^2`, `Syy = |Y|^2`, and sums
over the bins within +/- 0.1 Hz of `f0` and over `K` windows:

```
C = |sum Sxy|^2 / ( sum Sxx * sum Syy )          in [0, 1]
```

**The bias, and the fix.** Each 4 s window contributes only about one independent spectral value
near `f0` (the zero-padded neighbouring bins are redundant). So for two *unrelated* signals pooled
over `K` windows, `C` is not 0 but about `1/K`: a short clip looks more coherent than a long one
by chance alone. `b_coherence` therefore (1) uses **non-overlapping** 4 s windows (overlapping
windows share samples and would fake consistency); a pair needs `K >= 2` windows where both regions
are usable, so a clip needs at least 8 s, else the feature is NaN; and (2) subtracts the floor:

```
C_debiased = (K * C - 1) / (K - 1)         # 1 stays 1; chance level (about 1/K) goes to about 0
```

It is **not clipped at 0**: for unrelated signals the estimates scatter around 0, and cutting off
the negative ones would leave a positive average that again depends on `K`. A slightly negative
value just means "no shared pulse". Measured on simulated unrelated signals (50 seeds,
`tests/test_dsp.py`), the mean raw coherence is 0.45 / 0.31 / 0.17 for `K` = 2 / 3 / 6, and after
debiasing -0.10 / -0.04 / 0.00; for a truly shared pulse the debiased value stays above 0.8.
`b_coherence` is the mean over the 21 region pairs, read at `f0` = the median face heart rate of the
kept windows. (With the default 15 s cap `K` is at most 3; K = 6 only occurs in the test.)

**What group B does not do.** It does not penalise an *inherited* pulse: a pulse pasted uniformly
onto the whole face is shared perfectly by all regions. B catches faces with no common pulse at all
(generator noise). See the synthetic showcase (section 19) for what B does separate there.

### Group C: physiological consistency (the contribution)

In a living face the pulse wave is expected to reach different regions at slightly different times
and with different strength (plan.md section 6.2). A pulse copied onto the whole face arrives
everywhere at once and equally. Group C measures that pattern. Reference region: `forehead_center`.

**Phase lag (`c_lag_<region>`, six regions; `c_lag_stability`).** If `y(t) = x(t - d)`, then
`Y(f) = X(f) e^{-i 2 pi f d}`, so the cross-spectrum has phase `angle(X conj(Y)) = +2 pi f d`:

```
lag d = angle( Sxy(f0) ) / (2 pi f0)      seconds; positive = the region lags the forehead
```

`Sxy` is read at the bin nearest `f0`, and the result lies in `(-1/(2 f0), 1/(2 f0)]` (about +/- 0.4
s at 72 bpm; a delay beyond that would wrap around). Why the cross-spectrum and not the peak of a
cross-correlation: the 30 Hz sampling period is 33 ms, and the delays in question are expected to be
of that order or smaller. The phase of a spectral line is estimated to a fraction of a sample,
whereas a correlation peak is quantised to whole samples. The feature is the median lag over valid
windows; `c_lag_stability` is, per region, the std of its window lags, averaged over regions (a
real pulse wave keeps its timing, noise-driven lags wander). A lag measured in a video is partly
**not blood flow** (section 17).

**Amplitude map (`c_amp_<region>`, seven regions).** The region's pulse amplitude at the heart rate,
`a_r = sqrt( power within +/- 0.1 Hz of f0 )`, as a share of the seven regions' total:
`share_r = a_r / sum_r a_r`. Each window's map sums to 1; the feature is the median share over
windows. Only windows where **all seven** skin regions are usable count (a share of an incomplete
sum would be silently inflated), so one region that is never usable makes all seven NaN. Each
region's pulse is built from its own colour after dividing by that region's mean colour, so the map
compares pulse strength, not skin brightness.

**Waveform shape (`c_harmonic_face`, `c_harmonic_spread`).** A heartbeat is not a sine: a sharp
rise and a slower fall put energy at twice the heart rate. `harmonic_ratio` = power within +/- 0.2 Hz
of `2 f0` over power within +/- 0.1 Hz of `f0`, for the face pulse (median over windows), and the std
across regions of their median ratio (one heart should give one shape everywhere).

**Control-region ratio (`c_control_ratio`).** Power at `f0` in the background control region's
signal over the face's, median over windows. Background does not pulse, so a high value means the
"pulse" is lighting or compression flicker.

**Correlation stability (`c_corr_stability`).** The std over windows of the per-window median pairwise
correlation.

### Group D: quality

`d_duration_s`, `d_face_found_ratio`, `d_face_width_px`, `d_motion`, `d_brightness`, `d_n_windows`.
These describe the video, not the pulse. They can also separate datasets for reasons unrelated to
heartbeats (a fake set filmed or encoded differently), so ABC and ABCD are reported separately and
the importance of group D is shown (section 18).

## 16. The real-face reference: Mahalanobis distance with Ledoit-Wolf shrinkage

A single number such as "the chin lags by 30 ms" is hard to judge, but the whole six-number lag map
and seven-number amplitude map can be compared with the maps of known real faces.
`reference.RealFaceReference` learns, for each map, the mean `mu` and the covariance `Sigma` of the
**training reals** and scores every video by

```
d(x) = sqrt( (x - mu)^T  Sigma^-1  (x - mu) )          Mahalanobis distance
```

giving two extra columns, `c_amp_mahal` and `c_lag_mahal`.

- **Why Mahalanobis, not Euclidean distance.** The entries of a map are correlated (the amplitude
  shares sum to 1; the two cheeks move together) and have different natural spreads (a chin lag
  may vary more than a forehead one). `Sigma^-1` accounts for both, so "unusual for a real face" is
  measured in units of how much real faces actually vary.
- **Why Ledoit-Wolf shrinkage.** With a few dozen training reals, the sample covariance of 6-7
  columns is noisy and can be nearly singular. For the amplitude map it is close to singular by
  construction, because the shares sum to about 1. Ledoit-Wolf blends the sample covariance `S`
  with a scaled identity, `Sigma = (1 - s) S + s (tr(S)/p) I`, with the blend `s` chosen
  automatically by a formula that minimises the expected squared error, so the inverse is stable.
- **Rows with NaN** are dropped from the fit of that map; fewer than 5 complete training reals gives
  NaN distances.
- **Why it is fitted inside the pipeline.** The reference is statistics of real faces. If it were
  fitted on all the data before splitting, held-out reals would help define "normal" and so look
  closer to it than they should (and fakes would be scored against a reference that has seen them),
  inflating the score. As the first step of the scikit-learn `Pipeline`, it is refitted on each
  training fold's reals (`y == 0`) only, and a held-out video never influences the reference it is
  scored against. For the same reason the two distance columns are not in the per-video feature
  table: they depend on which videos are the training reals, so they are not a property of one
  video and are added at model time.

## 17. Rolling shutter: part of a measured lag may not be blood

Most cameras read the sensor row by row, so a lower part of the image (the chin) is captured a few
milliseconds after a higher one (the forehead). Part of a measured lag between regions at
different heights can therefore be **row-readout time**, not the pulse wave. The offset scales with
the face's vertical extent in the frame (so with distance, zoom and resolution); it is not one
constant per camera. The real-face reference absorbs it **only to the extent that the training reals
share a similar capture setup and framing**. Videos filmed differently get a different offset, and
the raw `c_lag_*` columns, which also reach the classifier, carry the offset unadjusted. A model
could therefore partly learn "which camera" instead of "which heart". This has not been measured on
real data; it is a reason to read cross-dataset results (E4) with care.

## 18. Models, metrics and the experiments

**Model.** One scikit-learn `Pipeline` over the feature table:

```
[RealFaceReference] -> median imputer -> [StandardScaler] -> classifier
```

The reference is present when a `c_amp_` / `c_lag_` column is used. The imputer fills NaN with the
training median (a column that is NaN throughout training becomes 0 instead of vanishing). The
scaler is used for logistic regression only. Two classifiers: **logistic regression** (class-balanced;
a linear, readable baseline whose weights say which features push towards "fake") and
**gradient boosting** (`HistGradientBoostingClassifier`, class-balanced, fixed seed; it can combine
features non-linearly, for example "short lag AND equal amplitudes"). The tables are a few thousand
rows by about 35 columns, where these models train in seconds on a CPU, and a neural network has
little to add on so few numbers per video; staying explainable is part of the point. Class weights
are "balanced" because FF++ has 1 real to 4 fake.

**Metrics.** **ROC-AUC** is the main metric: the probability that a random fake scores higher than a
random real. It does not depend on a threshold or on the real:fake ratio, which differs between
datasets. At the validation **Youden threshold** (the score maximising true-positive rate minus
false-positive rate on the validation split, chosen there and applied to test unchanged) we also
report balanced accuracy (mean of the recall on reals and on fakes), precision and recall of the
fakes, and the confusion matrix. Cross-validation reports the mean and std of the test-fold AUC
over 5 stratified grouped folds on the training split. Undefined metrics (one class only, nothing
predicted fake) are NaN, never a quiet 0.

**Experiments** (`scripts/run_experiments.py`, feature groups A, AB, ABC, ABCD, both classifiers):

- **E2.** FF++ c23: fit on the official train split, threshold on val, scored on test; test AUC per
  manipulation (that manipulation's test fakes against **all** test reals); 5-fold grouped CV on
  train.
- **E3.** The same models scored on the **c40** test split (trained on c23, tested on heavier
  compression).
- **E4.** The same models scored on the **Celeb-DF** test list (a dataset never trained on).
- **E6.** Permutation importance of the best ABCD model (the one with the best *validation* AUC):
  per input and per feature group, i.e. how far test AUC drops when the column(s) are shuffled
  across test videos (a group is shuffled together, 10 repeats); plus an ablation that refits
  without one group. The two Mahalanobis columns are computed inside the model, so they cannot be
  shuffled alone; their effect shows through the raw `c_amp_*` / `c_lag_*` inputs and through
  group C.

`run_experiments.py` writes `docs/results/detection.md` with the protocol, the input counts, the
NaN rate of every feature and every table above, plus ROC figures and two CSVs. It needs
`ffpp_c23.csv`; with it missing it writes nothing and exits with code 1. **Until it has been run on
real data, [results/detection.md](results/detection.md) is a pending placeholder: the repository
quotes no detection number on real videos.**

## 19. The synthetic showcase (simulated data, not real video)

**Set-up** (`tests/test_reference.py`, `tests/synth.py`). Simulated "videos" are the per-region
colour traces a face tracker would produce, with a known pulse, sensor noise and slow lighting drift:

- **real:** one heart drives all regions, arriving with small delays (forehead 0 ms, nose 20 ms,
  cheeks 30 ms, chin 50 ms) and different strength (cheeks 1.3x, nose 0.8x, chin 0.7x the forehead,
  each +/- 10% at random). These delays and strengths are values chosen for the test, **not
  measured from faces**.
- **inherited:** a deepfake that carries a pulse copied from its driver video and pasted
  uniformly: the same pulse, zero delay and equal strength in every region.
- **noise:** no heartbeat; independent noise in each region (used by the feature tests).

30 real vs 30 inherited videos, 5-fold cross-validation, logistic regression, every fitted
statistic (including the reference) refitted per training fold. Mean ROC-AUC (CV seed 0):

| Features | AUC |
|---|---|
| A (pulse presence only) | 0.483 (0.467-0.544 across five CV seeds) |
| B only | 0.906 |
| A + B | 0.861 |
| A + B + C | 1.000 |
| A + B + C without the real-face reference | 1.000 |
| A + only the two Mahalanobis distances (raw maps removed) | 1.000 |
| C only (with the reference) | 1.000 |

**What this shows.** The group-A baseline, which asks "is there a pulse?", is at chance on these
inherited-pulse fakes: they carry a pulse as strong as the real ones. Features that look at the
*pattern* across regions (group C) separate them perfectly, and the two reference distances alone
(with every raw lag and amplitude column removed) carry the separation, so the reference works as
designed. The code path from traces to features to a leak-free model is wired correctly end to end.

**What this does not show.**

- It does **not** show that real deepfakes lack a regional pulse pattern. The fake kind was *defined*
  as having none, and the "real" lags and amplitudes were chosen by the author. A perfect 1.000 on
  two noise-free simulated classes is what such a design gives, not evidence about real video.
- It does **not** show that the reference helps. On this data the full model reaches 1.000 with or
  without it, because the raw maps are already separable. The reference's expected value, absorbing
  an offset that the training reals share (section 17), can only appear on real data.
- It does **not** mean group B is the contribution. B alone scores 0.906, but not because the
  inherited fake's regions fail to share a heartbeat: they share it perfectly and are, if anything,
  *more* alike (median over windows of the weakest pairwise correlation `b_corr_min`: about 0.59 vs
  0.50 for real). The simulated real face has regions with a weaker pulse (chin 0.7x, nose 0.8x),
  whose lower SNR lowers their correlation with the others. B separates the kinds through that
  amplitude heterogeneity, an artifact of the simulation that may not carry over to real data.
- 60 videos is small: the numbers vary with the CV seed (A: 0.467-0.544, B: 0.906-0.956).

## 20. Detector limitations and what is not measured

- **No real-data result exists.** FF++ and Celeb-DF are not downloaded; every detection number in the
  repository is synthetic and labelled so. The honest outcome the plan allows includes "the new
  features do not help": if so, the results will say so.
- **Short clips.** Group B's coherence needs two non-overlapping 4 s windows, so clips under 8 s
  give NaN for `b_coherence`; clips under 3 s give no row at all. With the 15 s cap, `K` is at most 3.
- **Group A is weak by construction at 4 s** (section 14), which makes the baseline modest.
- **Missing regions** make all seven `c_amp_*` NaN (section 15), and the imputed median then stands
  in for them; the NaN-rate table shows how often this happens.
- **Rolling shutter and capture setup** can leak into lags (section 17); group D can separate datasets
  for reasons other than heartbeats (section 15).
- **Compression** weakens the pulse; how much is what E3 measures.
- **Skin tone, motion and lighting** are not controlled for or quantified here.
- **The FF++ file layout** (`.../videos/<id>.mp4`) comes from the dataset's download script and is
  confirmed only on the first real download (two path templates in `datasets/ffpp.py`).
- **Not built yet:** modern AI video (DF26), the calibrated "can't tell" verdict, the demo app.
