# PulseCheck

Detecting deepfakes by checking for a real heartbeat in facial video.

Every heartbeat changes skin colour by a tiny amount that a camera can pick up (remote
photoplethysmography, "rPPG"). In a real video all the skin patches on a face share one heart. In a
fake, the pulse may be missing, noisy, or inconsistent between patches. Modern face-swaps can even
*inherit* a believable pulse from the video they copy, so PulseCheck asks a harder question than
"is there a pulse?": is the pulse **physiologically consistent across the face** (does it arrive in
each region at the right time and with the right relative strength)? It rebuilds the idea from first
principles, tests it honestly, and checks how well it holds up. Design and plan:
[docs/plan.md](docs/plan.md).

## Status

**The code for the heart-rate engine and the detector (plan phases 0-5) is built and tested on
synthetic data. No result on real data exists yet.**

- **Heart-rate engine (phases 1-2):** reads a heart rate from a face video or a live webcam.
  Accuracy on real recordings is **not yet measured** (UBFC-rPPG is not downloaded, no own clips are
  recorded): [docs/results/hr_eval.md](docs/results/hr_eval.md) is "pending".
- **Detector (phases 3-5):** dataset manifests with leak-proof splits, resumable parallel
  extraction, four groups of per-video features, a leak-free real-face reference, logistic
  regression and gradient-boosting models, and an evaluation script. FaceForensics++ and Celeb-DF
  are **not downloaded yet**, so [docs/results/detection.md](docs/results/detection.md) is
  "pending" and this README quotes no detection number for real videos.
- **What is shown so far** is a synthetic showcase (clearly *not* real video): on simulated faces
  whose fake kind copies one pulse onto every region, group A alone ("is there a pulse?") scores
  AUC 0.483 and the full feature set scores 1.000 (5-fold CV, 30 + 30 videos). It shows the
  features can see a missing regional structure when it exists; it does not show that real
  deepfakes lack it. Details and caveats: [docs/METHOD.md](docs/METHOD.md) section 19.
- **Not built yet:** testing on modern AI video (DF26), the calibrated "likely real / likely AI /
  can't tell" verdict, the Streamlit demo.

## How it works

**Engine** (video to heart rate)

1. **Track the face:** MediaPipe Face Landmarker gives 478 landmarks per frame.
2. **Cut out skin regions:** seven skin patches (forehead x3, cheeks x2, nose, chin) plus one
   background "control" patch, each an outline of landmark points. The control sits to the
   image-left of the face unless less than 90% of it would be in frame.
3. **Average the colour** of every region in every frame. Tracking is the slow step, so the batch
   and evaluation scripts cache these colour traces on disk; the `pulsecheck` commands re-extract
   on every run.
4. **Combine and resample:** the whole-face signal averages the skin regions after dividing each
   by its own mean colour (so a region dropping in and out does not cause jumps), and every signal
   is resampled to an even 30 Hz grid using the real frame timestamps.
5. **Turn colour into a pulse** with GREEN, CHROM or POS (POS is the default).
6. **Band-pass** 0.7-3.3 Hz (42-198 bpm; CHROM filters inside the method, so it is not filtered
   twice), then read the strongest spectral peak in 8 s windows as the heart rate, together with a
   signal-to-noise ratio. A whole-clip heart rate is also given, unless more than 20% of the clip
   had no usable face.

**Detector** (cached traces to real / fake)

7. **Cut each video into 4 s windows** (1 s step), the same on every dataset, and compute four
   groups of explainable per-video features:
   - **A, pulse presence:** how strong and plausible is the heart-rate peak (the classic baseline).
   - **B, spatial agreement:** do the regions share one pulse (correlation, agreeing heart rates,
     coherence)?
   - **C, physiological consistency (the contribution):** when does the pulse arrive in each region
     (phase lag behind the forehead), how strong is it in each (amplitude map), what is its
     waveform shape, and is the background free of it? Two more features measure how far the lag
     and amplitude maps are from those of real faces (Mahalanobis distance, Ledoit-Wolf
     covariance), learned from the **training reals only**.
   - **D, quality:** duration, face size, motion, brightness.
8. **Classify** with logistic regression or gradient boosting inside one scikit-learn pipeline, so
   every fitted statistic is learned on training rows only. Train on the official FaceForensics++
   train split, set the threshold on its validation split, report ROC-AUC on its test split, on
   heavier compression (c40) and on Celeb-DF (never trained on).

The method, formulas and reasons for each choice are in [docs/METHOD.md](docs/METHOD.md). A Q&A on
every design decision is in [docs/INTERVIEW_NOTES.md](docs/INTERVIEW_NOTES.md).

![Skin regions on the public-domain test portrait](docs/figures/regions.png)

## Quickstart (Windows, PowerShell)

Python 3.12 or newer is needed (developed on 3.13). The virtual environment and all data live
**outside** the project folder, because the folder is synced by OneDrive.

```powershell
git clone https://github.com/JaskaranVaid/pulsecheck.git
cd pulsecheck
py -3.13 -m venv C:\dev\venvs\pulsecheck
C:\dev\venvs\pulsecheck\Scripts\python -m pip install -e ".[dev]"

$env:PULSECHECK_DATA = "C:\dev\pulsecheck-data"   # models, caches, recordings, datasets
```

`PULSECHECK_DATA` defaults to `~/pulsecheck-data` when unset. Set it permanently with
`setx PULSECHECK_DATA "C:\dev\pulsecheck-data"` (it applies to newly opened shells only). The
first run that needs face tracking downloads the Face Landmarker model (about 4 MB) into
`PULSECHECK_DATA\models`.

All commands below assume the `pulsecheck` executable from the venv, for example
`C:\dev\venvs\pulsecheck\Scripts\pulsecheck`, or activate the venv first with
`C:\dev\venvs\pulsecheck\Scripts\Activate.ps1`. MediaPipe prints a few harmless log lines to the
console on startup.

## Commands

| Command | What it does |
|---|---|
| `pulsecheck live [--camera 0] [--method pos]` | Webcam window with the regions outlined, a live pulse waveform and BPM. `q` or Esc quits. |
| `pulsecheck hr VIDEO [--method pos\|green\|chrom\|all] [--max-seconds 15]` | Heart rate and SNR per region and for the whole face. |
| `pulsecheck debug-video VIDEO --out OUT.avi` | Writes the video with the regions drawn on it, to check the tracking. |
| `pulsecheck extract VIDEO [--out traces.npz]` | Saves the per-region colour traces (the slow step) to an `.npz` file for inspection; the default name is `<video folder>_<video name>.npz` under `PULSECHECK_DATA\traces`. (This is not the batch cache; the batch scripts below manage that.) |
| `pulsecheck record [--seconds 60] [--out clip.avi]` | Records a webcam clip, then asks for the pulse you counted by hand and logs it to `clips.csv`. |

Examples:

```powershell
pulsecheck live
pulsecheck hr C:\dev\pulsecheck-data\own\clip.avi --method all
pulsecheck debug-video C:\dev\pulsecheck-data\own\clip.avi --out C:\dev\pulsecheck-data\debug.avi
pulsecheck record --seconds 60          # count your pulse at the wrist while it records
```

Evaluate against UBFC-rPPG and your own clips (writes a CSV and
`docs/results/hr_eval.md`; needs the dataset downloaded first, see below). UBFC is reported two
ways: pooled over 8 s windows, and per video (one whole-clip heart rate per subject, the protocol
most published results use). The script caches each video's colour traces under
`PULSECHECK_DATA\cache`, so later runs skip the face tracking:

```powershell
C:\dev\venvs\pulsecheck\Scripts\python scripts/eval_hr.py `
    --ubfc-root C:\dev\pulsecheck-data\ubfc\DATASET_2 `
    --own-clips C:\dev\pulsecheck-data\own\clips.csv --max-seconds 60
```

Regenerate the figures in `docs/figures/`: `python scripts/make_figures.py`.

## Running the detector on real data (your steps)

Nothing here has been run on real data yet; the numbers in the time column are estimates from
measured speeds, not measurements of these steps. Run everything from the project folder in
PowerShell with `PULSECHECK_DATA` set as above. Each script lists all its options with `--help`.

**1. Download the datasets (needs you).** FaceForensics++ (**videos only, c23 and c40**, no raw) and
the Celeb-DF v2 test videos both need an access form filled in your name; their licences allow
research use only and forbid redistribution. Put them here:

```
C:\dev\pulsecheck-data\ffpp\original_sequences\youtube\c23\videos\000.mp4 ...
C:\dev\pulsecheck-data\ffpp\manipulated_sequences\Deepfakes\c23\videos\<a>_<b>.mp4 ...   (also c40,
                                                  Face2Face, FaceSwap, NeuralTextures)
C:\dev\pulsecheck-data\celebdf\Celeb-real\, Celeb-synthesis\, YouTube-real\, List_of_testing_videos.txt
```

The FF++ folder layout is taken from the dataset's download script; the path templates are two
lines in `src/pulsecheck/datasets/ffpp.py` (`ORIGINAL_TEMPLATE`, `FAKE_TEMPLATE`) to change if your
download differs. Time: depends on your connection and on how long the access approval takes.

**2. Fetch the official FF++ splits** (three small JSON files, checked before they are written):

```powershell
C:\dev\venvs\pulsecheck\Scripts\python scripts/fetch_ffpp_splits.py
```

Time: seconds.

**3. Build the manifests** (one row per video: label, leakage group, split, compression). Prints a
split report with counts per split, label and kind, and how many expected files are missing:

```powershell
C:\dev\venvs\pulsecheck\Scripts\python scripts/build_manifest.py `
    --ffpp-root C:\dev\pulsecheck-data\ffpp `
    --celebdf-root C:\dev\pulsecheck-data\celebdf `
    --compression c23 c40 --max-train-pairs 200
```

`--max-train-pairs 200` keeps the first 200 FF++ training pairs (2,000 videos per compression);
validation and test are used in full. Writes `ffpp_c23.csv`, `ffpp_c40.csv` and `celebdf_test.csv`
to `C:\dev\pulsecheck-data\manifests`. Time: it lists files and decodes nothing, so seconds to a
minute or two.

**4. Extract the colour traces (the slow step; run it overnight).** Parallel, resumable, one face
tracker per video:

```powershell
C:\dev\venvs\pulsecheck\Scripts\python scripts/extract_all.py `
    --manifest C:\dev\pulsecheck-data\manifests\ffpp_c23.csv `
    --manifest C:\dev\pulsecheck-data\manifests\ffpp_c40.csv `
    --manifest C:\dev\pulsecheck-data\manifests\celebdf_test.csv `
    --workers 4
```

`--workers` defaults to half the logical cores minus one. If it is interrupted (Ctrl+C, a reboot, or
the built-in stall guard after 600 s without a finished video), run the same command again: finished
videos are skipped. A per-video log goes to `C:\dev\pulsecheck-data\results\extract_log.csv`; a
video that fails is logged and skipped and does not stop the run. Time, as an estimate:

```
hours ~ videos x frames per video / (workers x frames per second per worker x 3600)
```

Measured: one worker tracks about 60-65 frames/s on 640x800 video once MediaPipe is loaded; each
video is capped at 15 s (about 375-450 frames). How the rate drops when several workers share the
cores is not measured, so use this only as a lower bound (for example, 2,000 videos x 400 frames at
4 workers x 60 frames/s is about one hour) and read the real rate from the printed summary
(`videos_per_hour`, `frames_per_s`). Three manifests are many thousands of videos, so plan for
hours. Note that `ffpp_c40.csv` also lists c40 train and val videos, which the experiments do not
use (they score only the c40 test split) but this command still extracts.

**5. Build the feature tables** (reads cached traces only; nothing is extracted here):

```powershell
foreach ($m in "ffpp_c23", "ffpp_c40", "celebdf_test") {
    C:\dev\venvs\pulsecheck\Scripts\python scripts/build_features.py `
        --manifest C:\dev\pulsecheck-data\manifests\$m.csv
}
```

Writes `C:\dev\pulsecheck-data\features\<manifest name>.csv` and prints how many videos were
skipped (not cached, unreadable, unusable face) and the share of NaN values per feature. Time:
not measured on real data; it does no face tracking, so expect it to be much faster than step 4.

**6. Run the experiments:**

```powershell
C:\dev\venvs\pulsecheck\Scripts\python scripts/run_experiments.py `
    --features-dir C:\dev\pulsecheck-data\features
```

Trains and evaluates the models and **overwrites** the placeholder
[docs/results/detection.md](docs/results/detection.md) with the generated tables (write it
elsewhere with `--md-out`), writes the ROC figures `docs/figures/roc_e2.png` and `roc_e4.png`, and
two CSVs under `C:\dev\pulsecheck-data\results`. It needs `ffpp_c23.csv`; without it nothing is
written and the exit code is 1. Time: not measured on real data.

## Tests

```powershell
C:\dev\venvs\pulsecheck\Scripts\python -m pytest -q -W error
C:\dev\venvs\pulsecheck\Scripts\python -m ruff check .
```

The tests use synthetic signals and synthetic per-region videos with a known pulse and one
public-domain portrait. They need no webcam and no dataset. One test file (`test_face.py`) runs the
real face tracker and so needs the model download on first run.

## Data and licences

- **UBFC-rPPG, FaceForensics++, Celeb-DF** and the deepfake datasets planned for later stages are
  licensed for research and education only. Their videos are **never** committed to this repository
  and live under `PULSECHECK_DATA`. Request access from each dataset's authors.
- **Your own recordings** also stay out of git (`.gitignore` blocks `.avi`, `.mp4` and `.npz`).
- The only face image in the repository is an official NASA portrait (public domain, US government
  work); its source is in [tests/fixtures/README.md](tests/fixtures/README.md).

## Project layout

```
src/pulsecheck/
  face.py          MediaPipe Face Landmarker wrapper and model download
  regions.py       skin regions, control region, per-region mean colour
  extract.py       video -> per-region colour traces (with face landmarks), versioned cache
  dsp.py           resampling, band-pass, spectrum, heart rate, SNR, cross-spectrum, phase lag,
                   coherence
  rppg.py          GREEN, CHROM and POS methods
  pipeline.py      traces -> per-region and whole-face heart rate
  evaluation.py    heart-rate error metrics and ground-truth series
  datasets/        ubfc.py (UBFC loader), manifest.py (VideoItem + CSV), ffpp.py, celebdf.py
  splits.py        leakage guard, grouped cross-validation folds, split report
  batch.py         resumable parallel extraction (spawn pool, run log, stall guard)
  features.py      feature groups A-D, one row per video
  reference.py     RealFaceReference: Mahalanobis distance to real faces (Ledoit-Wolf)
  model.py         scikit-learn pipelines, metrics, threshold, permutation importances
  live.py          webcam monitor and clip recorder
  viz.py           region overlay drawing
  cli.py           the `pulsecheck` command
  config.py        settings and data folders
scripts/           eval_hr.py (heart-rate evaluation), fetch_ffpp_splits.py, build_manifest.py,
                   extract_all.py, build_features.py, run_experiments.py, make_figures.py,
                   make_landmark_fixture.py
tests/             unit tests, synthetic-signal and synthetic-video helpers, public-domain fixture
docs/              plan.md, METHOD.md, INTERVIEW_NOTES.md, figures/, results/
```

## Roadmap

From [docs/plan.md](docs/plan.md), section 8. Code for phases 0-5 is done; what is missing is real
data.

- **Phases 2-5 (remaining, needs you):** download UBFC-rPPG and record own clips for the
  heart-rate evaluation; download FaceForensics++ and Celeb-DF and run the six steps above.
- **Phase 6:** test on modern AI video (DF26) and document where it fails.
- **Phase 7:** calibrated verdict: likely real / likely AI / can't tell.
- **Phase 8:** Streamlit demo app.
- **Phase 9:** presentation: results tables, limitations, CI.
- **Phase 10 (stretch):** small CNN comparison.

## License

MIT. See [LICENSE](LICENSE).
