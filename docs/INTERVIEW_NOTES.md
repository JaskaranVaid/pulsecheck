# Interview notes: the heart-rate engine and the detector

Questions an interviewer might ask about the two build stages (the rPPG heart-rate engine, then the
deepfake detector), with short answers in plain language. Jargon is explained the first time it
appears. Each answer points to the code or doc where the detail lives. The maths is in
[METHOD.md](METHOD.md); the overall design is in [plan.md](plan.md).

**The 30-second version.** Each heartbeat changes skin colour by a tiny amount. PulseCheck tracks
a face in video, averages the colour of several skin patches, turns that colour into a pulse
signal, filters it to heart-rate frequencies and reads the strongest frequency as beats per minute.
It is built so the pulse can be checked region by region, which is what the deepfake detector uses:
a real heart drives every skin region with small timing and strength differences, whereas a pulse
copied onto a fake face arrives everywhere identically. The detector turns each video into about 35
explainable numbers and trains a simple classifier, with splits designed so nothing leaks from
training to test. Everything so far is tested on synthetic data; real-video results are pending (see
"What is and isn't measured" at the end).

**Words used below.** *rPPG* (remote photoplethysmography): measuring blood volume from a camera.
*bpm*: beats per minute (Hz x 60). *Band-pass filter*: keeps frequencies inside a range and
removes the rest. *Spectrum*: how much of the signal's power sits at each frequency. *SNR*:
signal-to-noise ratio, in decibels (dB). *AUC*: the chance that a random fake scores higher than a
random real (0.5 = guessing, 1.0 = perfect). *Leakage*: information from the test data reaching the
training, which inflates results.

---

## Tracking and regions

**Why MediaPipe Face Landmarker?**
It gives 478 landmark points per face, enough to outline the forehead, cheeks, nose and chin
precisely, and it runs on a CPU. In its video mode it tracks the face from frame to frame, which is
steadier than detecting from scratch every time. The older MediaPipe Face Mesh API has been
retired, so the newer Tasks API is the maintained choice. Older alternatives either give only 68
points (dlib, no forehead) or only a box. (`face.py`)

**Why convex hulls for the regions?**
A convex hull is the tightest outline around a set of points that never has a dent. It is always
a valid polygon, it does not matter in which order the landmark indices are listed, and it
follows the face as the landmarks move. The alternative, a trained skin-segmentation model, would
add a heavy dependency for little gain at this stage. The trade-off is that a hull is not an exact
skin mask (a stray hair or glasses frame could fall inside). (`regions.py`)

**Why seven skin regions rather than one big face patch?**
One patch is the least noisy, and the code does compute one (the "face" signal, the mean of the
regions that have data, each first normalised by its own mean colour). But the later detector needs to ask whether different parts of the face
agree on the pulse, and that needs separate regions. A real heart drives all of them.

**Why leave out the eyes, eyebrows and lips?**
They move, blink and cast shadows on their own, and those changes are far larger than the pulse.
A test checks on the test portrait that none of those landmarks falls inside a skin region.

**Why a control region?**
It is a background patch beside the head, with no skin in it. A real pulse should not show up
there; if one does, the cause is lighting or the camera. It gives a built-in "false alarm" check
that the later detector can use. (`regions.py`)

**How is the control region's side chosen?**
It sits to the image-left of the face whenever at least 90% of that patch is inside the frame;
otherwise it goes on the side with the larger share inside the frame (ties go left). The first
version compared pixel areas instead. When both patches were whole, their areas differed only by
rounding, so tiny tracker jitter (+/-0.3 px) flipped the side 65 times in 200 frames of the test
portrait, and because the background differs left and right, the control trace became a square
wave far bigger than a pulse. Comparing *shares* with a preference for one side makes the choice
stable; a test jitters the landmarks over 200 frames and checks the side never changes.

**Why "left" and "right" as seen in the image?**
So there is one unambiguous convention. Subject-left would flip depending on whether you talk about
the person or the picture.

## Signal processing

**Why 0.7-3.3 Hz?**
That band is 42-198 beats per minute: it covers a resting heart rate (about 60) through
hard exercise, while excluding slow drift and breathing below 0.7 Hz and fast noise above 3.3 Hz.
A narrower band would reject real but unusual pulses; a wider one lets in more noise. (`config.py`)

**Why a Butterworth filter, order 3, run forwards and backwards?**
Butterworth is flat inside the pass band, so it does not reshape the pulse amplitude. Running it
forwards and then backwards ("zero-phase", `sosfiltfilt`) cancels the delay a filter normally adds,
so the pulse peaks stay in the right place in time. That matters later, when comparing the timing of
the pulse between regions. Order 3 is gentle enough to avoid ringing on short clips.

**Why resample everything to 30 Hz?**
Filters and Fourier transforms assume evenly spaced samples, but webcam frames arrive at slightly
uneven times and datasets mix 24, 25 and 30 fps. Uneven spacing creates false frequencies. So each
frame's real timestamp is used to interpolate the colours onto an even 30 Hz grid. 30 Hz is far
above the 3.3 Hz maximum heart-rate frequency, so nothing is lost. (`dsp.resample_uniform`)

**Why anchor the shared time grid to the start of the video?**
Every region and the face signal are interpolated onto one grid that runs from the first to the last
frame. Leading frames without a face are marked invalid instead of cut off. Two benefits: all
regions line up sample for sample, and a window centred at "12 s" means 12 s from the video start,
the same clock the ground-truth heart rate uses. Without this, windows from a trimmed signal would
be shifted and the comparison would be off by the amount trimmed. (`pipeline.analyze`)

**Why per-region validity instead of dropping bad frames?**
If one region is lost (a hand covers the chin, say), throwing away the whole frame would also lose
good data from the six other regions. Instead each region carries its own "valid" flag, gaps up to
0.5 s are bridged by interpolation, longer gaps are marked invalid, and a window with more than 20%
invalid samples is reported as NaN ("no answer") rather than a guess. The whole-clip heart rate
follows the same 20% rule over the whole clip (it reuses the same constant). Each region's result
also exposes its validity mask, so later stages can see exactly where a region had data. A region
that cannot be analysed at all gives NaN and an all-False mask instead of crashing the run.

**Why normalise each region before averaging?**
Regions differ in brightness (a lit cheek next to a shadowed chin, say). If the face signal were
a plain average and the bright cheek went missing for one frame (hair, a hand, the tracker clipping
it), the average would drop by a few percent and then jump back, which is many times the size of
the pulse (a fraction of a percent). So each region is first divided by its own average colour,
making every region hover around 1.0, and only then averaged; the result is scaled back to an
RGB-like level. A region dropping in or out now changes only the noise, not the level. A test
drops the brightest cheek on a random 10% of frames: GREEN, which has no protection of its own,
still reads 72 bpm within 0.5 bpm on average. The live view uses the same helper.
(`pipeline.face_signal`)

**Why 8 s windows with a 1 s step?**
The spectrum's bin spacing is 1 divided by the window length: 0.125 Hz, or 7.5 bpm, at 8 s. A single
peak can be located more finely than that, but two separate rhythms need to be about 2/T apart
(about 15 bpm at 8 s) to show up as two peaks with a Hann window, whose main lobe is +/- 2/T wide.
Longer windows are more precise but react slowly to a changing pulse. The 1 s step gives a new estimate every second, and consecutive windows overlap heavily.

**Why a Hann periodogram and not a Welch spectrum?**
The plan mentioned Welch, which averages the spectra of shorter overlapping segments. With only
8 s of data, shorter segments would blur the peak. A single Hann-windowed spectrum, zero-padded so
the peak position is sampled finely, keeps the full resolution. This is a deliberate change from
the plan. (`dsp.power_spectrum`)

**Why implement GREEN, CHROM and POS yourself instead of calling a library?**
Each is only a few lines of NumPy, so I can explain every line, and comparing the three is a results
row. The plan keeps pyVHR or rPPG-Toolbox as an optional cross-check of the numbers, not as the
engine. (`rppg.py`)

**Why POS as the default?**
It was designed (Wang et al., 2017) so that a change in brightness cancels out: its two projection
axes are perpendicular to the "all channels move together" direction. A test shows the practical
effect: with a lighting flicker at 108 bpm five times stronger than a 72 bpm pulse, GREEN reports
108 and POS reports 72. The honest caveat: POS is the default because of the literature and that
synthetic test; real-video accuracy still has to be measured.

**Why does GREEN fail under flicker?**
Flickering light changes R, G and B by the same percentage, so it changes the green channel as much
as the pulse does. GREEN looks at green alone and cannot tell which change is which, and if the
flicker is in the heart-rate band, the filter cannot remove it. CHROM and POS look at the *differences
between channels*, where the pulse changes them unequally but a brightness change changes them
equally, so the brightness part cancels.

**How is SNR defined?**
Find the strongest peak `f0` in the heart-rate band. "Signal" is the power within 0.1 Hz of `f0` plus
power within 0.2 Hz of twice `f0` (a real pulse wave is not a perfect sine, so it has a second
harmonic). "Noise" is all other power between 0.5 and 5 Hz. SNR = 10 log10(signal / noise) in dB. It
follows the style of de Haan & Jeanne (2013). (`dsp.estimate_hr`)

**Why does SNR depend on clip length?**
A tone seen through a short window is smeared out over a main lobe about +/- 2/T Hz wide (T is the
window length in seconds). At 8 s, that is +/- 0.25 Hz, wider than the +/- 0.1 Hz counted as signal,
so some of the pulse's own power is counted as noise. A clean, band-passed 72 bpm sine measured
32.9 dB at 20 s, 6.7 dB at 8 s, 1.6 dB at 5 s and -0.0 dB at 4 s, while the heart rate read from it
stayed within half a beat (72.1 bpm at 20, 8 and 5 s, 71.6 bpm at 4 s). The consequence
for the detector: compare SNR only between clips of equal length. The modern-AI dataset has clips of
only 4-6 s, so every dataset must be analysed on windows of the same length. (Details in
METHOD.md, section 11.)

## Software design

**Why is the face tracker injectable?**
`extract_traces` takes any object with a `detect(frame, timestamp)` method. Tests pass a fake that
returns stored landmarks, so they run in seconds, need no model download, and give exact results.
The real tracker is built in one place and imported lazily, because importing MediaPipe takes
seconds. It also means the command-line tool can check that the video file exists *before*
creating a tracker. (`extract.py`, `cli.py`, `tests/conftest.py`)

**Why cache the colour traces?**
Face tracking is the slow step, about 60-65 frames/second per worker on this CPU. The evaluation
and batch scripts cache each video's per-region colour traces (and face landmarks), keyed by the file
path, size and the `max_seconds` cap (plus the modification time when no dataset root is given), so
re-running with another rPPG method, filter or window never repeats the tracking. If a cache file is
unreadable, it is recomputed. The command-line tool does not use the cache: `pulsecheck hr`
re-extracts every time, and `pulsecheck extract` just saves one video's traces to a file for
inspection. (`extract.cached_traces`)

**Why version the cache?**
A cache key built only from the video's identity cannot tell that the *code* producing the traces
has changed. When the control-region rule changed, every old cache entry still matched its video
and would have been reused with the old, flipping control region. So the key and every saved file
carry `TRACES_VERSION`, raised whenever extraction changes what a trace contains; a file with a
missing or different version counts as a cache miss and is recomputed. (`extract.py`)

**Why are the settings a frozen dataclass and not a YAML file?**
There are about ten settings and they are used as plain Python values with type checking. "Frozen"
means no code can change them by accident mid-run. YAML would add a parser and a way to get
typos. (`config.py`)

**Why does `pulsecheck record` write MJPG `.avi` at a fixed frame rate?**
MJPG stores each frame as its own JPEG, so there is no inter-frame video compression, which can
smooth away the tiny colour changes that carry the pulse. The recorder repeats or skips frames so each lands at its true
capture time at a constant 30 fps, whatever rate the camera really delivers. The ground-truth pulse
you count is logged to `clips.csv` next to the clip.

**Why compare against the oximeter's PPG signal rather than its heart-rate column?**
The video estimate and the reference then go through the same band-pass, windows and peak rule, so
a difference reflects the video and not two different estimators. (`evaluation.reference_series`)

**Why report both windowed and per-video metrics?**
They answer different questions. The windowed numbers (8 s windows every second, pooled over every
window of every subject) say how well the estimate follows the heart rate over time, and a long
clean recording counts for more windows than a short one. The per-video numbers (one whole-clip
heart rate per subject, against the whole-clip rate of the oximeter signal over the same
duration) are what most published UBFC-rPPG papers report, so they are the ones to compare with
the literature; putting a pooled-window MAE next to a paper's per-video MAE would compare two
different things. Both tables are generated, with the protocol stated above each, and a table
warns if its methods ended up with different numbers of pairs. The per-subject numbers are kept
in the CSV so a single bad subject is visible.

## Environment and data

**Why is the data folder outside OneDrive?**
The project folder is synced by OneDrive. Datasets (tens of GB), the model file and the cache would
be uploaded needlessly, and the virtual environment is thousands of small files that sync badly. So
data lives in `C:\dev\pulsecheck-data` (set with the `PULSECHECK_DATA` environment variable) and the
venv in `C:\dev\venvs\pulsecheck`. Dataset licences also forbid redistributing the videos, and
keeping them out of the repo folder makes it harder to commit them by accident (`.gitignore` also
blocks video and `.npz` files).

**Why do the tests use a public-domain portrait and synthetic signals?**
Dataset footage cannot be redistributed, and a photo of a real person needs their permission. The
repository's one allowed face is an official NASA portrait (US government work, public domain;
source in `tests/fixtures/README.md`). Synthetic signals have a *known* heart rate, so a test can
assert "this must come out as 72 bpm" or "the flicker must not fool POS", which real footage cannot
offer. The 307 tests run without a webcam or a dataset.

**Why is everything CPU-only?**
The laptop has no NVIDIA GPU, and rPPG does not need one: it is filtering and a Fourier transform on
a few numbers per frame. The only heavy step is face tracking, which MediaPipe runs on the CPU at
about 60-65 frames/second per worker. That speed shapes the plan: only the first 15 s of each video
is processed, and batch runs use subsets and parallel workers (plan.md section 7).

## The detector: data and evaluation

**Why group FaceForensics++ by pair?**
A fake is built from two real videos, so it shares scenery, lighting and motion with them. If a real
video is in training and its fake is in test, the model can recognise the scene instead of the pulse
and the score is inflated. So each pair of originals and all of their fakes (10 videos) form one
group that is never split across train, validation, test or cross-validation folds, and the code
raises an error if one does. (`splits.py`, `datasets/ffpp.py`)

**Why is Celeb-DF test-only?**
It answers a different question: does what was learned on FaceForensics++ work on other people, other
fake-making software and other video quality? Training on it, or even tuning a threshold on it, would
remove that test. So it is only ever scored, using its official test list. (`datasets/celebdf.py`)

**Why are thresholds and "best model" chosen on validation, never on test?**
Picking the threshold, or the best of several models, on test data quietly fits the test set and
makes the reported number optimistic. The threshold (Youden's J) and the best-model choice use the
validation split; test is only scored. (`model.py`, `run_experiments.py`)

**Why 4 s windows when 8 s gives better heart rates?**
SNR depends on window length, so features have to use one length for every dataset, and the modern-AI
clips planned for later are only 4-6 s long. The cost is real: a 4 s window gives a coarse heart rate,
and a clean pulse cannot score above about 0 dB SNR, so group A (the classic baseline) is weak by
construction. I say so rather than hide it. (METHOD.md, section 14)

**Why phase lags between regions?**
If one pulse is pasted onto the whole face, it arrives everywhere at the same instant; in a living
face the pulse wave is expected to reach regions at slightly different times. The cross-spectrum
gives the delay as a phase at the heart-rate frequency, which can resolve fractions of the 33 ms
sample period where a correlation peak could not. The sign is fixed: positive means the region
lags the forehead. (`dsp.phase_lag_s`)

**What is coherence?**
It is correlation in the frequency domain: at the heart-rate frequency, does one region follow
another with a consistent timing and strength from window to window? It is 1 when they move in
lock-step and near 0 when they are unrelated. I use it because it ignores the other frequencies,
where noise lives. (`dsp.band_coherence`)

**Why debias coherence, and why not clip it at 0?**
Averaging only K windows makes even unrelated signals look coherent, about 1/K, so short clips would
look more "shared" than long ones; I subtract that floor with (K*c - 1)/(K - 1). I do not clip at 0
because unrelated signals scatter around 0, and cutting off the negative half would leave a positive
average that again depends on K. On simulated noise the raw means were 0.45 / 0.31 / 0.17 for
K = 2 / 3 / 6, and -0.10 / -0.04 / 0.00 after debiasing. (`dsp.debias_coherence`)

**Why Mahalanobis distance with Ledoit-Wolf shrinkage?**
A map of amplitudes or lags has correlated entries (amplitude shares sum to 1) and different natural
spreads, so plain Euclidean distance would treat all directions alike; Mahalanobis divides by how
much real faces actually vary. With only dozens of real videos the covariance is noisy and nearly
singular, so Ledoit-Wolf blends it with a scaled identity to keep the inverse stable. (`reference.py`)

**Why fit the real-face reference inside the pipeline?**
The reference is a statistic of real faces. If I fitted it on all data first, the held-out reals would
help define "normal", and the fakes would be scored against a reference that had seen them, which
inflates the result. Inside the scikit-learn pipeline it is refitted on each training fold's reals
only. That is also why the two distance columns are made at model time, not stored in the feature
table. (`reference.py`, `model.make_model`)

**Why logistic regression and gradient boosting rather than a neural network?**
There are about 35 numbers per video and a few thousand videos: both models train in seconds on a CPU,
and a network has little to learn from on so few inputs. Logistic regression is readable (its weights
say which feature pushes towards fake); boosting can combine features non-linearly. Keeping the
features explainable is the point of the project; a small CNN is only a planned stretch comparison.

**Why ROC-AUC as the main metric?**
AUC is the chance that a random fake scores higher than a random real, so it needs no threshold and
does not change with the real-to-fake ratio, which differs between datasets (1:4 in FF++). I also
report balanced accuracy, precision and recall at the validation threshold, and the confusion matrix,
because AUC alone hides what happens at the operating point. (`model.metrics`)

**Why is group A weak, and is the comparison with it fair?**
At 4 s the SNR saturates near 0 dB even for a clean pulse, so the baseline's best feature loses its
top end. That makes the baseline easier to beat than a full-strength FakeCatcher-style one, so any
"baseline versus ours" gap must be read with that in mind.

**Could the lags just be camera artifacts?**
Partly, yes: most cameras read the sensor row by row, so a lower region is captured slightly later
than a higher one. That offset grows with the face's vertical size in the frame, and the real-face
reference only absorbs it if the training reals were filmed similarly; the raw lag columns still
reach the classifier. I have not measured this on real data, so cross-dataset results are read with
care. (METHOD.md, section 17)

## The detector: synthetic showcase

**What does the synthetic showcase prove and not prove?**
On simulated faces where the fake copies one pulse onto every region, the pulse-presence baseline is
at chance (AUC 0.483) and the regional-pattern features reach 1.000. That shows the pipeline is wired
correctly and leak-free, and that the reference distances can carry the separation alone. It does
not show that real deepfakes lack such a pattern, since I defined the fake that way and chose the
"real" delays myself, nor that the reference helps, since the data is separable without it. These
numbers are synthetic, and I label them so everywhere.

**Why does group B separate the synthetic data?**
Not because the inherited fake's regions fail to share a heartbeat: they share it perfectly, and
are slightly more alike than the real ones. In my simulation some real regions (chin, nose) have a
weaker pulse, which lowers their SNR and correlation. So B is picking up an amplitude difference
built into the simulation, which may not carry over to real video. This is why I added tests that
isolate the reference and group C.

**What if the new features do not help on real data?**
Then the results will say so, with the ablation showing which groups did or did not matter. The plan
is explicit that an honest negative result with an analysis is still a result. Nothing in the
repository claims otherwise yet.

## The detector: running at scale

**Why spawn-safe workers and one tracker per video?**
On Windows, worker processes start as fresh interpreters ("spawn"), so scripts need a
`if __name__ == "__main__"` guard and everything sent to a worker must be picklable. A tracker per
video keeps videos independent: MediaPipe's video mode needs increasing timestamps and follows the
face frame to frame, but each video's clock restarts at zero. Each worker also limits OpenCV to one
thread, so N workers do not start N threads each and fight over the cores. (`batch.py`)

**How did you make the overnight extraction safe?**
It is resumable: every finished video is in the trace cache, so re-running the same command skips
it, and cache files are written under a temporary name then moved into place. A per-video error is
logged and skipped rather than ending the run, and a stall guard stops it cleanly, with a resume
hint, if nothing finishes for 600 s. Ctrl+C is responsive, and the face model is downloaded once in
the parent, because a race between workers downloading it on a fresh install was found and fixed.

**Why does the cache key use the relative path and no modification time?**
So the data folder can be moved or copied (to another drive, say) without invalidating days of
tracking; only the file size, the path inside the dataset and the extraction cap matter. Without a
dataset root (the heart-rate evaluation) it falls back to the absolute path, size and modification
time. The traces also store the face landmarks, so regions can be recomputed without tracking again.
(`extract.cache_path_for`)

## What is and isn't measured

**What have you actually shown so far?**
That the algorithm recovers a known heart rate from synthetic signals, tolerates gaps, jittered
timestamps and a lighting flicker, and that the full pipeline runs on a real face image through the
real tracker. For the detector, I have shown on simulated faces that a pulse-presence baseline cannot
see a uniformly copied pulse while the regional-pattern features can (synthetic, see above). What is
*not* yet shown is anything on real recordings: UBFC-rPPG, FaceForensics++ and Celeb-DF are not
downloaded, no own clips are recorded, and `results/hr_eval.md` and `results/detection.md` say
"pending" until the scripts have produced real numbers. The plan's heart-rate target is POS within
about 5 bpm on UBFC.

**What are the weaknesses?**
Short clips give coarse heart rates, and SNR changes with clip length (above), so the baseline group
is weak at 4 s. Part of a measured lag may be camera readout time, group D can separate datasets for
reasons unrelated to heartbeats, and clips under 8 s cannot give a coherence value. Dark skin, head
motion and heavy video compression weaken the pulse, and none of those effects is quantified yet.
The regions are hulls, not true skin masks.
