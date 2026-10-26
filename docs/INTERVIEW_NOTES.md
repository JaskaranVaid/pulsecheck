# Interview notes: the heart-rate engine

Questions an interviewer might ask about the first build stage (the rPPG heart-rate engine), with
short answers in plain language. Jargon is explained the first time it appears. Each answer points
to the code or doc where the detail lives. The maths is in [METHOD.md](METHOD.md); the overall
design is in [plan.md](plan.md).

**The 30-second version.** Each heartbeat changes skin colour by a tiny amount. PulseCheck tracks
a face in video, averages the colour of several skin patches, turns that colour into a pulse
signal, filters it to heart-rate frequencies and reads the strongest frequency as beats per minute.
It is built so the pulse can be checked region by region, which is what the later deepfake detector
will use. So far it has been tested on synthetic signals; real-video accuracy is pending (see
"What is and isn't measured" at the end).

**Words used below.** *rPPG* (remote photoplethysmography): measuring blood volume from a camera.
*bpm*: beats per minute (Hz x 60). *Band-pass filter*: keeps frequencies inside a range and
removes the rest. *Spectrum*: how much of the signal's power sits at each frequency. *SNR*:
signal-to-noise ratio, in decibels (dB).

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
regions that have data). But the later detector needs to ask whether different parts of the face
agree on the pulse, and that needs separate regions. A real heart drives all of them.

**Why leave out the eyes, eyebrows and lips?**
They move, blink and cast shadows on their own, and those changes are far larger than the pulse.
A test checks on the test portrait that none of those landmarks falls inside a skin region.

**Why a control region?**
It is a background patch beside the head, with no skin in it. A real pulse should not show up
there; if one does, the cause is lighting or the camera. It gives a built-in "false alarm" check
that the later detector can use. (`regions.py`)

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
invalid samples is reported as NaN ("no answer") rather than a guess. A region that cannot be
analysed at all gives NaN instead of crashing the run.

**Why 8 s windows with a 1 s step?**
Frequency resolution is 1 divided by the window length, so 8 s resolves about 7.5 bpm (finer than
that comes from locating the peak). Longer windows are more precise but react slowly to a changing
pulse. The 1 s step gives a new estimate every second, and consecutive windows overlap heavily.

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
so some of the pulse's own power is counted as noise. A clean 72 bpm sine measured 32.9 dB at 20 s,
6.7 dB at 8 s, 1.7 dB at 5 s and 0.0 dB at 4 s, with the heart rate right every time. The consequence
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
Face tracking is the slow step, about 20 frames/second on this CPU. The cache stores the
per-region colour traces keyed by file path, size, modification time and the `max_seconds` cap, so
changing the rPPG method, filter or window never repeats it. If a cache file is unreadable, it is
recomputed. (`extract.cached_traces`)

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

**Why pooled windows in the evaluation?**
Metrics are computed over every window of every subject together, so a subject with a long clean
recording counts for more windows than one that is short. The per-subject numbers are kept in the
CSV so a single bad subject is visible.

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
offer. The 164 tests run without a webcam or a dataset.

**Why is everything CPU-only?**
The laptop has no NVIDIA GPU, and rPPG does not need one: it is filtering and a Fourier transform on
a few numbers per frame. The only heavy step is face tracking, which MediaPipe runs on the CPU at
about 20 frames/second. That speed shapes the plan: only the first 15 s of each video is processed,
and batch runs use subsets (plan.md section 7).

## What is and isn't measured

**What have you actually shown so far?**
That the algorithm recovers a known heart rate from synthetic signals, tolerates gaps, jittered
timestamps and a lighting flicker, and that the full pipeline runs on a real face image through the
real tracker. What is *not* yet shown is accuracy against real recordings: UBFC-rPPG is not
downloaded and no own clips are recorded. The result tables say "pending" until the evaluation
script has produced real numbers. The plan's target is POS within about 5 bpm on UBFC.

**What are the weaknesses?**
Short clips give coarse heart rates, and SNR changes with clip length (above). Dark skin, head
motion and heavy video compression weaken the pulse, and none of those effects is quantified yet.
The regions are hulls, not true skin masks.
