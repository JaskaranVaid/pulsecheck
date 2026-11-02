"""Synthetic RGB traces with a known pulse, and per-video feature tables built from them,
shared by the rPPG/DSP, feature and detector tests.

Import as a plain module (``tests/`` has no ``__init__.py``)::

    from synth import synth_rgb, synth_region_traces, synth_ffpp_table
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from pulsecheck.extract import RegionTraces
from pulsecheck.features import video_features
from pulsecheck.regions import ALL_REGIONS, CONTROL

DC = np.array([180.0, 130.0, 110.0])  # mean skin colour (R, G, B)
BLOOD_VOLUME = np.array([0.33, 0.77, 0.53])  # relative pulse amplitude per channel


def synth_rgb(
    bpm: float,
    seconds: float,
    fs: float = 30.0,
    pulse_amp: float = 0.002,
    intensity_noise: tuple[float, float] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(t, rgb)`` with ``rgb`` of shape ``(T, 3)``.

    ``rgb = DC * (1 + pulse_amp * v * w(t)) * (1 + a_n * sin(2 pi f_n t))`` plus Gaussian sensor
    noise (sigma 0.05). ``w(t) = sin(2 pi f0 t) + 0.3 sin(4 pi f0 t)``. ``intensity_noise =
    (f_n, a_n)`` adds a lighting flicker that scales all channels equally.
    """
    t = np.arange(round(seconds * fs)) / fs
    f0 = bpm / 60.0
    w = np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(4 * np.pi * f0 * t)
    rgb = DC * (1 + pulse_amp * np.outer(w, BLOOD_VOLUME))
    if intensity_noise is not None:
        f_n, a_n = intensity_noise
        rgb = rgb * (1 + a_n * np.sin(2 * np.pi * f_n * t))[:, None]
    rgb = rgb + np.random.default_rng(seed).normal(0.0, 0.05, rgb.shape)
    return t, rgb


# --- synthetic per-region videos (``synth_region_traces``) -------------------------------------
#
# A "video" here is the output of face tracking: the mean RGB of each facial region per frame.
# Three kinds, differing only in where the heartbeat comes from:
#
# "real"      One heartbeat drives every skin region, but as in a living face it arrives at
#             slightly different times (REAL_LAGS_S, the pulse wave travels from the forehead
#             down the face) and with different strength (REAL_AMPS, cheeks perfuse most).
#             The control (background) region carries no pulse.
# "inherited" A deepfake whose face carries a heartbeat copied from the driver/source video and
#             pasted uniformly: the same pulse, zero lag and equal amplitude in every skin
#             region. Pulse *presence* looks just as strong as in "real"; only the consistency
#             across regions (lag, amplitude ratios) differs.
# "noise"     No heartbeat. Each skin region gets its own independent noise confined to the
#             heart-rate band, as strong as the pulse (a generator-made face with random
#             colour flicker). No common component across regions.
#
# All kinds share the skin colour, sensor noise and drift; the seed controls the pulse phase,
# per-region skin brightness, per-region amplitude jitter, sensor noise and the noise-kind signal.
# Each region draws from its own child generator (indexed by its position in ALL_REGIONS) so the
# values of one region do not depend on how many other regions exist; the shared pulse phase and
# the shared illumination drift have their own children too.

REAL_LAGS_S: dict[str, float] = {  # seconds the pulse arrives after the forehead (kind "real")
    "forehead_left": 0.0,
    "forehead_center": 0.0,
    "forehead_right": 0.0,
    "cheek_left": 0.03,
    "cheek_right": 0.03,
    "nose": 0.02,
    "chin": 0.05,
}
REAL_AMPS: dict[str, float] = {  # pulse strength relative to the forehead (kind "real")
    "forehead_left": 1.0,
    "forehead_center": 1.0,
    "forehead_right": 1.0,
    "cheek_left": 1.3,
    "cheek_right": 1.3,
    "nose": 0.8,
    "chin": 0.7,
}
PULSE_AMP = 0.002  # relative colour modulation of the pulse at relative amplitude 1.0 (as synth_rgb)
SENSOR_SIGMA = 0.1  # per-channel, per-frame Gaussian sensor noise (colour levels, 0-255 scale)
DC_JITTER = 0.05  # per-region skin brightness varies uniformly within +-5 % of DC
AMP_JITTER = 0.10  # per-region pulse amplitude varies uniformly within +-10 % (kind "real")
DRIFT_AMP = 0.003  # shared slow illumination drift: relative amplitude, all channels equally
DRIFT_HZ = (0.05, 0.3)  # drift frequencies (below the heart-rate band)
NOISE_BAND_HZ = (0.7, 3.3)  # kind "noise": the heart-rate band, SETTINGS.band
KINDS = ("real", "inherited", "noise")
_SHARED_STREAM = len(ALL_REGIONS)  # child generator for the shared pulse phase
_DRIFT_STREAM = len(ALL_REGIONS) + 1  # child generator for the shared illumination drift
_WAVE_STD = float(np.sqrt(0.5 + 0.3**2 / 2))  # std of w(t) = sin(x) + 0.3 sin(2x)


def _child(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


def _band_noise(rng: np.random.Generator, n: int, fs: float) -> np.ndarray:
    """Unit-variance white noise band-limited to ``NOISE_BAND_HZ`` (zero-phase), then rescaled
    to the pulse waveform's std so noise and pulse have the same power."""
    pad = 2 * round(fs)  # margin that absorbs the filter's edge transients
    sos = butter(3, NOISE_BAND_HZ, btype="bandpass", fs=fs, output="sos")
    x = sosfiltfilt(sos, rng.normal(size=n + 2 * pad))[pad : pad + n]
    return x / x.std() * _WAVE_STD


def synth_region_traces(
    kind: str,
    seconds: float = 15.0,
    fs: float = 30.0,
    seed: int = 0,
    bpm: float = 72.0,
) -> RegionTraces:
    """A ``RegionTraces`` over ``ALL_REGIONS`` with no face loss; see the block comment above.

    ``rgb[t, r] = DC_r * (1 + PULSE_AMP * amp_r * BLOOD_VOLUME * w(t - lag_r) + drift(t)) + noise``
    with ``w(x) = sin(2 pi f0 x + phi) + 0.3 sin(4 pi f0 x + 2 phi)``, ``f0 = bpm / 60``. For
    kind "noise" ``w`` is replaced by independent band-limited noise per region. The control
    region never carries a pulse. Frames are ``round(seconds * fs)``, timestamps ``k / fs``;
    ``face_width_px`` is 150, ``motion`` 0.01, landmarks are zero placeholders (float16).
    Raises ``ValueError`` for an unknown kind.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; choose from {KINDS}")
    n = round(seconds * fs)
    t = np.arange(n) / fs
    f0 = bpm / 60.0
    phi = _child(seed, _SHARED_STREAM).uniform(0, 2 * np.pi)

    def wave(lag: float) -> np.ndarray:
        x = 2 * np.pi * f0 * (t - lag) + phi
        return np.sin(x) + 0.3 * np.sin(2 * x)

    drift_rng = _child(seed, _DRIFT_STREAM)
    f_d = drift_rng.uniform(*DRIFT_HZ, size=2)
    p_d = drift_rng.uniform(0, 2 * np.pi, size=2)
    drift = DRIFT_AMP * sum(np.sin(2 * np.pi * f * t + p) for f, p in zip(f_d, p_d)) / 2

    rgb = np.empty((n, len(ALL_REGIONS), 3))
    for r, name in enumerate(ALL_REGIONS):
        rng = _child(seed, r)  # draw order below is fixed so one region's stream never shifts
        dc = DC * (1 + rng.uniform(-DC_JITTER, DC_JITTER))
        amp_jitter = 1 + rng.uniform(-AMP_JITTER, AMP_JITTER)
        sensor = rng.normal(0.0, SENSOR_SIGMA, (n, 3))
        if name == CONTROL:
            pulse = np.zeros(n)
        elif kind == "real":
            pulse = REAL_AMPS[name] * amp_jitter * wave(REAL_LAGS_S[name])
        elif kind == "inherited":
            pulse = wave(0.0)
        else:  # "noise"
            pulse = _band_noise(rng, n, fs)
        rgb[:, r, :] = dc * (1 + PULSE_AMP * np.outer(pulse, BLOOD_VOLUME) + drift[:, None])
        rgb[:, r, :] += sensor

    return RegionTraces(
        rgb=rgb.astype(np.float32),
        t=t,
        region_names=ALL_REGIONS,
        source=f"synthetic:{kind}",
        native_fps=fs,
        face_found=np.ones(n, bool),
        face_width_px=np.full(n, 150.0, np.float32),
        motion=np.full(n, 0.01, np.float32),
        landmarks=np.zeros((n, 478, 2), np.float16),
    )


# --- synthetic per-video feature tables (detector tests) ---------------------------------------
#
# Small stand-ins for the tables ``scripts/build_features.py`` writes: the manifest fields
# followed by the features of ``video_features`` computed on ``synth_region_traces``. The
# FF++-like table follows the real layout: one leakage group ``ffpp:{a}_{b}`` per pair, holding
# one real ("original") and one fake per manipulation, with whole groups assigned to a split.

TABLE_SECONDS = 8.0  # clip length of the synthetic videos behind the feature tables (s)
# FF++-like manipulations and the synthetic kind that stands in for each
FFPP_FAKES: dict[str, str] = {"Deepfakes": "inherited", "FaceSwap": "inherited",
                              "NeuralTextures": "noise"}


@functools.cache
def _video_features(kind: str, seed: int) -> tuple[tuple[str, float], ...]:
    """Features of one synthetic video, cached across tests (immutable so callers cannot edit)."""
    feats = video_features(synth_region_traces(kind, seconds=TABLE_SECONDS, seed=seed))
    return tuple(feats.items())


def _row(dataset: str, rel_path: str, label: int, group: str, kind: str, compression: str,
         split: str, synth_kind: str, seed: int) -> dict:
    manifest = {
        "dataset": dataset,
        "path": f"/synthetic/{dataset}/{rel_path}",
        "rel_path": rel_path,
        "label": label,
        "group": group,
        "kind": kind,
        "compression": compression,
        "split": split,
    }
    return {**manifest, **dict(_video_features(synth_kind, seed))}


def synth_ffpp_table(
    pairs: dict[str, int] | None = None, compression: str = "c23", seed0: int = 0
) -> pd.DataFrame:
    """FF++-like feature table: ``pairs`` maps split -> number of pairs (groups) in it.

    Each group holds one real (synth "real") and one fake per ``FFPP_FAKES`` entry. Every video
    gets its own seed ``seed0 + running index``, so no two rows share a trace.
    """
    pairs = pairs or {"train": 6, "val": 3, "test": 3}
    rows = []
    seed = seed0
    pair = 0
    for split, n in pairs.items():
        for _ in range(n):
            a, b = 2 * pair, 2 * pair + 1
            group = f"ffpp:{a:03d}_{b:03d}"
            rows.append(_row("ffpp", f"original/{a:03d}.mp4", 0, group, "original",
                             compression, split, "real", seed))
            seed += 1
            for kind, synth_kind in FFPP_FAKES.items():
                rows.append(_row("ffpp", f"{kind}/{a:03d}_{b:03d}.mp4", 1, group, kind,
                                 compression, split, synth_kind, seed))
                seed += 1
            pair += 1
    return pd.DataFrame(rows)


def synth_celebdf_table(n_real: int = 6, n_fake: int = 6, seed0: int = 5000) -> pd.DataFrame:
    """Celeb-DF-like test-only table: reals (synth "real") and Celeb-synthesis (synth
    "inherited"), one group per video, split "test"."""
    rows = []
    for i in range(n_real):
        rows.append(_row("celebdf", f"Celeb-real/id{i}_0000.mp4", 0, f"celebdf:id{i}",
                         "Celeb-real", "", "test", "real", seed0 + i))
    for i in range(n_fake):
        rows.append(_row("celebdf", f"Celeb-synthesis/id{100 + i}_id{i}_0000.mp4", 1,
                         f"celebdf:id{100 + i}", "Celeb-synthesis", "", "test", "inherited",
                         seed0 + n_real + i))
    return pd.DataFrame(rows)
