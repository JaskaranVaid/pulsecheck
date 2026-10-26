"""Central settings and data-directory helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    fs: float = 30.0  # uniform analysis rate (Hz)
    band: tuple[float, float] = (0.7, 3.3)  # heart-rate band (Hz): 42-198 bpm
    filter_order: int = 3  # Butterworth order (applied zero-phase)
    window_s: float = 8.0  # HR estimation window (s)
    step_s: float = 1.0  # HR window step (s)
    feature_window_s: float = 4.0  # detector feature window (s); identical for every dataset
    feature_step_s: float = 1.0  # detector feature window step (s)
    pos_window_s: float = 1.6  # POS algorithm window (s)
    max_gap_s: float = 0.5  # longest gap that may be interpolated (s)
    min_seconds: float = 3.0  # shortest analysable clip (s)
    max_seconds: float = 15.0  # default extraction cap (s)
    min_region_pixels: int = 20  # minimum pixels for a usable region


SETTINGS = Settings()


def data_dir() -> Path:
    """Root for models, caches and datasets. Kept outside the repo (and OneDrive)."""
    env = os.environ.get("PULSECHECK_DATA")
    path = Path(env) if env else Path.home() / "pulsecheck-data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    path = data_dir() / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path
