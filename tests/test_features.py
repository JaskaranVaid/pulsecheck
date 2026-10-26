"""Per-video features: group A (pulse presence), group D (quality) and the feature-table builder."""

import importlib.util
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pulsecheck import features
from pulsecheck.config import SETTINGS
from pulsecheck.datasets.manifest import COLUMNS, VideoItem, write_manifest
from pulsecheck.extract import cache_path_for
from pulsecheck.pipeline import analyze
from pulsecheck.regions import ALL_REGIONS
from synth import synth_region_traces

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SEEDS = range(10)


def _analysis(traces):
    return analyze(
        traces, "pos", window_s=SETTINGS.feature_window_s, step_s=SETTINGS.feature_step_s
    )


def _median_feature(kind: str, name: str) -> float:
    vals = [features.group_a(_analysis(synth_region_traces(kind, seed=s)))[name] for s in SEEDS]
    return float(np.median(vals))


def test_group_a_real_vs_noise():
    real = _median_feature("real", "a_snr_face")
    noise = _median_feature("noise", "a_snr_face")
    assert real > noise + 3.0, (real, noise)


def test_group_a_cannot_separate_inherited_from_real():
    # Documents the limitation of the classic baseline: a pulse copied uniformly into every
    # region looks as "present" as a genuine one, so group A alone cannot catch it.
    real = _median_feature("real", "a_snr_face")
    inherited = _median_feature("inherited", "a_snr_face")
    assert abs(real - inherited) < 1.5, (real, inherited)


def test_synth_region_traces_is_deterministic_and_well_formed():
    a = synth_region_traces("real", seed=3)
    b = synth_region_traces("real", seed=3)
    c = synth_region_traces("real", seed=4)
    np.testing.assert_array_equal(a.rgb, b.rgb)
    assert not np.array_equal(a.rgb, c.rgb)
    assert a.rgb.shape == (450, len(ALL_REGIONS), 3) and a.rgb.dtype == np.float32
    assert a.region_names == ALL_REGIONS
    assert a.landmarks.shape == (450, 478, 2) and a.landmarks.dtype == np.float16
    assert np.isfinite(a.rgb).all() and a.face_found.all()
    with pytest.raises(ValueError):
        synth_region_traces("bogus")


def test_features_nan_safe_when_region_unusable():
    traces = synth_region_traces("real", seed=0)
    names = list(traces.region_names)
    traces.rgb[:, names.index("control"), :] = np.nan
    traces.rgb[:, names.index("cheek_left"), :] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = features.video_features(traces, groups="AD")
    assert all(np.isfinite(out[k]) for k in features.FEATURE_GROUPS["A"])
    assert set(out) == set(features.FEATURE_GROUPS["A"]) | set(features.FEATURE_GROUPS["D"])


def test_group_a_all_nan_when_no_valid_window():
    # A face with no valid windows gives NaN (not an exception, not a warning).
    analysis = _analysis(synth_region_traces("real", seed=0))
    for res in [analysis.face, *analysis.regions.values()]:
        res.series.bpm[:] = np.nan
        res.series.snr_db[:] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = features.group_a(analysis)
    assert np.isnan(out["a_snr_face"]) and np.isnan(out["a_snr_regions_max"])
    assert np.isnan(out["a_hr_std_face"]) and np.isnan(out["a_hr_in_rest_range"])
    assert out["a_valid_frac"] == 1.0


def test_group_d_values():
    traces = synth_region_traces("real", seed=0)
    out = features.group_d(traces, _analysis(traces))
    assert out["d_duration_s"] == pytest.approx(15.0, abs=0.05)
    assert out["d_face_found_ratio"] == 1.0
    assert out["d_face_width_px"] == 150.0
    assert out["d_motion"] == pytest.approx(0.01)
    assert out["d_n_windows"] == 12
    assert 100 < out["d_brightness"] < 160  # skin green channel, DC about 130


def test_feature_names_are_grouped_and_prefixed():
    assert list(features.FEATURE_GROUPS) == ["A", "B", "C", "D"]
    for letter, names in features.FEATURE_GROUPS.items():
        assert all(re.fullmatch(f"{letter.lower()}_[a-z0-9_]+", n) for n in names)
    traces = synth_region_traces("real", seed=0)
    out = features.video_features(traces, groups="AD")
    assert list(out) == [*features.FEATURE_GROUPS["A"], *features.FEATURE_GROUPS["D"]]
    assert features.group_a(_analysis(traces)).keys() == set(features.FEATURE_GROUPS["A"])


def test_video_features_rejects_unfinished_and_unknown_groups():
    traces = synth_region_traces("real", seconds=5.0)
    with pytest.raises(NotImplementedError):
        features.video_features(traces, groups="AB")
    with pytest.raises(ValueError):
        features.video_features(traces, groups="AX")


def test_video_features_propagates_analyze_error():
    with pytest.raises(ValueError):
        features.video_features(synth_region_traces("real", seconds=2.0), groups="AD")


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "build_features_script", SCRIPTS / "build_features.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _item(video: Path, label: int) -> VideoItem:
    rel = f"videos/{video.name}"
    return VideoItem("ffpp", video, rel, label, f"g{video.name}", "k", "c23", "train")


def test_build_features_skips_uncached_and_reports(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("PULSECHECK_DATA", str(tmp_path / "data"))
    root = tmp_path / "ds root"
    specs = [  # file name, label, what the cache holds
        ("real.avi", 0, synth_region_traces("real", seed=1)),
        ("fake.avi", 1, synth_region_traces("inherited", seed=2)),
        ("uncached.avi", 1, None),
        ("corrupt.avi", 1, "corrupt"),
        ("short.avi", 0, synth_region_traces("real", seconds=2.0)),
    ]
    items = []
    for name, label, cached in specs:
        video = root / "videos" / name
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"placeholder, never decoded: " + name.encode())
        cache = cache_path_for(video, root)
        if cached == "corrupt":
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(b"garbage")
        elif cached is not None:
            cached.save(cache)
        items.append(_item(video, label))
    manifest = tmp_path / "m.csv"
    write_manifest(items, manifest)

    assert _load_script().main(["--manifest", str(manifest), "--groups", "AD"]) == 0

    df = pd.read_csv(tmp_path / "data" / "features" / "m.csv")
    expected = [
        *COLUMNS,
        *features.FEATURE_GROUPS["A"],
        *features.FEATURE_GROUPS["D"],
    ]
    assert list(df.columns) == expected
    assert list(df["rel_path"]) == ["videos/real.avi", "videos/fake.avi"]
    assert df["path"].iloc[0] == (root / "videos" / "real.avi").as_posix()
    assert df["d_n_windows"].tolist() == [12, 12]

    out = capsys.readouterr().out
    assert "2 written" in out
    assert re.search(r"uncached\D+1", out)
    assert re.search(r"load error\D+1", out)
    assert re.search(r"analyze error\D+1", out)
    assert "a_snr_face" in out


def test_build_features_custom_out_and_no_rows(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("PULSECHECK_DATA", str(tmp_path / "data"))
    video = tmp_path / "ds" / "videos" / "a.avi"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    manifest = tmp_path / "m.csv"
    write_manifest([_item(video, 0)], manifest)
    out = tmp_path / "o" / "f.csv"
    assert _load_script().main(["--manifest", str(manifest), "--out", str(out)]) == 1
    assert not out.exists()
    assert "uncached" in capsys.readouterr().err
