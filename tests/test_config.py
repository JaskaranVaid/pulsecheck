def test_data_dir_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PULSECHECK_DATA", str(tmp_path / "d"))
    from pulsecheck.config import cache_dir, data_dir, models_dir

    assert data_dir() == tmp_path / "d" and (tmp_path / "d").is_dir()
    assert models_dir() == tmp_path / "d" / "models"
    assert cache_dir() == tmp_path / "d" / "cache"


def test_settings_values():
    from pulsecheck.config import SETTINGS

    assert SETTINGS.fs == 30.0 and SETTINGS.band == (0.7, 3.3)
    assert SETTINGS.window_s == 8.0 and SETTINGS.pos_window_s == 1.6


def test_feature_window_settings():
    from pulsecheck.config import SETTINGS

    assert SETTINGS.feature_window_s == 4.0 and SETTINGS.feature_step_s == 1.0
