import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from stt import config


def test_config_defaults_keys():
    """Verify all critical configuration keys are present in DEFAULTS."""
    assert "threads" in config.DEFAULTS
    assert "models_dir" in config.DEFAULTS
    assert "offline_model" in config.DEFAULTS
    assert "streaming_model" in config.DEFAULTS
    assert "vad_model" in config.DEFAULTS
    assert "robust_denoiser" in config.DEFAULTS
    assert "aggressiveness" in config.DEFAULTS
    assert "windows_mic_device" in config.DEFAULTS
    assert "windows_backend" in config.DEFAULTS


def test_config_load_creates_file_if_missing():
    """When config file does not exist, load() should write template and return defaults."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_cfg = Path(tmpdir) / "stt" / "config.toml"
        with mock.patch.object(config, "CONFIG_PATH", tmp_cfg):
            assert not tmp_cfg.exists()
            cfg = config.load()
            assert tmp_cfg.exists()
            assert cfg["threads"] == config.DEFAULTS["threads"]
            assert cfg["offline_model"] == config.DEFAULTS["offline_model"]


def test_config_load_overrides():
    """load() should read values from existing config.toml."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_cfg = Path(tmpdir) / "stt" / "config.toml"
        tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
        tmp_cfg.write_text('threads = 8\naggressiveness = 3\n', encoding="utf-8")
        with mock.patch.object(config, "CONFIG_PATH", tmp_cfg):
            cfg = config.load()
            assert cfg["threads"] == 8
            assert cfg["aggressiveness"] == 3
            # default unwritten keys still present
            assert cfg["offline_model"] == config.DEFAULTS["offline_model"]
