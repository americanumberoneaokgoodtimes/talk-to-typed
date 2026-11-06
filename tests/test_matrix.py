import numpy as np
import pytest
from pathlib import Path

from stt import config, matrix, audio


def test_palette_uniqueness():
    """Verify that PALETTE contains valid and distinct ANSI color codes."""
    codes = [p[0] for p in matrix.PALETTE]
    assert len(codes) >= 10
    assert len(codes) == len(set(codes))
    for code, name in matrix.PALETTE:
        assert code.startswith("\033[")
        assert len(name) > 0


def test_format_ts():
    """Verify format_ts returns MM:SS.mmm timeline intervals."""
    assert matrix.format_ts(0.0) == "00:00.000"
    assert matrix.format_ts(9.42) == "00:09.420"
    assert matrix.format_ts(75.508) == "01:15.508"
    assert matrix.format_ts(3661.123) == "61:01.123"


def test_get_available_combinations():
    """Verify combination generation dynamically inspects config and models_dir."""
    cfg = config.load()
    combos = matrix.get_available_combinations(cfg)
    assert isinstance(combos, list)
    assert len(combos) > 0

    ids = [c["id"] for c in combos]
    assert "parakeet_gtcrn_beam" in ids
    assert "parakeet_hybrid_beam" in ids
    assert "dynamic_ensemble" in ids

    for c in combos:
        assert "id" in c
        assert "name" in c
        assert "type" in c


def test_run_matrix_mocked(monkeypatch):
    """Verify run_matrix executes across combinations and returns structured results."""
    cfg = config.load()
    sr = audio.SR
    # 2 seconds synthetic audio
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
    sig = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    # Mock audio pre-filters and denoisers to run instantaneously
    monkeypatch.setattr(audio, "denoise", lambda s, c, nz=None: s)
    monkeypatch.setattr(audio, "spectral_crossover_blend", lambda d, h: d)
    monkeypatch.setattr(audio, "speech_formant_filter", lambda s, boost_db: s)
    monkeypatch.setattr(audio, "segments", lambda s, c: [(0.2, 1.2, s[int(0.2*sr):int(1.2*sr)])])

    class MockStream:
        def __init__(self):
            self.result = type("Obj", (), {"text": "hello test utterance", "tokens": ["hello"]})()
        def accept_waveform(self, sr, chunk): pass
        def input_finished(self): pass

    class MockOfflineRec:
        def create_stream(self):
            return MockStream()
        def decode_stream(self, st): pass

    class MockOnlineRec:
        def create_stream(self):
            return MockStream()
        def is_ready(self, st): return False
        def decode_stream(self, st): pass
        def get_result(self, st): return "hello test utterance"

    from stt import asr, robust
    monkeypatch.setattr(asr, "get_offline", lambda *args, **kwargs: MockOfflineRec())
    monkeypatch.setattr(robust, "subnoise_transcribe", lambda *args, **kwargs: [(0.2, 1.2, "hello test utterance")])

    res = matrix.run_matrix(sig, cfg, audio_path="mock.wav", threads=1)
    assert len(res) > 0
    assert res[0]["index"] == 1
    assert "name" in res[0]
    assert "color_code" in res[0]
    assert res[0]["num_segments"] >= 1
    assert res[0]["results"][0]["text"] == "hello test utterance"


def test_cli_all_flag_parsing():
    """Verify --all flag is correctly parsed by the file subcommand parser."""
    from stt import cli
    p = cli._build_parser() if hasattr(cli, "_build_parser") else None
    if p is None:
        import argparse
        # Test argparse setup
        ap = argparse.ArgumentParser()
        cli._add_shared_groups(ap, is_file=True)
        args = ap.parse_args(["--all"])
        assert args.all is True
