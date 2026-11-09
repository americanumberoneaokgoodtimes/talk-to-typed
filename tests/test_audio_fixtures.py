"""Tests verifying processing across test_audio/*.wav fixtures."""
from pathlib import Path
import pytest
from stt import asr, audio, config, robust

AUDIO_DIR = Path(__file__).resolve().parent.parent / "test_audio"


def test_decode_fixtures():
    for wav in AUDIO_DIR.glob("*.wav"):
        samples = audio.decode(str(wav))
        assert samples is not None
        assert len(samples) > 0
        dur = len(samples) / audio.SR
        assert dur > 1.0


def test_profile_fixtures():
    for wav in AUDIO_DIR.glob("*.wav"):
        samples = audio.decode(str(wav))
        prof = audio.profile_audio(samples)
        assert prof.hpf_cutoff_hz in (80.0, 95.0, 115.0, 130.0)
        assert prof.blank_penalty >= 0.30
        assert prof.snr_db is not None


def test_transcribe_clean_fixture():
    cfg = config.load()
    models_dir = Path(cfg["models_dir"])
    if not (models_dir / cfg["offline_model"]).is_dir():
        pytest.skip("Models not installed")

    clean_wav = AUDIO_DIR / "clean.wav"
    samples = audio.decode(str(clean_wav))
    rec = asr.get_offline(cfg)
    triples = robust.transcribe(rec, samples, cfg, level=0)
    assert len(triples) >= 1
