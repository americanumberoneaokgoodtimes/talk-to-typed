import numpy as np
import pytest
from unittest import mock

from stt import audio


def test_normalize():
    # Empty array
    empty = np.array([], dtype=np.float32)
    assert len(audio.normalize(empty)) == 0

    # All zeros
    zeros = np.zeros(100, dtype=np.float32)
    res_zeros = audio.normalize(zeros)
    assert np.all(res_zeros == 0.0)

    # Signal normalization
    sig = np.array([-0.5, 0.25, 0.1], dtype=np.float32)
    normed = audio.normalize(sig, peak=0.7)
    assert np.isclose(np.abs(normed).max(), 0.7)


def test_agc():
    agc = audio.AGC(target=0.1, max_gain=20.0, noise_cap=0.05)
    
    # Process silence / low noise
    silence = np.random.normal(0, 0.001, 512).astype(np.float32)
    out1 = agc(silence)
    assert out1.shape == silence.shape

    # Process louder speech burst
    speech = np.random.normal(0, 0.08, 512).astype(np.float32)
    out2 = agc(speech)
    assert out2.shape == speech.shape
    # Check bounds
    assert np.all(out2 >= -1.0) and np.all(out2 <= 1.0)


def test_noise_profile():
    # 0.5s of noise at 16kHz
    clip = np.random.normal(0, 0.02, 8000).astype(np.float32)
    np_prof = audio.NoiseProfile(clip)
    assert np_prof.rms > 0
    assert len(np_prof.mag) > 0

    # Test spectral subtraction
    test_sig = np.random.normal(0, 0.05, 8000).astype(np.float32)
    cleaned = np_prof.subtract(test_sig)
    assert len(cleaned) == len(test_sig)
    assert cleaned.dtype == np.float32


def test_decode_missing_file():
    with pytest.raises(RuntimeError):
        audio.decode("/nonexistent/file/path.wav")
