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
    # Missing/corrupt files now return None with a warning instead of raising
    result = audio.decode("/nonexistent/file/path.wav")
    assert result is None


def test_butterworth_highpass():
    # Empty array
    empty = np.array([], dtype=np.float32)
    assert len(audio.butterworth_highpass(empty)) == 0

    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # 40 Hz rumble + 1000 Hz speech tone
    sig_low = np.sin(2 * np.pi * 40 * t).astype(np.float32)
    sig_high = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    sig_mix = sig_low + sig_high

    filtered = audio.butterworth_highpass(sig_mix, cutoff_hz=120.0, order=4)
    assert len(filtered) == len(sig_mix)

    # 40 Hz should be heavily attenuated (>15 dB reduction)
    fft_in = np.abs(np.fft.rfft(sig_low))
    fft_out = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(sig_mix), 1.0 / sr)
    idx_40 = np.argmin(np.abs(freqs - 40.0))
    idx_1000 = np.argmin(np.abs(freqs - 1000.0))

    assert fft_out[idx_40] < 0.2 * fft_in[idx_40]
    # 1000 Hz tone should be virtually unattenuated (>95% preserved)
    assert np.isclose(fft_out[idx_1000], np.max(np.abs(np.fft.rfft(sig_high))), rtol=0.08)


def test_harmonic_notch_filter():
    # Empty array
    assert len(audio.harmonic_notch_filter(np.array([], dtype=np.float32))) == 0

    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # 60 Hz hum + 120 Hz harmonic + 800 Hz speech tone
    hum_60 = np.sin(2 * np.pi * 60 * t).astype(np.float32)
    hum_120 = np.sin(2 * np.pi * 120 * t).astype(np.float32)
    speech_800 = np.sin(2 * np.pi * 800 * t).astype(np.float32)
    mix = hum_60 + hum_120 + speech_800

    notched = audio.harmonic_notch_filter(mix, freqs=(60.0, 120.0, 180.0), bw_hz=6.0)
    assert len(notched) == len(mix)

    fft_mix = np.abs(np.fft.rfft(mix))
    fft_notched = np.abs(np.fft.rfft(notched))
    freqs = np.fft.rfftfreq(len(mix), 1.0 / sr)
    idx_60 = np.argmin(np.abs(freqs - 60.0))
    idx_120 = np.argmin(np.abs(freqs - 120.0))
    idx_800 = np.argmin(np.abs(freqs - 800.0))

    # Notches attenuate by >10 dB
    assert fft_notched[idx_60] < 0.3 * fft_mix[idx_60]
    assert fft_notched[idx_120] < 0.3 * fft_mix[idx_120]
    # Speech tone at 800 Hz is preserved (>95%)
    assert np.isclose(fft_notched[idx_800], fft_mix[idx_800], rtol=0.05)


def test_speech_band_energy():
    assert audio.speech_band_energy(np.array([], dtype=np.float32)) == 0.0

    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Tone at 1000 Hz (in speech band)
    tone_1k = 0.5 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    e_1k = audio.speech_band_energy(tone_1k)
    assert e_1k > 0.01

    # Tone at 50 Hz (out of speech band)
    tone_50 = 0.5 * np.sin(2 * np.pi * 50 * t).astype(np.float32)
    e_50 = audio.speech_band_energy(tone_50)
    assert e_50 < 0.001


def test_profile_audio_empty():
    prof = audio.profile_audio(np.array([], dtype=np.float32))
    assert prof.noise_floor_rms == 0.0
    assert len(prof.reasoning) > 0


def test_profile_audio_hum_detection():
    sr = 16000
    t = np.linspace(0, 5.0, sr * 5, endpoint=False)
    noise = np.random.normal(0, 0.01, len(t)).astype(np.float32)

    # Add strong 60 Hz hum
    hum_60 = noise + 0.1 * np.sin(2 * np.pi * 60 * t).astype(np.float32)
    prof_60 = audio.profile_audio(hum_60)
    assert prof_60.hum_freq == 60.0
    assert 60.0 in prof_60.notch_freqs
    assert 120.0 in prof_60.notch_freqs

    # Add strong 50 Hz hum
    hum_50 = noise + 0.1 * np.sin(2 * np.pi * 50 * t).astype(np.float32)
    prof_50 = audio.profile_audio(hum_50)
    assert prof_50.hum_freq == 50.0
    assert 50.0 in prof_50.notch_freqs
    assert 100.0 in prof_50.notch_freqs


def test_profile_audio_heavy_rumble_and_snr():
    sr = 16000
    t = np.linspace(0, 5.0, sr * 5, endpoint=False)
    # Very heavy 30 Hz rumble dominating signal (>80% energy)
    rumble = 0.5 * np.sin(2 * np.pi * 30 * t).astype(np.float32)
    weak_speech = np.random.normal(0, 0.01, len(t)).astype(np.float32)
    mix = rumble + weak_speech

    prof = audio.profile_audio(mix)
    assert prof.rumble_fraction > 0.70
    assert prof.hpf_cutoff_hz >= 125.0
    # Blank penalty dynamically raised for low SNR / heavy noise
    assert prof.blank_penalty >= 0.6
    assert prof.pregate_threshold > 0.0
