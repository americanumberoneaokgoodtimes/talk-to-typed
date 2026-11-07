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


def test_speech_formant_filter():
    assert len(audio.speech_formant_filter(np.array([], dtype=np.float32), boost_db=4.0)) == 0

    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Formant tone at 1000 Hz, sub-bass at 50 Hz, and high consonant tone at 5000 Hz
    s_50 = 0.2 * np.sin(2 * np.pi * 50 * t).astype(np.float32)
    s_1000 = 0.2 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    s_5000 = 0.2 * np.sin(2 * np.pi * 5000 * t).astype(np.float32)
    mix = s_50 + s_1000 + s_5000

    boosted = audio.speech_formant_filter(mix, boost_db=6.0)
    assert len(boosted) == len(mix)

    fft_mix = np.abs(np.fft.rfft(mix))
    fft_boosted = np.abs(np.fft.rfft(boosted))
    freqs = np.fft.rfftfreq(len(mix), 1.0 / sr)

    idx_50 = np.argmin(np.abs(freqs - 50.0))
    idx_1000 = np.argmin(np.abs(freqs - 1000.0))
    idx_5000 = np.argmin(np.abs(freqs - 5000.0))

    # 50 Hz unchanged (ratio close to 1.0)
    assert np.isclose(fft_boosted[idx_50] / fft_mix[idx_50], 1.0, atol=0.05)
    # 1000 Hz boosted by ~6 dB (factor of ~2.0)
    assert np.isclose(fft_boosted[idx_1000] / fft_mix[idx_1000], 10.0 ** (6.0 / 20.0), rtol=0.05)
    # 5000 Hz boosted by consonant shelf (~2.5 dB, factor of ~1.33)
    assert np.isclose(fft_boosted[idx_5000] / fft_mix[idx_5000], 10.0 ** (2.5 / 20.0), rtol=0.05)


def test_noise_bounded_agc():
    assert len(audio.noise_bounded_agc(np.array([], dtype=np.float32), noise_floor_rms=0.001)) == 0

    sr = 16000
    # Quiet speech-like chunk
    quiet_speech = (0.01 * np.sin(np.linspace(0, 100, sr))).astype(np.float32)
    agc_out = audio.noise_bounded_agc(quiet_speech, noise_floor_rms=0.0005, target_speech_rms=0.15, max_noise_floor_rms=0.025)
    # Scaled up
    assert np.max(np.abs(agc_out)) > np.max(np.abs(quiet_speech))

    # Pure noise chunk with very low noise floor
    hiss = np.random.normal(0, 0.002, sr).astype(np.float32)
    agc_hiss = audio.noise_bounded_agc(hiss, noise_floor_rms=0.002, target_speech_rms=0.18, max_noise_floor_rms=0.025)
    # Guaranteed noise bounded: RMS cannot explode past max_noise_floor_rms * 1.5
    hiss_rms = float(np.sqrt(np.mean(agc_hiss ** 2)))
    assert hiss_rms <= 0.035


def test_spectral_crossover_blend():
    assert len(audio.spectral_crossover_blend(np.array([], dtype=np.float32), np.array([], dtype=np.float32))) == 0

    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Low-band vowel tone (500 Hz) and high-band consonant tone (5000 Hz)
    vowel = np.sin(2 * np.pi * 500 * t).astype(np.float32)
    consonant = np.sin(2 * np.pi * 5000 * t).astype(np.float32)

    # Denoiser has clean low band, but smeared high band (zero consonant)
    den = vowel
    # HP has both vowel and crisp consonant
    hp = vowel + consonant

    fused = audio.spectral_crossover_blend(den, hp, sr=sr, f_split_low=3200.0, f_split_high=4200.0, hp_high_weight=0.70)
    assert len(fused) == len(den)

    fft_fused = np.abs(np.fft.rfft(fused))
    freqs = np.fft.rfftfreq(len(fused), 1.0 / sr)
    idx_500 = np.argmin(np.abs(freqs - 500.0))
    idx_5000 = np.argmin(np.abs(freqs - 5000.0))

    # Low-band 500 Hz is preserved
    assert fft_fused[idx_500] > 100.0
    # High-band 5000 Hz consonant is restored by crossover from HP (>50% of consonant energy)
    assert fft_fused[idx_5000] > 50.0


def test_dynamic_pregate_and_hfc():
    sr = 16000
    t = np.linspace(0, 4.0, sr * 4, endpoint=False)
    # 2 seconds of quiet noise (RMS ~0.001), 2 seconds of louder noise (RMS ~0.01)
    noise_part1 = np.random.normal(0, 0.001, sr * 2).astype(np.float32)
    noise_part2 = np.random.normal(0, 0.01, sr * 2).astype(np.float32)
    full_noise = np.concatenate([noise_part1, noise_part2])

    profile = audio.compute_dynamic_pregate_profile(full_noise, sr=sr, frame_sec=0.25, local_win_sec=2.0)
    assert len(profile) == len(full_noise) // (sr // 4)
    # Part 1 local floor is lower than part 2 local floor
    assert profile[2] < profile[-2]

    # Faint whisper with strong harmonic formant structure (formant at 1000 Hz)
    whisper = 0.0015 * np.sin(2 * np.pi * 1000 * t[:sr]).astype(np.float32)
    # Pass check against part 1 floor (0.001)
    assert audio.evaluate_window_pregate(whisper, local_floor_rms=0.001, sr=sr) is True


def test_detect_micro_bursts():
    sr = 16000
    # 5.0s window with mostly quiet noise and a 0.6s burst centered at 2.5s
    samples = np.random.normal(0, 0.001, sr * 5).astype(np.float32)
    t_burst = np.linspace(0, 0.6, int(0.6 * sr), endpoint=False)
    burst = 0.02 * np.sin(2 * np.pi * 800 * t_burst).astype(np.float32)
    start_b = int(2.2 * sr)
    samples[start_b : start_b + len(burst)] += burst

    bursts = audio.detect_micro_bursts(samples, sr=sr, sub_sec=0.5, par_threshold=2.0)
    assert len(bursts) == 1
    # Micro window spans around the burst
    m_start, m_end = bursts[0]
    assert m_start <= start_b <= m_end


def test_adaptive_spectral_gate():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    tone = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    noise = np.random.normal(0, 0.05, sr).astype(np.float32)
    noisy = tone + noise

    denoised = audio.adaptive_spectral_gate(noisy, sr=sr)
    assert len(denoised) == len(noisy)
    assert denoised.dtype == np.float32
    # Denoised signal should have reduced residual noise
    assert np.var(denoised) > 0.0


def test_harmonic_pitch_vad():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    # 200 Hz harmonic tone simulating voiced speech pitch
    voiced = (0.3 * np.sin(2 * np.pi * 200.0 * t) + 0.15 * np.sin(2 * np.pi * 400.0 * t)).astype(np.float32)
    noise = np.random.normal(0, 0.02, sr).astype(np.float32)
    stimulus = voiced + noise

    active_frames = audio.harmonic_pitch_vad(stimulus, sr=sr)
    assert len(active_frames) > 0
    # Voiced periodic tone should trigger harmonic pitch VAD frames
    assert np.sum(active_frames) > 0


def test_multiband_formant_agc():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    # Faint vowel formant at 800 Hz
    faint_formant = (0.01 * np.sin(2 * np.pi * 800.0 * t)).astype(np.float32)
    # Sub-rumble at 40 Hz
    rumble = (0.10 * np.sin(2 * np.pi * 40.0 * t)).astype(np.float32)
    s = faint_formant + rumble

    boosted = audio.multiband_formant_agc(s, sr=sr)
    assert len(boosted) == len(s)
    assert boosted.dtype == np.float32
    # Formant should be lifted relative to rumble
    assert np.max(np.abs(boosted)) > np.max(np.abs(faint_formant))


def test_suppress_impulsive_clicks():
    # Empty / short signal handling
    empty_res, c0 = audio.suppress_impulsive_clicks(np.array([], dtype=np.float32))
    assert len(empty_res) == 0
    assert c0 == 0

    sr = 16000
    t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False, dtype=np.float32)
    clean_sine = (0.05 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    # Clean signal should have zero clicks removed
    clean_out, c_none = audio.suppress_impulsive_clicks(clean_sine, sr=sr)
    assert c_none == 0
    assert np.allclose(clean_out, clean_sine, atol=1e-5)

    # Insert 3 sharp mechanical keyboard click spikes (width = 4 samples, amplitude = 0.6)
    clicky = clean_sine.copy()
    click_positions = [1000, 3000, 5000]
    for pos in click_positions:
        clicky[pos : pos + 4] += np.array([0.5, 0.6, -0.4, 0.2], dtype=np.float32)

    assert np.max(np.abs(clicky)) >= 0.55
    declicked, count = audio.suppress_impulsive_clicks(clicky, sr=sr, threshold_sigma=4.0)

    # All 3 clicks should be detected and inpainted
    assert count >= 3
    # Peak amplitude should be substantially suppressed
    assert np.max(np.abs(declicked)) < 0.20


def test_whisper_formant_filter():
    sr = 16000
    t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False, dtype=np.float32)

    # Empty / short signal handling
    empty_res = audio.whisper_formant_filter(np.array([], dtype=np.float32))
    assert len(empty_res) == 0

    # 2500 Hz whisper formant tone
    whisper_tone = (0.05 * np.sin(2 * np.pi * 2500.0 * t)).astype(np.float32)
    boosted = audio.whisper_formant_filter(whisper_tone, sr=sr, force_whisper=True)
    assert len(boosted) == len(whisper_tone)
    assert boosted.dtype == np.float32
    gain_db = 20 * np.log10(np.std(boosted) / np.std(whisper_tone))
    assert 5.0 <= gain_db <= 8.5

    # 100 Hz rumble tone should be attenuated
    rumble_tone = (0.05 * np.sin(2 * np.pi * 100.0 * t)).astype(np.float32)
    cut = audio.whisper_formant_filter(rumble_tone, sr=sr, force_whisper=True)
    cut_db = 20 * np.log10(np.std(cut) / np.std(rumble_tone))
    assert cut_db <= -6.0


def test_detect_formant_modulation_spans():
    sr = 16000
    t = np.linspace(0, 4.0, int(4.0 * sr), endpoint=False, dtype=np.float32)
    noise = (0.005 * np.random.randn(len(t))).astype(np.float32)

    # Short burst at 2.0s in formant band (2000 Hz)
    burst = (0.15 * np.sin(2 * np.pi * 2000.0 * t[32000:32000 + 4000])).astype(np.float32)
    sig = noise.copy()
    sig[32000:32000 + 4000] += burst

    spans = audio.detect_formant_modulation_spans(sig, sr=sr, min_smi=1.80, span_sec=2.2)
    assert len(spans) >= 1
    # Check that burst at sample 32000 (t=2.0s) is covered by one of the spans
    covered = any(s <= 32000 <= e for s, e in spans)
    assert covered


def test_streaming_declicker():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    clean_tone = (0.05 * np.sin(2 * np.pi * 500.0 * t)).astype(np.float32)

    spike = np.array([0.5, 0.6, -0.4, 0.2], dtype=np.float32)
    sig = clean_tone.copy()
    # Click at sample 3000 and boundary click at sample 511
    sig[3000:3004] += spike
    sig[510:514] += spike

    chunk_size = 512
    margin = 64
    declicker = audio.StreamingDeclicker(sr=sr, margin_samples=margin, threshold_sigma=4.0)

    streamed_chunks = []
    for i in range(0, len(sig), chunk_size):
        chunk = sig[i : i + chunk_size]
        streamed_chunks.append(declicker.process(chunk))

    out = np.concatenate(streamed_chunks)
    assert len(out) == len(sig)
    assert out.dtype == np.float32
    assert declicker.clicks_detected >= 2

    # Verify peak at 3000 + 64 is suppressed
    assert np.max(np.abs(out[3050:3080])) < 0.20
    # Verify boundary click at 510 + 64 is suppressed
    assert np.max(np.abs(out[560:590])) < 0.20



