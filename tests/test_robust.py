import numpy as np
import pytest

from stt import config, robust


def test_auto_tune_settings_short():
    res = robust.auto_tune_settings(10.0)
    assert res["win_sec"] == 5.0
    assert res["step_sec"] == 1.25
    assert res["overlap_pct"] == 75
    assert res["low_hz"] == 120.0
    assert res["blank_penalty"] == 0.7
    assert len(res["reasoning"]) > 0


def test_auto_tune_settings_medium():
    res = robust.auto_tune_settings(60.0)
    assert res["win_sec"] == 5.0
    assert res["step_sec"] == 1.75
    assert res["overlap_pct"] == 65
    assert res["blank_penalty"] == 0.6


def test_auto_tune_settings_long():
    res = robust.auto_tune_settings(180.0)
    assert res["win_sec"] == 6.0
    assert res["step_sec"] == 3.0
    assert res["overlap_pct"] == 50
    assert res["low_hz"] == 130.0
    assert res["blank_penalty"] == 0.5


def test_score_coherence():
    # Empty
    score, reason = robust.score_coherence("", 5.0)
    assert score == 0.0
    assert reason == "empty"

    # Too fast hallucination (>5 wps)
    fast_text = "one two three four five six seven eight nine ten"
    score, reason = robust.score_coherence(fast_text, 1.0)
    assert score < 0
    assert "hallucination" in reason

    # Repetitive single token
    rep_text = "the the the the the the"
    score, reason = robust.score_coherence(rep_text, 3.0)
    assert score <= 0.05
    assert "repetitive token" in reason

    # Repetitive bigram (where single token dom_ratio < 0.5)
    rep_bi = "thank you ok thank you sir"
    score, reason = robust.score_coherence(rep_bi, 3.0)
    assert score <= 0.05
    assert "repetitive phrase" in reason

    # Coherent text
    good_text = "The quick brown fox jumps over the lazy dog"
    score, reason = robust.score_coherence(good_text, 5.0)
    assert score > 1.0
    assert reason == "coherent"


def test_pregain():
    # Empty array
    empty = np.array([], dtype=np.float32)
    assert robust.pregain(empty) == 1.0

    # Very quiet signal (amplified up to max_gain)
    zeros = np.zeros(1024, dtype=np.float32)
    assert robust.pregain(zeros, max_gain=50.0) == 50.0

    # Loud signal (never attenuates below 1.0)
    loud = np.ones(2048, dtype=np.float32) * 0.8
    gain = robust.pregain(loud, target=0.1)
    assert gain == 1.0


def test_variants():
    cfg = dict(config.DEFAULTS)
    for lvl in [1, 2, 3, 4, 5]:
        v = robust._variants(lvl, cfg)
        assert isinstance(v, list)
        assert len(v) >= 1
        for variant in v:
            assert "denoise" in variant
            assert "vad" in variant


def test_auto_tune_settings_adaptive_samples():
    sr = 16000
    t = np.linspace(0, 10.0, sr * 10, endpoint=False)
    # Heavy rumble + weak noise
    sig = 0.4 * np.sin(2 * np.pi * 35 * t).astype(np.float32)
    sig += np.random.normal(0, 0.01, len(t)).astype(np.float32)

    tuned = robust.auto_tune_settings(10.0, samples=sig)
    assert "snr_db" in tuned
    assert "pregate_threshold" in tuned
    assert tuned["low_hz"] >= 120.0
    assert len(tuned["reasoning"]) > 0


def test_score_coherence_natural_repetition():
    # Spanish natural sentence containing 'no' 3 times
    spanish_phrase = "No, no tengo nada, no"
    score, reason = robust.score_coherence(spanish_phrase, 3.0)
    assert score > 0.5
    assert reason == "coherent"

    # Degenerate loop should still be suppressed
    loop_phrase = "no no no no no"
    score_loop, reason_loop = robust.score_coherence(loop_phrase, 3.0)
    assert score_loop <= 0.05
    assert "repetitive token" in reason_loop


def test_subnoise_transcribe_deduplication():
    # Test deduplication logic directly with mock recognizer
    from unittest.mock import MagicMock
    mock_rec = MagicMock()
    mock_rec.create_stream.return_value = MagicMock()

    # Empty samples returns empty
    cfg = dict(config.DEFAULTS)
    res_empty = robust.subnoise_transcribe(mock_rec, np.array([], dtype=np.float32), cfg)
    assert res_empty == []


def test_score_coherence_short_clip_relaxation():
    # Short 2-word phrase in 0.5s (4 wps, relaxed up to 8.0 wps)
    score, reason = robust.score_coherence("Yes please", 0.5)
    assert score > 0.0
    assert reason == "coherent"


def test_cn_rover_fuse():
    # Empty candidate list
    assert robust.cn_rover_fuse([], 2.0) == []

    # Stream 1 has "I", "am", "here"
    s1 = {
        "model": "hybrid",
        "weight": 0.40,
        "words": [
            {"word": "I", "global_start": 0.1, "global_end": 0.3, "conf": 0.9},
            {"word": "am", "global_start": 0.35, "global_end": 0.5, "conf": 0.85},
            {"word": "here.", "global_start": 0.55, "global_end": 0.9, "conf": 0.92},
        ],
    }
    # Stream 2 missed "I", but has "am", "here"
    s2 = {
        "model": "dp4",
        "weight": 0.35,
        "words": [
            {"word": "am", "global_start": 0.36, "global_end": 0.52, "conf": 0.80},
            {"word": "here", "global_start": 0.56, "global_end": 0.88, "conf": 0.88},
        ],
    }

    fused = robust.cn_rover_fuse([s1, s2], 1.5)
    assert len(fused) == 3
    assert fused[0]["word"] == "I"
    assert fused[1]["word"] == "am"
    assert "here" in fused[2]["word"]


def test_merge_words_morphological():
    emitted = [{"word": "Like", "global_start": 0.0, "global_end": 0.4}]
    # 1. Duplicate within 0.45s
    new_words = [{"word": "like", "global_start": 0.05, "global_end": 0.42}]
    robust.merge_words_morphological(emitted, new_words)
    assert len(emitted) == 1

    # 2. Prefix expansion: "pur" -> "purse"
    emitted.append({"word": "pur", "global_start": 0.8, "global_end": 1.1})
    new_words2 = [{"word": "purse.", "global_start": 0.82, "global_end": 1.25}]
    robust.merge_words_morphological(emitted, new_words2)
    assert len(emitted) == 2
    assert emitted[1]["word"] == "purse."


def test_offline_beam_search_configuration(monkeypatch):
    """Verify asr.get_offline passes decoding_method and max_active_paths to recognizer."""
    from stt import asr

    captured_kwargs = {}

    def mock_offline(cfg, model=None, threads=None, blank_penalty=None, decoding_method="greedy_search", max_active_paths=4):
        captured_kwargs["decoding_method"] = decoding_method
        captured_kwargs["max_active_paths"] = max_active_paths
        captured_kwargs["blank_penalty"] = blank_penalty
        return "mock_recognizer"

    monkeypatch.setattr(asr, "offline", mock_offline)
    asr._offline_cache.clear()

    rec = asr.get_offline(
        {"threads": 2, "offline_model": "test"},
        blank_penalty=0.55,
        decoding_method="modified_beam_search",
        max_active_paths=16,
    )
    assert rec == "mock_recognizer"
    assert captured_kwargs["decoding_method"] == "modified_beam_search"
    assert captured_kwargs["max_active_paths"] == 16
    assert captured_kwargs["blank_penalty"] == 0.55


def test_transcribe_rumble_prefilter(monkeypatch):
    """Verify robust.transcribe profiles audio and applies Butterworth HPF when rumble > 70%."""
    from stt import audio

    sr = audio.SR
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # 20 Hz infrasound rumble with high amplitude + small 1000 Hz tone
    heavy_rumble = (1.0 * np.sin(2 * np.pi * 20.0 * t) + 0.05 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)

    hpf_called = []
    orig_hpf = audio.butterworth_highpass

    def mock_hpf(samples, cutoff_hz=120.0, order=4):
        hpf_called.append(cutoff_hz)
        return orig_hpf(samples, cutoff_hz=cutoff_hz, order=order)

    monkeypatch.setattr(audio, "butterworth_highpass", mock_hpf)

    class DummyRecognizer:
        pass

    monkeypatch.setattr(robust.asr, "transcribe", lambda r, chunks: [""] * len(chunks))

    cfg = config.load()
    robust.transcribe(DummyRecognizer(), heavy_rumble, cfg, level=0, no_fallback=True)

    assert len(hpf_called) > 0
    assert hpf_called[0] >= 120.0


def test_compute_dynamic_stream_weights():
    from stt.audio import AcousticProfile

    model_names = ["gtcrn", "hybrid", "dp4", "formant_agc", "spec_gate", "hp"]

    # 1. Drone / heavy rumble regime
    drone_prof = AcousticProfile(
        duration_sec=10.0, peak=0.5, rms=0.1, crest_factor=5.0, dynamic_range_db=30.0,
        noise_floor_rms=0.08, snr_db=3.0, hpf_cutoff_hz=130.0, rumble_fraction=0.92,
        formant_boost_db=0.0, win_sec=5.0, step_sec=1.25, blank_penalty=0.6,
        pregate_threshold=0.08, notch_freqs=(), hum_freq=60.0
    )
    weights_drone = robust.compute_dynamic_stream_weights(drone_prof, model_names)
    assert np.isclose(sum(weights_drone.values()), 1.0, atol=1e-2)
    # GTCRN and Hybrid should dominate in drone regime
    assert weights_drone["gtcrn"] > weights_drone["formant_agc"]

    # 2. Whisper / subnoise regime (low rumble, low SNR)
    whisper_prof = AcousticProfile(
        duration_sec=10.0, peak=0.08, rms=0.02, crest_factor=4.0, dynamic_range_db=20.0,
        noise_floor_rms=0.015, snr_db=2.5, hpf_cutoff_hz=120.0, rumble_fraction=0.15,
        formant_boost_db=4.0, win_sec=5.0, step_sec=1.25, blank_penalty=0.8,
        pregate_threshold=0.015, notch_freqs=(), hum_freq=None
    )
    weights_whisper = robust.compute_dynamic_stream_weights(whisper_prof, model_names)
    assert np.isclose(sum(weights_whisper.values()), 1.0, atol=1e-2)
    # Hybrid and Formant AGC should have highest weights in whisper regime
    assert weights_whisper["formant_agc"] > weights_whisper["hp"]
    assert weights_whisper["hybrid"] > weights_whisper["hp"]


def test_cn_rover_fuse_weighted():
    # Stream 1: Hybrid (high weight 0.35) says "purse"
    stream1 = {
        "model": "hybrid",
        "weight": 0.35,
        "words": [{"word": "purse", "global_start": 1.0, "global_end": 1.4, "conf": 0.85}]
    }
    # Stream 2: Noise hallucination (low weight 0.08) says "thank"
    stream2 = {
        "model": "hp",
        "weight": 0.08,
        "words": [{"word": "thank", "global_start": 1.0, "global_end": 1.4, "conf": 0.40}]
    }

    fused = robust.cn_rover_fuse([stream1, stream2], 2.0)
    assert len(fused) == 1
    assert fused[0]["word"] == "purse"



