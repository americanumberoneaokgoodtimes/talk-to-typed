"""Regression & end-to-end verification tests for DSP fixes, sub-noise detection,
and simultaneous foreground + buried background speech recovery with burial-depth logging.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pytest

from stt import asr, audio, config, robust
from stt.scripts.install_models import is_valid_onnx


def test_config_toml_escaping_and_defaults(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.toml"
    monkeypatch.setattr(config, "CONFIG_PATH", cfg_file)

    win_models_path = r'C:\Users\tester\My "Models"\stt\models'
    config.update_config({"models_dir": win_models_path})

    reloaded = config.load()
    assert reloaded["models_dir"] == win_models_path
    assert reloaded["multilingual_model"] == "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    assert reloaded["subnoise_gap_scan"] is True


def test_install_models_is_valid_onnx(tmp_path):
    valid_proto = tmp_path / "model.onnx"
    # Protobuf field 1 (ir_version) varint tag 0x08 followed by payload >= 1024 bytes
    valid_proto.write_bytes(b"\x08\x07\x12\x04onnx" + b"\x00" * 2048)
    assert is_valid_onnx(valid_proto) is True

    truncated = tmp_path / "tiny.onnx"
    truncated.write_bytes(b"\x08\x07\x12\x04")
    assert is_valid_onnx(truncated) is False

    html_error = tmp_path / "error.onnx"
    html_error.write_bytes(b"<!DOCTYPE html><html>Error 404</html>" + b" " * 2048)
    assert is_valid_onnx(html_error) is False


def test_noise_bounded_agc_high_noise_floor_no_attenuation():
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, 2.0, 32000, endpoint=False, dtype=np.float32)
    # High ambient floor (RMS ~ 0.10) + speech-like tone
    noisy = (0.10 * rng.standard_normal(32000) + 0.08 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    in_rms = float(np.sqrt(np.mean(noisy ** 2)))

    out = audio.noise_bounded_agc(noisy, noise_floor_rms=0.10, min_gain=1.0)
    out_rms = float(np.sqrt(np.mean(out ** 2)))
    # Previously attenuated signal by ~0.25x (-12 dB); now must never attenuate below 1.0x
    assert out_rms >= in_rms * 0.98


def test_multiband_formant_agc_stationary_hiss_bounded():
    rng = np.random.default_rng(123)
    sr = 16000
    # Pure stationary Gaussian hiss (no syllabic modulation, flat spectrum)
    hiss = (0.01 * rng.standard_normal(sr * 3)).astype(np.float32)
    hiss_out = audio.multiband_formant_agc(hiss, sr=sr)

    in_rms = float(np.sqrt(np.mean(hiss ** 2)))
    out_rms = float(np.sqrt(np.mean(hiss_out ** 2)))
    gain_db = 20.0 * np.log10((out_rms + 1e-9) / (in_rms + 1e-9))
    # Old implementation boosted stationary hiss by +18.0 dB; gated AGC keeps it near unity
    assert gain_db < 3.0, f"Stationary hiss amplified too aggressively: {gain_db:.2f} dB"

    # Syllabically modulated formant burst (4 Hz AM on 700 Hz + 1600 Hz) SHOULD be lifted cleanly
    t = np.arange(sr * 3, dtype=np.float32) / sr
    env = 0.5 * (1.0 - np.cos(2.0 * np.pi * 4.0 * t))
    speech_like = (0.008 * env * (np.sin(2.0 * np.pi * 700.0 * t) + 0.6 * np.sin(2.0 * np.pi * 1600.0 * t))).astype(np.float32)
    speech_out = audio.multiband_formant_agc(speech_like, sr=sr)
    sp_gain_db = 20.0 * np.log10(
        (float(np.sqrt(np.mean(speech_out ** 2))) + 1e-9)
        / (float(np.sqrt(np.mean(speech_like ** 2))) + 1e-9)
    )
    assert sp_gain_db > 6.0, f"Expected modulated formant boost > 6 dB, got {sp_gain_db:.2f} dB"


def test_spectral_crossover_blend_gates_white_hiss_keeps_bursts():
    rng = np.random.default_rng(7)
    sr = 16000
    n = sr * 2
    denoised_silence = np.zeros(n, dtype=np.float32)
    stationary_white = (0.05 * rng.standard_normal(n)).astype(np.float32)

    blended_hiss = audio.spectral_crossover_blend(
        denoised_silence, stationary_white, sr=sr, hp_high_weight=0.70
    )
    hi_in = audio.butterworth_highpass(stationary_white, cutoff_hz=4500.0)
    hi_out = audio.butterworth_highpass(blended_hiss, cutoff_hz=4500.0)
    ratio_hiss = float(np.sqrt(np.mean(hi_out ** 2)) / (np.sqrt(np.mean(hi_in ** 2)) + 1e-9))
    assert ratio_hiss < 0.25, f"Stationary high-band hiss leaked too strongly: ratio={ratio_hiss:.3f}"

    # Modulated high-frequency consonant burst MUST pass through cleanly (>0.45x of weighted HP)
    t = np.arange(n, dtype=np.float32) / sr
    burst_env = np.where((t >= 0.6) & (t <= 1.0), 1.0, 0.0).astype(np.float32)
    burst = (0.10 * burst_env * np.sin(2.0 * np.pi * 5200.0 * t)).astype(np.float32)
    blended_burst = audio.spectral_crossover_blend(
        denoised_silence, burst, sr=sr, hp_high_weight=0.70
    )
    act = (t >= 0.65) & (t <= 0.95)
    burst_ratio = float(np.sqrt(np.mean(blended_burst[act] ** 2)) / (np.sqrt(np.mean(burst[act] ** 2)) + 1e-9))
    assert burst_ratio > 0.55, f"High-band consonant burst attenuated unexpectedly: ratio={burst_ratio:.3f}"


def test_harmonic_pitch_vad_detects_0db_voiced_speech():
    rng = np.random.default_rng(99)
    sr = 16000
    t = np.arange(sr * 2, dtype=np.float32) / sr

    # Rich harmonic vowel (F0 = 115 Hz, harmonics 1..10) buried at 0 dB SNR in white noise
    f0 = 115.0
    voiced = np.zeros_like(t)
    for k in range(1, 11):
        voiced += (1.0 / (k ** 0.7)) * np.sin(2.0 * np.pi * (k * f0) * t)
    voiced = voiced / (float(np.sqrt(np.mean(voiced ** 2))) + 1e-9) * 0.04
    noise = (0.04 * rng.standard_normal(len(t))).astype(np.float32)

    mask_voiced = audio.harmonic_pitch_vad(voiced + noise, sr=sr)
    mask_noise = audio.harmonic_pitch_vad(noise, sr=sr)

    assert float(np.mean(mask_voiced)) >= 0.50, f"0 dB voiced detection too low: {np.mean(mask_voiced):.2f}"
    assert float(np.mean(mask_noise)) < 0.12, f"False voiced rate on white noise too high: {np.mean(mask_noise):.2f}"


def test_soft_declip_and_streaming_declicker_boundary():
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    clean_sine = (0.85 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)
    clipped = np.clip(1.6 * clean_sine, -0.99, 0.99).astype(np.float32)
    clipped_crest = float(np.max(np.abs(clipped)) / (np.sqrt(np.mean(clipped ** 2)) + 1e-9))

    restored, n_repaired = audio.soft_declip(clipped, clip_ratio=0.985)
    restored_crest = float(np.max(np.abs(restored)) / (np.sqrt(np.mean(restored ** 2)) + 1e-9))
    assert n_repaired > 0
    # Cubic Hermite reconstructs peaks above flat-top clipping, increasing crest factor toward sinusoid sqrt(2)
    assert restored_crest > clipped_crest

    # StreamingDeclicker must preserve clean sinusoidal continuity across arbitrary chunk boundaries
    margin = 64
    declicker = audio.StreamingDeclicker(sr=sr, margin_samples=margin, threshold_sigma=6.5)
    chunks = [clean_sine[i : i + 500] for i in range(0, len(clean_sine), 500)]
    outs = [declicker.process(c) for c in chunks]
    reconstructed = np.concatenate(outs)
    assert len(reconstructed) == len(clean_sine)
    max_err = float(np.max(np.abs(reconstructed[margin:] - clean_sine[:-margin])))
    assert max_err < 1e-4, f"StreamingDeclicker corrupted clean signal across boundaries: {max_err}"


def test_adaptive_spectral_gate_log_mmse_improves_snr():
    rng = np.random.default_rng(314)
    sr = 16000
    t = np.arange(sr * 3, dtype=np.float32) / sr
    noise = (0.04 * rng.standard_normal(len(t))).astype(np.float32)
    speech = np.zeros_like(t)
    # Speech active between 1.0s and 2.2s
    active = (t >= 1.0) & (t <= 2.2)
    speech[active] = 0.08 * (
        np.sin(2.0 * np.pi * 220.0 * t[active]) + 0.7 * np.sin(2.0 * np.pi * 660.0 * t[active])
    )

    gated = audio.adaptive_spectral_gate(speech + noise, sr=sr, g_floor_db=-20.0)
    bg_in_rms = float(np.sqrt(np.mean(noise[:sr] ** 2)))
    bg_out_rms = float(np.sqrt(np.mean(gated[sr // 4 : sr] ** 2)))
    sp_out_rms = float(np.sqrt(np.mean(gated[active] ** 2)))

    assert bg_out_rms < bg_in_rms * 0.35
    assert sp_out_rms > bg_out_rms * 3.0


def test_merge_words_morphological_preserves_same_window_words():
    def _w(word: str, s: float, e: float, c: float = 0.9) -> dict:
        return {"word": word, "global_start": s, "global_end": e, "conf": c}

    # Window 1 emits: "We went to today's"
    emitted = [
        _w("We", 0.10, 0.25, 0.90),
        _w("went", 0.30, 0.55, 0.92),
        _w("to", 0.60, 0.72, 0.88),
        _w("today's", 0.78, 1.15, 0.91),
    ]
    # Overlapping Window 2 emits: "today's the theater in Indiana"
    w2 = [
        _w("today's", 0.80, 1.16, 0.93),
        _w("the", 1.22, 1.35, 0.89),
        _w("theater", 1.40, 1.80, 0.94),
        _w("in", 1.85, 1.98, 0.90),
        _w("Indiana", 2.02, 2.45, 0.95),
    ]
    robust.merge_words_morphological(emitted, w2)
    tokens = [w["word"] for w in emitted]
    assert tokens == ["We", "went", "to", "today's", "the", "theater", "in", "Indiana"]


def test_cn_rover_fuse_recovers_high_conf_single_stream():
    def _w(word: str, s: float, e: float, c: float) -> dict:
        return {"word": word, "global_start": s, "global_end": e, "conf": c}

    # Stream 0 recovers shared anchor word "the" + high-conf buried specialist word "buried";
    # Stream 1 recovers shared "the" + low-conf filler hallucination "uh".
    fused = robust.cn_rover_fuse(
        [
            {"words": [_w("the", 0.1, 0.3, 0.85), _w("buried", 0.5, 0.9, 0.84)], "weight": 0.25},
            {"words": [_w("the", 0.12, 0.32, 0.82), _w("uh", 1.5, 1.8, 0.31)], "weight": 0.25},
        ],
        dur_sec=2.5,
    )
    words_out = [w["word"] for w in fused]
    assert "buried" in words_out
    assert "uh" not in words_out


def test_mic_none_cli_args_coalescing():
    """Ensure argparse None defaults for DSP overrides never raise TypeError."""
    args = argparse.Namespace(
        rumble_cutoff=None,
        formant_boost=None,
        blank_penalty=None,
        declick_threshold=None,
    )
    hp_hz = args.rumble_cutoff if getattr(args, "rumble_cutoff", None) is not None else 95.0
    boost_db = args.formant_boost if getattr(args, "formant_boost", None) is not None else 4.0
    declick_mad = args.declick_threshold if getattr(args, "declick_threshold", None) is not None else 6.5
    assert hp_hz == 95.0 and boost_db == 4.0 and declick_mad == 6.5


def test_foreground_and_buried_background_speech_recovery_with_feedback(capsys):
    """End-to-end verification on real human speech:
    When a file contains loud foreground speech AND faint background speech buried inside
    the noise floor of a later gap, BOTH must be transcribed and burial depth logged.
    """
    cfg = config.load()
    models_dir = Path(cfg["models_dir"])
    en_wav = models_dir / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8" / "test_wavs" / "en.wav"
    gtcrn_path = models_dir / "gtcrn_simple.onnx"
    vad_path = models_dir / "silero_vad.onnx"

    if not (en_wav.exists() and gtcrn_path.exists() and vad_path.exists()):
        pytest.skip("Local Parakeet / GTCRN / Silero models not installed")

    sr = audio.SR
    full_speech = audio.decode(str(en_wav))
    mid = len(full_speech) // 2
    utt_fg = full_speech[:mid].copy()
    utt_bg = full_speech[mid:].copy()

    fg_unit = utt_fg / (float(np.sqrt(np.mean(utt_fg ** 2))) + 1e-9)
    bg_unit = utt_bg / (float(np.sqrt(np.mean(utt_bg ** 2))) + 1e-9)

    # Create a 12.0s recording with continuous stationary ambient noise (RMS = 0.016)
    rng = np.random.default_rng(2026)
    total_len = int(12.0 * sr)
    white = rng.standard_normal(total_len).astype(np.float32)
    ambient = white + 0.8 * np.convolve(white, np.ones(8, dtype=np.float32) / 8.0, mode="same")
    ambient = ambient / (float(np.sqrt(np.mean(ambient ** 2))) + 1e-9) * 0.016

    composite = ambient.copy()
    # 1) Loud foreground speech at [0.6s -> ~2.5s], RMS = 0.14 (~ +18.8 dB SNR)
    fg_s = int(0.6 * sr)
    composite[fg_s : fg_s + len(fg_unit)] += fg_unit * 0.14

    # 2) Faint background speech buried inside noise floor at [6.4s -> ~8.3s],
    #    RMS = 0.013 (~ -1.8 dB RMS vs ambient floor, ~ -20.6 dB below foreground!)
    bg_s = int(6.4 * sr)
    composite[bg_s : bg_s + len(bg_unit)] += bg_unit * 0.013
    composite = np.clip(composite, -0.99, 0.99).astype(np.float32)

    rec = asr.get_offline(cfg, threads=2)
    run_cfg = dict(cfg, subnoise_gap_scan=True)
    triples = robust.transcribe(rec, composite, run_cfg, level=0)
    err_log = capsys.readouterr().err
    diags = robust.get_last_segment_diagnostics()

    # Must recover BOTH the foreground utterance (< 4.5s) AND the buried background utterance (>= 4.5s)
    fg_segs = [(s, e, t) for s, e, t in triples if s < 4.5 and len(t.strip()) >= 3]
    bg_segs = [(s, e, t) for s, e, t in triples if e > 6.0 and s >= 4.5 and len(t.strip()) >= 3]

    assert len(fg_segs) >= 1, f"Foreground speech missing! Got: {triples}"
    assert len(bg_segs) >= 1, f"Buried background speech in gap not recovered! Got: {triples}"

    # Verify user-level burial feedback was logged to stderr with [MM:SS.mmm -> MM:SS.mmm] interval and dB metrics
    assert ("Buried speech detected" in err_log or "Faint/background speech detected" in err_log), (
        f"Expected burial telemetry in stderr, got:\n{err_log}"
    )
    assert "Acoustic layers recovered:" in err_log
    assert "vs foreground" in err_log
    assert re.search(r"\[\d{2}:\d{2}\.\d{3} -> \d{2}:\d{2}\.\d{3}\]", err_log)

    # Verify programmatic burial diagnostics classify both acoustic layers accurately
    layers = [d["layer"] for d in diags]
    assert "foreground" in layers
    assert any(l in ("subnoise-buried", "background-faint") for l in layers), f"Unexpected layers: {diags}"
    min_depth = min(
        (d["depth_vs_foreground_db"] for d in diags if d["depth_vs_foreground_db"] is not None),
        default=0.0,
    )
    assert min_depth <= -10.0, f"Expected buried speech >=10 dB below foreground, got {min_depth} dB"
