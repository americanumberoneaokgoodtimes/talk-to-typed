#!/usr/bin/env python3
"""Deep scanning and parameter tuning lab for weak/sub-noise speech in ishehereblake.wav.

Focuses on uncovering faint speech (especially Spanish phrases) around 5s - 35s.
Tinkers with:
  - Bandpass filtering & formant equalization
  - Multi-stage denoising (GTCRN, DPDFNet, combinations)
  - Window lengths (3.5s, 5.0s, 6.5s) and dense stepping (1.0s)
  - Modified beam search & blank penalty tuning in sherpa-onnx
"""
import subprocess
import sys
import numpy as np
import soundfile as sf
from pathlib import Path

import sherpa_onnx
import stt.config as config
import stt.audio as audio

WAV_PATH = "/mnt/g/0-WAV/ishehereblake.wav"
SR = 16000


def get_audio_slice(start_sec: float, dur_sec: float) -> np.ndarray:
    raw_path = f"/tmp/blake_slice_{int(start_sec)}_{int(dur_sec)}.raw"
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", str(start_sec), "-t", str(dur_sec),
            "-i", WAV_PATH, "-vn", "-f", "f32le", "-ac", "1", "-ar", str(SR), raw_path
        ],
        capture_output=True, check=True
    )
    return np.fromfile(raw_path, dtype=np.float32)


def bandpass_and_formant_boost(x: np.ndarray, low_hz=120, high_hz=5500, boost_db=3.0) -> np.ndarray:
    """Filter out rumble and hiss, gently boosting speech formant frequencies (300-3000 Hz)."""
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    
    # Cut below low_hz and above high_hz
    fft[freqs < low_hz] = 0.0
    fft[freqs > high_hz] = 0.0
    
    # Formant boost in speech region (300 - 3200 Hz)
    speech_band = (freqs >= 300) & (freqs <= 3200)
    boost_linear = 10.0 ** (boost_db / 20.0)
    fft[speech_band] *= boost_linear
    
    return np.fft.irfft(fft).astype(np.float32)


def build_beam_recognizer(cfg: dict, method="greedy_search", max_paths=4, blank_penalty=0.0, dither=0.0):
    d = Path(cfg["models_dir"]) / cfg["offline_model"]
    files = {
        "encoder": str(d / "encoder.int8.onnx"),
        "decoder": str(d / "decoder.int8.onnx"),
        "joiner": str(d / "joiner.int8.onnx"),
        "tokens": str(d / "tokens.txt"),
    }
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        **files,
        num_threads=min(cfg.get("threads", 4), 4),
        model_type="nemo_transducer",
        decoding_method=method,
        max_active_paths=max_paths,
        blank_penalty=blank_penalty,
        dither=dither,
    )


def transcribe_samples(rec, samples_list: list[np.ndarray]) -> list[str]:
    streams = []
    for s in samples_list:
        st = rec.create_stream()
        st.accept_waveform(SR, s)
        streams.append(st)
    rec.decode_streams(streams)
    return [st.result.text.strip() for st in streams]


def run_experiment(start_sec=0.0, dur_sec=40.0):
    print(f"=== Loading {dur_sec}s slice from {start_sec}s ===")
    raw = get_audio_slice(start_sec, dur_sec)
    cfg = config.load()

    # Pre-processed audio variants
    print("Generating audio variants...")
    bp = bandpass_and_formant_boost(raw, low_hz=120, high_hz=5500, boost_db=3.0)
    bp_strong = bandpass_and_formant_boost(raw, low_hz=140, high_hz=5000, boost_db=5.0)

    # Denoiser passes
    cfg_gt = dict(cfg, robust_denoiser="gtcrn_simple.onnx")
    cfg_dp4 = dict(cfg, robust_denoiser="dpdfnet4.onnx")
    cfg_dp2 = dict(cfg, robust_denoiser="dpdfnet2.onnx")

    variants = {
        "raw_bp": bp,
        "bp_strong": bp_strong,
        "gtcrn": audio.denoise(bp, cfg_gt),
        "dp4": audio.denoise(bp, cfg_dp4),
        "gtcrn_mix30": audio.normalize(audio.denoise(bp, cfg_gt) + 0.3 * bp),
        "gtcrn_dp2": audio.denoise(audio.denoise(bp, cfg_gt), cfg_dp2),
    }

    norm = lambda c: c * (0.8 / max(float(np.abs(c).max()), 1e-6))

    # Test parameter combinations
    decoding_configs = [
        ("greedy_p0", "greedy_search", 1, 0.0),
        ("greedy_p1.0", "greedy_search", 1, 1.0),
        ("greedy_p2.0", "greedy_search", 1, 2.0),
        ("beam4_p0", "modified_beam_search", 4, 0.0),
        ("beam4_p1.0", "modified_beam_search", 4, 1.0),
        ("beam4_p2.0", "modified_beam_search", 4, 2.0),
    ]

    print("\nTesting dense sliding window scan (window=5.0s, step=1.25s) across decoders...")
    win_len = int(5.0 * SR)
    step = int(1.25 * SR)

    windows = [(i, i + win_len) for i in range(0, len(raw) - win_len + 1, step)]

    for dec_name, method, paths, penalty in decoding_configs:
        print(f"\n--- Decoder: {dec_name} (method={method}, paths={paths}, penalty={penalty}) ---")
        try:
            rec = build_beam_recognizer(cfg, method=method, max_paths=paths, blank_penalty=penalty)
        except Exception as e:
            print(f"Failed to build recognizer: {e}")
            continue

        found_any = False
        for start_idx, end_idx in windows:
            t0 = start_sec + (start_idx / SR)
            t1 = start_sec + (end_idx / SR)

            # Test each audio variant
            row = []
            for vname, vdata in variants.items():
                chunk = norm(vdata[start_idx:end_idx])
                txt = transcribe_samples(rec, [chunk])[0]
                if txt:
                    row.append((vname, txt))

            if row:
                found_any = True
                print(f"[{t0:5.2f}s - {t1:5.2f}s]:")
                for vn, t in row:
                    print(f"   {vn:<12}: {t}")

        if not found_any:
            print("  (no speech detected)")


if __name__ == "__main__":
    t_start = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    t_dur = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    run_experiment(t_start, t_dur)
