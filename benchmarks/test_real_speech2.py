#!/usr/bin/env python3
"""Test STT with the real speech files provided by the model - fixed imports."""
import sys, json, os
sys.path.insert(0, '.')
from stt import config, asr, audio, robust
import numpy as np

def mix(clean, kind, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "white":
        n = rng.standard_normal(len(clean))
    elif kind in ("pink", "brown"):
        f = np.fft.rfft(rng.standard_normal(len(clean)))
        k = np.maximum(np.arange(len(f)), 1)
        f /= np.sqrt(k) if kind == "pink" else k
        n = np.fft.irfft(f, len(clean))
    else:
        n = rng.standard_normal(len(clean))
    
    fr = clean[: len(clean) // 512 * 512].reshape(-1, 512)
    e = np.sqrt((fr**2).mean(1))
    sp = np.sqrt(np.mean(fr[e > 0.3 * e.max()] ** 2))
    n = (n / np.sqrt(np.mean(n**2))) * sp / (10 ** (snr_db / 20))
    return np.clip(clean + n, -1, 1).astype(np.float32)

def main():
    cfg = config.load()
    rec = asr.offline(cfg)
    print(f"Offline ASR loaded")
    print(f"Model: {cfg['offline_model']}")
    print(f"Denoiser: {cfg['robust_denoiser']}")
    
    # Test with provided English test wav
    audio_path = r'C:\Users\o\AppData\Local\stt\models\sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8\test_wavs\en.wav'
    samples = audio.decode(audio_path)
    print(f"\nEnglish WAV: duration={len(samples)/16000:.1f}s")
    if samples is not None:
        print(f"Range: [{samples.min():.4f}, {samples.max():.4f}]")
        for level in range(6):
            triples = robust.transcribe(rec, samples, cfg, level=level, subnoise=False)
            text = ' '.join([t for _, _, t in triples]) if triples else ''
            segments = len(triples)
            print(f"  Level {level}: '{text}' ({segments} seg)")
    
    # Test noisy versions
    clean = samples
    print("\n=== Noisy Tests (White Noise) ===")
    for snr in [20, 10, 5, 0, -5, -10]:
        noisy = mix(clean, "white", snr)
        print(f"\n  SNR {snr} dB:")
        for level in range(6):
            triples = robust.transcribe(rec, noisy, cfg, level=level, subnoise=False)
            text = ' '.join([t for _, _, t in triples]) if triples else ''
            segments = len(triples)
            print(f"    Level {level}: '{text[:80]}' ({segments} seg)")
    
    # Test subnoise
    print("\n=== Subnoise Tests ===")
    for snr in [10, 5, 0, -5, -10]:
        noisy = mix(clean, "white", snr)
        params = robust.auto_tune_settings(len(noisy)/16000)
        triples = robust.subnoise_transcribe(rec, noisy, cfg, params=params)
        text = ' '.join([t for _, _, t in triples]) if triples else ''
        print(f"  SNR {snr}: '{text[:80]}' ({len(triples)} segs)")
    
    # Test pink noise
    print("\n=== Pink Noise Tests ===")
    for snr in [10, 5, 0, -5]:
        noisy = mix(clean, "pink", snr)
        print(f"\n  SNR {snr} dB:")
        for level in [0, 3, 5]:
            triples = robust.transcribe(rec, noisy, cfg, level=level, subnoise=False)
            text = ' '.join([t for _, _, t in triples]) if triples else ''
            print(f"    Level {level}: '{text[:80]}' ({len(triples)} seg)")
    
    # Test subnoise on pink
    print("\n=== Subnoise on Pink Noise ===")
    for snr in [5, 0, -5]:
        noisy = mix(clean, "pink", snr)
        params = robust.auto_tune_settings(len(noisy)/16000)
        triples = robust.subnoise_transcribe(rec, noisy, cfg, params=params)
        text = ' '.join([t for _, _, t in triples]) if triples else ''
        print(f"  SNR {snr}: '{text[:80]}' ({len(triples)} segs)")
    
    # Test other languages
    print("\n=== Other Languages ===")
    for lang in ["de", "es", "fr"]:
        apath = f'C:\\Users\\o\\AppData\\Local\\stt\\models\\sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8\\test_wavs\\{lang}.wav'
        a = audio.decode(apath)
        if a is not None:
            print(f"\n  {lang.upper()} WAV: duration={len(a)/16000:.1f}s")
            for level in range(3):
                triples = robust.transcribe(rec, a, cfg, level=level, subnoise=False)
                text = ' '.join([t for _, _, t in triples]) if triples else ''
                print(f"    Level {level}: '{text[:80]}' ({len(triples)} seg)")

if __name__ == "__main__":
    main()