#!/usr/bin/env python3
"""Better STT test with longer, clearer audio."""

import json
import os
import sys
import numpy as np
import soundfile as sf
from pathlib import Path
from stt import config, asr, audio, robust

def create_long_clean_audio(duration=5.0, sample_rate=16000):
    """Create a clean audio file with clear speech."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Clear speech: "hello world" repeated
    words = ["hello", "world", "hello", "world", "hello", "world"]
    text = " ".join(words)
    
    # Simple text-to-speech-like generation
    # Use a basic approach: concatenate sine waves at different frequencies
    samples = []
    for word in words:
        # Each word gets a short segment
        seg_len = int(sample_rate * 0.15)
        if seg_len < 2:
            continue
        
        # Different frequencies for each word
        freq_map = {word: 300 + (i * 100) % 500 for i, word in enumerate(words)}
        freq = freq_map[word]
        
        # Generate tone for this word
        tone = np.sin(2 * np.pi * freq * t[:seg_len]) * 0.5
        samples.append(tone)
    
    audio_data = np.concatenate(samples)
    
    # Add some slight variation to make it realistic
    noise = np.random.normal(0, 0.02, len(audio_data))
    final_audio = audio_data + noise
    
    # Normalize
    final_audio = final_audio / (np.abs(final_audio).max() + 1e-8)
    
    # Save
    output_path = Path("test_audio/clean_long.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), final_audio, sample_rate, subtype='PCM_16')
    print(f"Created clean long audio: {output_path}")
    return output_path

def create_long_noisy_audio(duration=5.0, sample_rate=16000):
    """Create a noisy audio file with background noise."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Clear speech
    words = ["hello", "world", "hello", "world", "hello", "world"]
    text = " ".join(words)
    
    # Generate clean segments
    samples = []
    for word in words:
        seg_len = int(sample_rate * 0.15)
        if seg_len < 2:
            continue
        freq = 300 + (words.index(word) * 100) % 500
        tone = np.sin(2 * np.pi * freq * t[:seg_len]) * 0.5
        samples.append(tone)
    
    clean_audio = np.concatenate(samples)
    
    # Add background noise (broadband)
    noise = np.random.normal(0, 0.05, len(clean_audio))
    final_audio = clean_audio + noise
    
    # Normalize
    final_audio = final_audio / (np.abs(final_audio).max() + 1e-8)
    
    # Save
    output_path = Path("test_audio/noisy_long.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), final_audio, sample_rate, subtype='PCM_16')
    print(f"Created noisy long audio: {output_path}")
    return output_path

def test_stt_with_long_audio():
    """Test STT with longer, clearer audio."""
    print("\n=== LONG AUDIO TESTS ===")
    
    # Create test files
    clean_path = create_long_clean_audio(duration=5.0)
    noisy_path = create_long_noisy_audio(duration=5.0)
    
    # Load config and models
    cfg = config.load()
    rec = asr.offline(cfg)
    
    print(f"\nConfig loaded: offline_model={cfg['offline_model']}")
    print(f"Models dir: {cfg['models_dir']}")
    
    # Test clean audio
    print("\n--- CLEAN LONG AUDIO ---")
    print(f"Audio duration: {len(clean_path.read_bytes()) / 16000:.1f}s")
    
    results = {}
    for level in range(6):
        try:
            triples = robust.transcribe(rec, clean_path, cfg, level=level, subnoise=False)
            if triples:
                text = " ".join([t for _, _, t in triples])
                results[f"level_{level}"] = {
                    "text": text,
                    "segments": len(triples),
                    "success": True
                }
            else:
                results[f"level_{level}"] = {
                    "text": "",
                    "segments": 0,
                    "success": False
                }
        except Exception as e:
            results[f"level_{level}"] = {
                "text": "",
                "segments": 0,
                "error": str(e),
                "success": False
            }
    
    print("\n--- NOISY LONG AUDIO ---")
    print(f"Audio duration: {len(noisy_path.read_bytes()) / 16000:.1f}s")
    
    for level in range(6):
        try:
            triples = robust.transcribe(rec, noisy_path, cfg, level=level, subnoise=False)
            if triples:
                text = " ".join([t for _, _, t in triples])
                results[f"level_{level}_noisy"] = {
                    "text": text,
                    "segments": len(triples),
                    "success": True
                }
            else:
                results[f"level_{level}_noisy"] = {
                    "text": "",
                    "segments": 0,
                    "success": False
                }
        except Exception as e:
            results[f"level_{level}_noisy"] = {
                "text": "",
                "segments": 0,
                "error": str(e),
                "success": False
            }
    
    return results

def main():
    print("Better STT Test with Longer Audio")
    print("=" * 50)
    
    # Test clean long audio
    clean_results = test_stt_with_long_audio()
    
    # Test noisy long audio
    noisy_results = test_stt_with_long_audio()
    
    # Print summary
    print("\n=== SUMMARY ===")
    print("\nClean Long Audio:")
    for level, res in sorted(clean_results.items()):
        status = "✓" if res.get("success") else "✗"
        print(f"  Level {level}: {status} - {res.get('text', '')}")
    
    print("\nNoisy Long Audio:")
    for level, res in sorted(noisy_results.items()):
        status = "✓" if res.get("success") else "✗"
        print(f"  Level {level}: {status} - {res.get('text', '')}")
    
    # Save results
    report = {
        "clean_audio": clean_results,
        "noisy_audio": noisy_results,
        "total_files": 2
    }
    
    with open("stt_long_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nReport saved to stt_long_test_report.json")

if __name__ == "__main__":
    main()