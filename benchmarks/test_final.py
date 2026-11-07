#!/usr/bin/env python3
"""Final comprehensive STT test using actual files."""

import json
import os
import sys
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stt import config, asr, audio, robust

def create_test_audio_files():
    """Create test audio files with known content."""
    test_dir = Path("test_audio")
    test_dir.mkdir(exist_ok=True)
    
    sample_rate = 16000
    
    # Create a 5-second clean speech-like audio file
    t = np.linspace(0, 5, int(sample_rate * 5), endpoint=False)
    
    # Create "hello world" pattern - 6 words over 5 seconds
    # Each word gets ~0.5s, with gaps
    words = ["hello", "world", "hello", "world", "hello", "world"]
    freqs = [300, 400, 300, 400, 300, 400]
    
    clean_segments = []
    for i, (word, freq) in enumerate(zip(words, freqs)):
        # 0.5s tone
        tone = np.sin(2 * np.pi * freq * t[:int(sample_rate * 0.5)]) * 0.7
        # 0.33s silence
        silence = np.zeros(int(sample_rate * 0.33))
        clean_segments.append(tone)
        clean_segments.append(silence)
    
    clean_audio = np.concatenate(clean_segments)
    # Trim to exactly 5 seconds
    clean_audio = clean_audio[:int(sample_rate * 5)]
    
    # Normalize
    clean_audio = clean_audio / (np.abs(clean_audio).max() + 1e-8)
    
    # Save clean audio
    clean_path = test_dir / "clean_long.wav"
    sf.write(str(clean_path), clean_audio, sample_rate, subtype='PCM_16')
    print(f"Generated clean audio: {clean_path} (5.0s)")
    
    # Create noisy version
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, len(clean_audio))
    noisy_audio = clean_audio + noise
    noisy_audio = noisy_audio / (np.abs(noisy_audio).max() + 1e-8)
    
    noisy_path = test_dir / "noisy_long.wav"
    sf.write(str(noisy_path), noisy_audio, sample_rate, subtype='PCM_16')
    print(f"Generated noisy audio: {noisy_path} (SNR ~{10*np.log10(np.var(clean_audio)/(np.var(noise)+1e-9)):.1f} dB)")
    
    # Create very noisy version
    noise2 = np.random.normal(0, 0.2, len(clean_audio))
    very_noisy_audio = clean_audio + noise2
    very_noisy_audio = very_noisy_audio / (np.abs(very_noisy_audio).max() + 1e-8)
    
    very_noisy_path = test_dir / "very_noisy_long.wav"
    sf.write(str(very_noisy_path), very_noisy_audio, sample_rate, subtype='PCM_16')
    print(f"Generated very noisy audio: {very_noisy_path} (SNR ~{10*np.log10(np.var(clean_audio)/(np.var(noise2)+1e-9)):.1f} dB)")
    
    return clean_path, noisy_path, very_noisy_path

def test_audio_file(audio_path, description=""):
    """Test STT on a single audio file with all aggressiveness levels."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Audio file: {audio_path}")
    print(f"{'='*60}")
    
    # Load config and models
    cfg = config.load()
    rec = asr.offline(cfg)
    
    # Decode audio
    samples = audio.decode(str(audio_path))
    if samples is None:
        print(f"ERROR: Could not decode {audio_path}")
        return None
    
    dur = len(samples) / audio.SR
    print(f"Audio duration: {dur:.1f}s")
    
    results = {}
    
    for level in range(6):
        print(f"\n--- Aggressiveness Level {level} ---")
        
        try:
            triples = robust.transcribe(rec, samples, cfg, level=level, subnoise=False)
            
            if not triples:
                print("  No speech detected")
                results[f"level_{level}"] = {
                    "text": "", "segments": 0, "duration": 0, "success": False
                }
                continue
            
            text = " ".join([t for _, _, t in triples])
            duration = max([e - s for s, e, _ in triples]) if triples else 0
            
            print(f"  Transcribed: '{text}'")
            print(f"  Duration: {duration:.1f}s, Segments: {len(triples)}")
            
            results[f"level_{level}"] = {
                "text": text,
                "segments": len(triples),
                "duration": duration,
                "success": bool(text)
            }
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results[f"level_{level}"] = {"error": str(e), "success": False}
    
    return results

def test_subnoise(audio_path, description=""):
    """Test subnoise mode."""
    print(f"\n{'='*60}")
    print(f"Testing Subnoise Mode: {description}")
    print(f"Audio file: {audio_path}")
    print(f"{'='*60}")
    
    try:
        cfg = config.load()
        rec = asr.offline(cfg)
        
        samples = audio.decode(str(audio_path))
        if samples is None:
            print(f"ERROR: Could not decode {audio_path}")
            return None
        
        dur = len(samples) / audio.SR
        print(f"Audio duration: {dur:.1f}s")
        
        params = robust.auto_tune_settings(dur)
        print(f"Auto-tune: win={params['win_sec']}s, step={params['step_sec']}s, boost={params['boost_db']}dB")
        
        triples = robust.subnoise_transcribe(rec, samples, cfg, params=params)
        
        if not triples:
            print("  No speech detected (subnoise)")
            return {"text": "", "segments": 0, "success": False}
        
        text = " ".join([t for _, _, t in triples])
        duration = max([e - s for s, e, _ in triples])
        
        print(f"  Transcribed: '{text}'")
        print(f"  Duration: {duration:.1f}s, Segments: {len(triples)}")
        
        return {
            "text": text,
            "segments": len(triples),
            "duration": duration,
            "success": bool(text),
            "params": params
        }
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "success": False}

def main():
    print("COMPREHENSIVE STT TEST WITH REALISTIC AUDIO")
    print("=" * 60)
    
    # Create test audio files
    print("\n1. Creating test audio files...")
    clean_path, noisy_path, very_noisy_path = create_test_audio_files()
    
    # Test all conditions
    test_results = {}
    
    print("\n2. Testing STT pipeline...")
    test_results["clean"] = test_audio_file(clean_path, "Clean Audio (5s)")
    test_results["noisy"] = test_audio_file(noisy_path, "Noisy Audio (SNR ~10 dB)")
    test_results["very_noisy"] = test_audio_file(very_noisy_path, "Very Noisy Audio (SNR ~4 dB)")
    
    print("\n3. Testing Subnoise Mode...")
    test_results["noisy_subnoise"] = test_subnoise(noisy_path, "Noisy Audio Subnoise")
    test_results["very_noisy_subnoise"] = test_subnoise(very_noisy_path, "Very Noisy Audio Subnoise")
    
    # Analyze and generate report
    print("\n4. Analyzing results...")
    
    summary = {
        "test_files": {
            "clean": str(clean_path),
            "noisy": str(noisy_path),
            "very_noisy": str(very_noisy_path)
        },
        "aggressiveness_levels": list(range(6)),
        "subnoise_tested": True,
        "models": {
            "offline": config.load()["offline_model"],
            "denoiser": config.load()["robust_denoiser"],
            "vad": config.load()["vad_model"]
        },
        "results": test_results
    }
    
    # Print summary table
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    for test_name, results in test_results.items():
        if not results:
            print(f"\n{test_name}: FAILED")
            continue
        
        print(f"\n{test_name.upper()}:")
        print(f"  {'Level':<12} {'Success':<8} {'Text':<50}")
        print(f"  {'-'*70}")
        
        for level_key, res in results.items():
            if "error" in res:
                print(f"  {level_key:<12} {'ERROR':<8} {res['error'][:50]}")
            else:
                success = "YES" if res.get("success") else "NO"
                text = res.get("text", "")[:50]
                print(f"  {level_key:<12} {success:<8} {text}")
    
    # Save report
    report_dir = Path("stt_test_reports")
    report_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"stt_comprehensive_report_{timestamp}.json"
    
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Report saved to: {report_path}")
    print(f"Audio files in: test_audio/")
    print("="*80)
    
    return summary

if __name__ == "__main__":
    main()