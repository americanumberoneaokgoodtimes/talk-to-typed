#!/usr/bin/env python3
"""Comprehensive test of STT system with known speech samples."""

import json
import os
import sys
import numpy as np
import soundfile as sf
import subprocess
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stt import config, asr, audio, robust

def generate_test_speech_files():
    """Generate test audio files with known content."""
    test_dir = Path("test_audio")
    test_dir.mkdir(exist_ok=True)
    
    # Generate clean audio - "test" with variations
    sample_rate = 16000
    
    # Create "test" audio: "t" (150 Hz), "e" (330 Hz), "s" (440 Hz), "t" (150 Hz)
    # with pauses between letters
    t = np.linspace(0, 2, int(sample_rate * 2), endpoint=False)
    # "t" tone
    t1 = np.sin(2 * np.pi * 150 * t[:int(sample_rate * 0.4)]) * 0.7
    # silence between t and e
    s1 = np.zeros(int(sample_rate * 0.6))
    # "e" tone  
    e = np.sin(2 * np.pi * 330 * t[int(sample_rate * 1.0):int(sample_rate * 1.4)]) * 0.7
    # silence between e and s
    s2 = np.zeros(int(sample_rate * 0.6))
    # "s" tone
    s = np.sin(2 * np.pi * 440 * t[int(sample_rate * 1.6):int(sample_rate * 2.0)]) * 0.7
    
    # Combine
    clean_audio = np.concatenate([t1, s1, e, s2, s])
    
    # Save clean audio
    clean_path = test_dir / "clean.wav"
    sf.write(str(clean_path), clean_audio, sample_rate, subtype='PCM_16')
    print(f"Generated clean audio: {clean_path} ({len(clean_audio)/sample_rate:.1f}s)")
    
    # Create noisy version with background noise
    np.random.seed(42)
    duration = len(clean_audio) / sample_rate
    noise = np.random.normal(0, 0.05, len(clean_audio))
    noisy_audio = clean_audio + noise
    
    noisy_path = test_dir / "noisy.wav"
    sf.write(str(noisy_path), noisy_audio, sample_rate, subtype='PCM_16')
    print(f"Generated noisy audio: {noisy_path} (SNR ~{10*np.log10(np.var(clean_audio)/(np.var(noise)+1e-9)):.1f} dB)")
    
    # Create whisper-quiet version (very quiet speech)
    whisper_audio = clean_audio * 0.2 + np.random.normal(0, 0.08, len(clean_audio))
    whisper_path = test_dir / "whisper.wav"
    sf.write(str(whisper_path), whisper_audio, sample_rate, subtype='PCM_16')
    print(f"Generated whisper audio: {whisper_path} (quiet + noise)")
    
    return clean_path, noisy_path, whisper_path

def test_stt_pipeline(audio_path, aggressiveness_levels=[0, 1, 2, 3, 4, 5], description=""):
    """Test STT pipeline with different aggressiveness levels."""
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
    
    for level in aggressiveness_levels:
        print(f"\n--- Testing Aggressiveness Level {level} ---")
        
        try:
            # Test regular transcription
            triples = robust.transcribe(rec, samples, cfg, level=level, subnoise=False)
            
            if not triples:
                print("  No speech detected")
                results[f"level_{level}"] = {"text": "", "segments": [], "duration": 0}
                continue
            
            # Extract results
            text = " ".join([t for _, _, t in triples])
            duration = max([e - s for s, e, _ in triples]) if triples else 0
            
            print(f"  Transcribed: '{text}'")
            print(f"  Duration: {duration:.1f}s")
            print(f"  Segments: {len(triples)}")
            
            results[f"level_{level}"] = {
                "text": text,
                "segments": [{"start": s, "end": e, "text": t} for s, e, t in triples],
                "duration": duration
            }
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results[f"level_{level}"] = {"error": str(e)}
    
    return results

def test_subnoise_mode(audio_path, description=""):
    """Test subnoise mode with aggressive multi-model denoising."""
    print(f"\n{'='*60}")
    print(f"Testing Subnoise Mode: {description}")
    print(f"Audio file: {audio_path}")
    print(f"{'='*60}")
    
    try:
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
        
        # Test subnoise transcription with auto-tune
        print("\n--- Testing Subnoise Transcribe with Auto-tune ---")
        params = robust.auto_tune_settings(dur)
        print(f"Auto-tune parameters: {params}")
        
        triples = robust.subnoise_transcribe(
            rec, samples, cfg, params=params, model=None, threads=cfg["threads"]
        )
        
        if not triples:
            print("  No speech detected with subnoise mode")
            return {"text": "", "segments": [], "auto_tune_params": params}
        
        text = " ".join([t for _, _, t in triples])
        duration = max([e - s for s, e, _ in triples]) if triples else 0
        
        print(f"  Transcribed (subnoise): '{text}'")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Segments: {len(triples)}")
        
        return {
            "text": text,
            "segments": [{"start": s, "end": e, "text": t} for s, e, t in triples],
            "auto_tune_params": params,
            "duration": duration
        }
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def analyze_results(clean_results, noisy_results, whisper_results):
    """Analyze and compare results across different audio conditions."""
    print(f"\n{'='*80}")
    print("ANALYSIS OF RESULTS")
    print(f"{'='*80}")
    
    analysis = {}
    
    # Determine expected transcriptions
    expected_clean = "test"
    expected_noisy = "test"
    expected_whisper = "test"
    
    for condition, results, expected in [
        ("clean", clean_results, expected_clean),
        ("noisy", noisy_results, expected_noisy),
        ("whisper", whisper_results, expected_whisper)
    ]:
        if not results:
            analysis[condition] = {"status": "failed", "reason": "No results available"}
            continue
        
        condition_analysis = {"levels": {}, "summary": ""}
        
        best_level = None
        best_score = -1
        
        for level_key, level_results in results.items():
            if "error" in level_results:
                condition_analysis["levels"][level_key] = {
                    "status": "error",
                    "error": level_results["error"]
                }
                continue
            
            text = level_results.get("text", "")
            
            # Calculate a simple score based on transcription length
            score = len(text) if text else 0
            
            condition_analysis["levels"][level_key] = {
                "status": "success",
                "transcribed": text,
                "segments": level_results.get("segments", []),
                "score": score
            }
            
            if score > best_score:
                best_score = score
                best_level = level_key
        
        # Determine if it matches expected
        best_text = results.get(f"{best_level}_text", "") if best_level else ""
        
        if best_text:
            condition_analysis["expected"] = expected
            condition_analysis["transcribed"] = best_text
            
            if best_text.lower().replace(" ", "").replace("-", "") == expected.lower().replace(" ", "").replace("-", ""):
                condition_analysis["accuracy"] = "exact_match"
                condition_analysis["success"] = True
            elif expected.lower() in best_text.lower() or best_text.lower() in expected.lower():
                condition_analysis["accuracy"] = "partial_match"
                condition_analysis["success"] = True
            else:
                condition_analysis["accuracy"] = "no_match"
                condition_analysis["success"] = False
        else:
            condition_analysis["accuracy"] = "no_transcription"
            condition_analysis["success"] = False
        
        analysis[condition] = condition_analysis
        
        # Print condition summary
        print(f"\n{condition.upper()} AUDIO:")
        print(f"  Expected: '{expected}'")
        print(f"  Best result (level {best_level}): '{best_text}'")
        print(f"  Status: {'✓ SUCCESS' if condition_analysis.get('success') else '✗ FAILED'}")
        print(f"  Accuracy: {condition_analysis.get('accuracy', 'unknown')}")
    
    return analysis

def generate_report(test_summary, analysis):
    """Generate a comprehensive report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = {
        "timestamp": timestamp,
        "test_summary": test_summary,
        "analysis": analysis,
        "recommendations": []
    }
    
    # Generate recommendations based on analysis
    for condition, condition_analysis in analysis.items():
        if condition_analysis.get("status") == "failed":
            report["recommendations"].append(
                f"{condition}: System issue - investigate transcription pipeline"
            )
            continue
        
        success_levels = []
        for level_key, level_data in condition_analysis.get("levels", {}).items():
            if level_data.get("status") == "success" and level_data.get("transcribed"):
                success_levels.append(int(level_key.split('_')[1]))
        
        if success_levels:
            min_success = min(success_levels)
            report["recommendations"].append(
                f"{condition}: Aggressiveness level {min_success} works best; consider using this level for similar noise conditions"
            )
        
        if condition_analysis.get("accuracy") == "no_match":
            report["recommendations"].append(
                f"{condition}: STT system fails to recognize expected text; consider adding more training data or improving audio quality"
            )
    
    return report

def main():
    """Main test function."""
    print("COMPREHENSIVE STT SYSTEM TEST")
    print("=" * 80)
    
    # Generate test speech files
    print("\n1. Generating test speech files...")
    clean_path, noisy_path, whisper_path = generate_test_speech_files()
    
    # Test all conditions
    test_results = {}
    
    print("\n2. Testing STT pipeline...")
    
    # Test clean audio
    print("\n--- CLEAN AUDIO TESTS ---")
    test_results["clean"] = test_stt_pipeline(clean_path, aggressiveness_levels=[0, 1, 2, 3, 4, 5], description="Clean Audio (no noise)")
    
    # Test noisy audio
    print("\n--- NOISY AUDIO TESTS ---")
    test_results["noisy"] = test_stt_pipeline(noisy_path, aggressiveness_levels=[0, 1, 2, 3, 4, 5], description="Noisy Audio (SNR ~5 dB)")
    
    # Test whisper audio
    print("\n--- WHISPER AUDIO TESTS ---")
    test_results["whisper"] = test_stt_pipeline(whisper_path, aggressiveness_levels=[0, 1, 2, 3, 4, 5], description="Whisper Audio (very quiet + noise)")
    
    # Test subnoise mode
    print("\n3. Testing subnoise mode...")
    
    for condition, audio_path in [("noisy", noisy_path), ("whisper", whisper_path)]:
        print(f"\n--- SUBNOISE MODE: {condition} ---")
        test_results[f"{condition}_subnoise"] = test_subnoise_mode(audio_path, description=f"Subnoise processing: {condition}")
    
    # Analyze results
    print("\n4. Analyzing results...")
    
    # Extract best results for analysis
    clean_best = {}
    noisy_best = {}
    whisper_best = {}
    
    for condition, results in test_results.items():
        if not results or "error" in results:
            continue
        
        if condition == "clean":
            clean_best = results
        elif condition == "noisy":
            noisy_best = results
        elif condition == "whisper":
            whisper_best = results
    
    # Create analysis
    analysis = analyze_results(clean_best, noisy_best, whisper_best)
    
    # Generate report
    test_summary = {
        "audio_files_generated": [
            str(clean_path),
            str(noisy_path),
            str(whisper_path)
        ],
        "aggressiveness_levels_tested": list(range(6)),
        "subnoise_mode_tested": True,
        "models_used": {
            "offline_asr": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
            "denoiser": "dpdfnet8.onnx",
            "vad": "silero_vad.onnx"
        },
        "test_conditions": ["clean", "noisy", "whisper", "noisy_subnoise", "whisper_subnoise"]
    }
    
    report = generate_report(test_summary, analysis)
    
    # Save report
    report_dir = Path("stt_test_reports")
    report_dir.mkdir(exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"stt_comprehensive_test_{timestamp_str}.json"
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TEST COMPLETE")
    print(f"{'='*80}")
    print(f"Report saved to: {report_path}")
    print(f"Audio files created in: {Path(test_path).parent}")
    
    # Print summary
    print("\nSUMMARY:")
    for condition, condition_analysis in analysis.items():
        status = "✓" if condition_analysis.get("success") else "✗"
        print(f"  {status} {condition}: {condition_analysis.get('accuracy', 'unknown')}")
    
    print(f"\nRecommendations:")
    for rec in report["recommendations"]:
        print(f"  • {rec}")
    
    return report

if __name__ == "__main__":
    test_path = generate_audio.py if len(sys.argv) > 1 and sys.argv[1] == "--quick" else None
    
    if test_path and test_path.exists():
        # Run quick test
        exec(open(test_path).read())
    else:
        main()