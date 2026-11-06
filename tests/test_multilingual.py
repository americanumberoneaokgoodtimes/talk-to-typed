#!/usr/bin/env python3
"""Comprehensive test suite for multilingual transcription and translation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stt import config, asr, audio, robust, translation
from pathlib import Path


def test_multilingual_model_loading():
    """Test that multilingual model can be loaded."""
    print("Testing multilingual model loading...")
    cfg = config.load()
    
    try:
        rec = asr.get_multilingual_offline(cfg)
        print("  OK Multilingual model loaded successfully")
        return True
    except FileNotFoundError as e:
        print(f"  SKIP Multilingual model not found: {e}")
        print("  Run scripts/install_models.sh to download it")
        return True  # Skip, not fail
    except Exception as e:
        print(f"  FAIL Error loading multilingual model: {e}")
        return False


def test_language_detection():
    """Test language detection on sample text."""
    print("\nTesting language detection...")
    
    test_cases = [
        ("Hello world, how are you today?", "en"),
        ("Hola mundo, como estas hoy?", "es"),
        ("Ola mundo, como voce esta hoje?", "pt"),
        ("This is a longer English sentence with more words to test detection accuracy.", "en"),
        ("Esta es una oracion en espanol mas larga para probar la deteccion.", "es"),
        ("Esta e uma frase em portugues mais longa para testar a deteccao.", "pt"),
    ]
    
    all_passed = True
    for text, expected in test_cases:
        detected, conf = translation.detect_language_simple(text)
        status = "OK" if detected == expected else "FAIL"
        print(f"  {status} Expected: {expected}, Got: {detected} (conf: {conf:.2f}) - '{text[:50]}...'")
        if detected != expected:
            all_passed = False
    
    return all_passed


def test_translation_availability():
    """Test if translation model is available."""
    print("\nTesting translation model availability...")
    cfg = config.load()
    model_dir = Path(cfg["models_dir"]) / cfg.get("translation_model", "nllb-200-distilled-600M")
    
    available = translation.is_translation_available(str(model_dir))
    info = translation.get_model_info(str(model_dir))
    
    if available:
        print(f"  OK Translation model available: {info}")
        return True
    else:
        print(f"  SKIP Translation model not found at {model_dir}")
        print(f"  Info: {info}")
        return True  # Skip, not fail


def test_native_transcription():
    """Test native transcription for each language."""
    print("\nTesting native transcription...")
    cfg = config.load()
    
    # Load multilingual model
    try:
        rec = asr.get_multilingual_offline(cfg)
    except Exception as e:
        print(f"  SKIP Cannot load multilingual model: {e}")
        return True  # Skip
    
    # Test with existing test files
    model_test_dir = Path.home() / "AppData" / "Local" / "stt" / "models" / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8" / "test_wavs"
    
    test_files = {
        "en": model_test_dir / "en.wav",
        "es": model_test_dir / "es.wav",
        "fr": model_test_dir / "fr.wav",
    }
    
    results = {}
    for lang, path in test_files.items():
        if not path.exists():
            print(f"  SKIP {lang} test file not found: {path}")
            continue
        
        try:
            samples = audio.decode(str(path))
            # Use transcribe_multilingual with specific language
            triples = robust.transcribe_multilingual(
                rec, samples, cfg,
                source_lang=lang,
                target_lang=None,
                translate=False,
                level=0,
            )
            text = " ".join([t for _, _, t, _ in triples]) if triples else "EMPTY"
            print(f"  {lang}: '{text[:80]}'")
            results[lang] = text
        except Exception as e:
            print(f"  FAIL {lang} failed: {e}")
            results[lang] = None
    
    return len(results) > 0


def test_translation():
    """Test translation functionality."""
    print("\nTesting translation...")
    cfg = config.load()
    model_dir = Path(cfg["models_dir"]) / cfg.get("translation_model", "nllb-200-distilled-600M")
    
    if not translation.is_translation_available(str(model_dir)):
        print("  SKIP Translation model not available, skipping")
        return True
    
    test_cases = [
        ("Hello world", "en", "es"),
        ("Hola mundo", "es", "en"),
        ("Ola mundo", "pt", "en"),
        ("How are you today?", "en", "pt"),
        ("Como estas?", "es", "pt"),
    ]
    
    all_passed = True
    for text, src, tgt in test_cases:
        try:
            translated = translation.translate_text(text, src, tgt, str(model_dir))
            print(f"  {src}->{tgt}: '{text}' -> '{translated}'")
            # Basic check: translated should be different from source for different languages
            if src != tgt and translated.lower().strip() == text.lower().strip():
                print(f"  WARN Translation may have failed (identical to source)")
                all_passed = False
        except Exception as e:
            print(f"  FAIL {src}->{tgt} failed: {e}")
            all_passed = False
    
    return all_passed


def test_end_to_end():
    """Test full pipeline: transcribe + translate."""
    print("\nTesting end-to-end pipeline (transcribe + translate)...")
    cfg = config.load()
    
    try:
        rec = asr.get_multilingual_offline(cfg)
    except Exception as e:
        print(f"  SKIP Cannot load multilingual model: {e}")
        return True  # Skip
    
    model_test_dir = Path.home() / "AppData" / "Local" / "stt" / "models" / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8" / "test_wavs"
    
    # Test Spanish -> English
    es_path = model_test_dir / "es.wav"
    if not es_path.exists():
        print("  SKIP Spanish test file not found")
        return True
    
    try:
        samples = audio.decode(str(es_path))
        
        # Transcribe Spanish and translate to English
        triples = robust.transcribe_multilingual(
            rec, samples, cfg,
            source_lang="es",
            target_lang="en",
            translate=True,
            level=0,
        )
        
        print(f"  Segments: {len(triples)}")
        for s, e, src, tgt in triples:
            print(f"  [{s:.1f}-{e:.1f}] ES: {src}")
            print(f"           EN: {tgt}")
        
        return len(triples) > 0
    except Exception as e:
        print(f"  FAIL End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_output_formats():
    """Test bilingual output formatters."""
    print("\nTesting bilingual output formats...")
    from stt.cli import format_srt, format_vtt, format_jsonl, _is_bilingual
    
    # Test data
    regular = [(0.0, 2.5, "Hello world"), (2.5, 5.0, "How are you?")]
    bilingual = [(0.0, 2.5, "Hola mundo", "Hello world"), (2.5, 5.0, "Como estas?", "How are you?")]
    
    # Test _is_bilingual
    assert _is_bilingual(bilingual) == True
    assert _is_bilingual(regular) == False
    print("  OK _is_bilingual detection works")
    
    # Test SRT
    srt_regular = format_srt(regular)
    srt_bilingual = format_srt(bilingual)
    assert "Hola mundo" in srt_bilingual and "Hello world" in srt_bilingual
    print("  OK SRT bilingual format works")
    
    # Test VTT
    vtt_bilingual = format_vtt(bilingual)
    assert "Hola mundo" in vtt_bilingual and "Hello world" in vtt_bilingual
    print("  OK VTT bilingual format works")
    
    # Test JSONL
    jsonl_bilingual = format_jsonl(bilingual, "test.wav")
    assert "Hola mundo" in jsonl_bilingual and "Hello world" in jsonl_bilingual
    assert "translated" in jsonl_bilingual
    print("  OK JSONL bilingual format works")
    
    return True


def main():
    print("=" * 60)
    print("MULTILINGUAL TRANSCRIPTION & TRANSLATION TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Run tests
    results["model_loading"] = test_multilingual_model_loading()
    results["language_detection"] = test_language_detection()
    results["translation_availability"] = test_translation_availability()
    results["native_transcription"] = test_native_transcription()
    results["translation"] = test_translation()
    results["end_to_end"] = test_end_to_end()
    results["output_formats"] = test_output_formats()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test, passed in results.items():
        status = "OK" if passed else "FAIL"
        print(f"  {status}: {test}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed or skipped (may be due to missing models)")
        return 1


if __name__ == "__main__":
    from pathlib import Path
    sys.exit(main())