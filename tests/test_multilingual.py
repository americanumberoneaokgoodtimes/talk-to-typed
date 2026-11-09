#!/usr/bin/env python3
"""Comprehensive test suite for multilingual transcription and translation."""
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stt import asr, audio, config, robust, translation


def test_multilingual_model_loading():
    """Test that multilingual model can be loaded when installed."""
    cfg = config.load()
    model_dir = Path(cfg["models_dir"]) / cfg.get(
        "multilingual_model", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    )
    if not model_dir.is_dir():
        pytest.skip(f"Multilingual model not installed at {model_dir}")

    rec = asr.get_multilingual_offline(cfg)
    assert rec is not None


def test_language_detection():
    """Test heuristic language detection on accented and unaccented text."""
    test_cases = [
        ("Hello world, how are you today?", "en"),
        ("Hola mundo, como estas hoy?", "es"),
        ("¡Hola mundo! ¿Cómo estás hoy?", "es"),
        ("Ola mundo, como voce esta hoje?", "pt"),
        ("Olá mundo, tudo bem com você hoje?", "pt"),
        ("This is a longer English sentence with more words to test detection accuracy.", "en"),
        ("Esta es una oracion en espanol mas larga para probar la deteccion.", "es"),
        ("Esta e uma frase em portugues mais longa para testar a deteccao.", "pt"),
    ]

    for text, expected in test_cases:
        detected, conf = translation.detect_language_simple(text)
        assert detected == expected, f"Expected {expected}, got {detected} for '{text}'"
        assert 0.0 <= conf <= 1.0


def test_nllb_flores_codes():
    """Ensure NLLB-200 uses FLORES-200 language tags instead of invalid ISO codes."""
    assert translation.NLLB_CODES["en"] == "eng_Latn"
    assert translation.NLLB_CODES["es"] == "spa_Latn"
    assert translation.NLLB_CODES["pt"] == "por_Latn"
    assert translation.NLLB_CODES["de"] == "deu_Latn"
    assert translation.NLLB_CODES["fr"] == "fra_Latn"


def test_translation_availability():
    """Test translation model inspection metadata."""
    cfg = config.load()
    model_dir = Path(cfg["models_dir"]) / cfg.get("translation_model", "nllb-200-distilled-600M")
    available = translation.is_translation_available(str(model_dir))
    info = translation.get_model_info(str(model_dir))
    assert info["available"] is available


def test_native_transcription():
    """Test native transcription for English, Spanish, and French test clips."""
    cfg = config.load()
    model_dir = Path(cfg["models_dir"]) / cfg.get(
        "multilingual_model", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    )
    if not model_dir.is_dir():
        pytest.skip(f"Multilingual model not installed at {model_dir}")

    rec = asr.get_multilingual_offline(cfg)
    model_test_dir = model_dir / "test_wavs"
    if not model_test_dir.is_dir():
        pytest.skip(f"Model test_wavs not found at {model_test_dir}")

    test_files = {
        "en": model_test_dir / "en.wav",
        "es": model_test_dir / "es.wav",
        "fr": model_test_dir / "fr.wav",
    }

    checked = 0
    for lang, path in test_files.items():
        if not path.exists():
            continue
        samples = audio.decode(str(path))
        triples = robust.transcribe_multilingual(
            rec,
            samples,
            cfg,
            source_lang=lang,
            target_lang=None,
            translate=False,
            level=0,
        )
        text = " ".join(t for _, _, t, _ in triples).strip()
        assert len(text) > 0, f"Empty native transcription for {lang}"
        checked += 1

    if checked == 0:
        pytest.skip("No sample wavs available in model_test_dir")


def test_translation():
    """Test fast, accurate NLLB-200 translation when model is installed."""
    cfg = config.load()
    model_dir = Path(cfg["models_dir"]) / cfg.get("translation_model", "nllb-200-distilled-600M")
    if not translation.is_translation_available(str(model_dir)):
        pytest.skip(f"Translation model not installed at {model_dir}")

    t_load0 = time.monotonic()
    translation.get_translator(str(model_dir))
    assert (time.monotonic() - t_load0) < 35.0

    test_cases = [
        ("Hello world, how are you today?", "en", "es"),
        ("Hola mundo, ¿cómo estás hoy?", "es", "en"),
        ("Olá mundo, como você está hoje?", "pt", "en"),
    ]

    for text, src, tgt in test_cases:
        t0 = time.monotonic()
        translated = translation.translate_text(text, src, tgt, str(model_dir))
        elapsed = time.monotonic() - t0
        assert translated.strip(), f"Empty translation for {src}->{tgt}"
        assert translated.lower().strip() != text.lower().strip()
        assert elapsed < 5.0, f"Warm translation took unexpectedly long ({elapsed:.1f}s)"


def test_end_to_end():
    """Test full pipeline: Spanish speech transcription + English translation."""
    cfg = config.load()
    model_dir = Path(cfg["models_dir"]) / cfg.get(
        "multilingual_model", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    )
    trans_dir = Path(cfg["models_dir"]) / cfg.get("translation_model", "nllb-200-distilled-600M")
    es_path = model_dir / "test_wavs" / "es.wav"

    if not model_dir.is_dir() or not es_path.exists():
        pytest.skip("Multilingual model or es.wav not installed")
    if not translation.is_translation_available(str(trans_dir)):
        pytest.skip("Translation model not installed")

    rec = asr.get_multilingual_offline(cfg)
    samples = audio.decode(str(es_path))
    triples = robust.transcribe_multilingual(
        rec,
        samples,
        cfg,
        source_lang="es",
        target_lang="en",
        translate=True,
        level=0,
    )
    assert len(triples) > 0
    src_all = " ".join(src for _, _, src, _ in triples).strip()
    tgt_all = " ".join(tgt or "" for _, _, _, tgt in triples).strip()
    assert len(src_all) > 0
    assert len(tgt_all) > 0
    assert src_all.lower() != tgt_all.lower()


def test_output_formats():
    """Test bilingual output formatters including >24h SRT/VTT timestamps."""
    from stt.cli import _is_bilingual, format_jsonl, format_srt, format_vtt

    regular = [(0.0, 2.5, "Hello world"), (2.5, 5.0, "How are you?")]
    bilingual = [
        (0.0, 2.5, "Hola mundo", "Hello world"),
        (90000.0, 90002.5, "Como estas?", "How are you?"),
    ]

    assert _is_bilingual(bilingual) is True
    assert _is_bilingual(regular) is False

    srt_bilingual = format_srt(bilingual)
    assert "Hola mundo" in srt_bilingual and "Hello world" in srt_bilingual
    assert "25:00:00,000" in srt_bilingual

    vtt_bilingual = format_vtt(bilingual)
    assert "Hola mundo" in vtt_bilingual and "Hello world" in vtt_bilingual

    jsonl_bilingual = format_jsonl(bilingual, "test.wav")
    assert "Hola mundo" in jsonl_bilingual and "Hello world" in jsonl_bilingual
    assert "translated" in jsonl_bilingual


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
