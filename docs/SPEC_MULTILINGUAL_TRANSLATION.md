# Multilingual Transcription & Translation Specification

**Project**: talk-to-typed (stt)  
**Date**: November 4, 2026  
**Status**: Planning Phase  
**Version**: 1.0

---

## Executive Summary

This specification outlines the complete implementation plan for adding multilingual transcription and translation capabilities to the STT system, including:
- Native transcription for Spanish (es), English (en), Portuguese (pt)
- Cross-language translation (transcribe in source language → translate to target)
- Automatic language detection
- Timestamp support for all output formats
- Updated Web UI with polished UX
- Comprehensive end-to-end testing

---

## 1. Requirements Specification

### 1.1 Supported Languages

| Language | Code | Native Name | Status |
|----------|------|-------------|--------|
| English | `en` | English | ✅ Existing (Parakeet TDT 0.6B) |
| Spanish | `es` | Español | 🔄 New |
| Portuguese | `pt` | Português | 🔄 New |

### 1.2 Core Features

| Feature | Description | Priority |
|---------|-------------|----------|
| Native transcription | Transcribe audio in source language using dedicated model | P0 |
| Translation | Transcribe source → translate to target language | P0 |
| Language detection | Auto-detect spoken language from audio | P0 |
| Timestamps | Per-segment timestamps in all output formats | P0 |
| Language-specific models | Dedicated models per language for best accuracy | P1 |
| Mixed-language handling | Gracefully handle code-switching | P2 |

### 1.3 User Workflows

#### Workflow A: Native Transcription
```
User selects audio file → Selects "Native: Spanish" → Transcribes → Outputs Spanish text with timestamps
```

#### Workflow B: Transcription + Translation
```
User selects audio file → Selects "Source: Spanish" → Selects "Target: English" 
→ Transcribes Spanish → Translates to English → Outputs both with timestamps
```

#### Workflow C: Auto-Detect
```
User selects audio file → Selects "Auto-detect language" → System detects → Transcribes in detected language
```

---

## 2. Architecture Design

### 2.1 Model Strategy

#### Option A: Single Multilingual Model (Recommended for Phase 1)
- Use **Sherpa-ONNX Multilingual Models**:
  - `sherpa-onnx-nemo-parakeet-tdt-1.1b-multilingual` - 28 languages
  - `sherpa-onnx-zipformer-multilingual` - 100+ languages
- Pros: Single model download, handles language detection, code-switching
- Cons: Larger model size, may be slightly less accurate per-language

#### Option B: Per-Language Models (Recommended for Phase 2)
- Download dedicated models:
  - English: `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` (existing)
  - Spanish: `sherpa-onnx-nemo-parakeet-tdt-0.6b-es-int8`
  - Portuguese: `sherpa-onnx-nemo-parakeet-tdt-0.6b-pt-int8`
- Pros: Best accuracy per language, smaller individual models
- Cons: Multiple downloads, need language routing logic

**Decision**: Implement **Option A first** (multilingual model) for immediate value, add Option B as enhancement.

### 2.2 Translation Architecture

```
Audio Input
    │
    ▼
┌─────────────────────────────────────────┐
│  Language Detection (VAD + short chunk) │
└─────────────────────────────────────────┘
    │
    ├── Native Mode ──► Transcribe in detected/source language
    │
    └── Translate Mode ──► Transcribe in source ──► Translate to target
                              │                       │
                              ▼                       ▼
                        Source Text             Target Text
```

**Translation Options**:
1. **Offline MT Model** (recommended): Use CTranslate2 + NLLB or MarianMT
2. **External API** (fallback): LibreTranslate, Google Translate API
3. **Sherpa-ONNX Translation**: Some models support direct translation

**Decision**: Implement **Option 1** (CTranslate2 + NLLB-200) for fully offline operation.

### 2.3 Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        STT System                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Audio I/O  │───►│   VAD/Segment│───►│  Language    │      │
│  │  (ffmpeg)    │    │   (Silero)   │    │  Detection   │      │
│  └──────────────┘    └──────────────┘    └──────┬───────┘      │
│                                                  │              │
│         ┌────────────────────────────────────────┼────────┐     │
│         ▼                                        ▼        ▼     │
│  ┌──────────────┐                        ┌──────────┐ ┌──────┐  │
│  │ Multilingual │                        │  Source  │ │Target│  │
│  │ ASR Model    │                        │  Lang ID │ │LangID│  │
│  └──────┬───────┘                        └──────────┘ └──────┘  │
│         │                                                    │    │
│         ▼                                                    ▼    │
│  ┌──────────────┐                                    ┌──────────┐ │
│  │  Transcript  │                                    │ Translation│
│  │  (source)    │───────────────────────────────────►│  Model    │ │
│  └──────┬───────┘                                    └────┬─────┘ │
│         │                                                 │      │
│         ▼                                                 ▼      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Output Formatter (Timestamps)               │  │
│  │  Text / SRT / VTT / JSONL (bilingual columns)            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Add multilingual model support to `asr.py`
- [ ] Implement language detection in `robust.py`
- [ ] Add CTranslate2 + NLLB translation to `translation.py`
- [ ] Update config schema for language settings
- [ ] Create model download script

### Phase 2: Core Pipeline (Week 2-3)
- [ ] Extend `robust.transcribe()` for multilingual
- [ ] Add translation pipeline integration
- [ ] Implement timestamp preservation through translation
- [ ] Add bilingual output formatters (SRT, VTT, JSONL)

### Phase 3: Web UI (Week 3-4)
- [ ] Redesign sidebar with language controls
- [ ] Add language selector (source/target/auto)
- [ ] Add translation mode toggle
- [ ] Implement bilingual segment display
- [ ] Add language detection indicator
- [ ] Polish visual design (typography, spacing, animations)

### Phase 4: Testing & Validation (Week 4)
- [ ] Download test audio for es, en, pt
- [ ] Create end-to-end test suite
- [ ] Validate translation quality
- [ ] Performance benchmarking
- [ ] Documentation update

---

## 4. Technical Specifications

### 4.1 Configuration Changes

```toml
# ~/.config/stt/config.toml additions

# Multilingual model (replaces offline_model when multilingual mode)
multilingual_model = "sherpa-onnx-nemo-parakeet-tdt-1.1b-multilingual-int8"

# Translation settings
translation_enabled = false
translation_model = "nllb-200-distilled-600M"  # CTranslate2 model
source_language = "auto"  # auto, en, es, pt
target_language = "en"    # en, es, pt

# Language detection
lang_detect_min_duration = 2.0  # seconds for reliable detection
lang_detect_confidence_threshold = 0.7
```

### 4.2 API Changes

#### New CLI Flags
```bash
stt file audio.wav --lang-source es --lang-target en --translate
stt file audio.wav --lang-source auto --lang-target pt --translate
stt file audio.wav --lang-source es  # native transcription only
```

#### New API Endpoints
```http
POST /api/transcribe_path
{
  "strategy": "auto",
  "source_lang": "es",
  "target_lang": "en",
  "translate": true,
  "window_size": 5.0,
  ...
}

SSE Events:
- event: language_detected { lang: "es", confidence: 0.95 }
- event: segment { start: 0.5, end: 3.2, text: "Hola mundo", lang: "es", translated: "Hello world" }
```

### 4.3 Model Requirements

| Model | Size | Source | Languages |
|-------|------|--------|-----------|
| Parakeet TDT 1.1B Multilingual | ~2.2 GB | sherpa-onnx | 28 (incl. es, pt, en) |
| NLLB-200 Distilled 600M | ~1.2 GB | CTranslate2 | 200 languages |
| Silero VAD | 0.6 MB | existing | language-agnostic |

**Total new downloads**: ~3.4 GB

---

## 5. Web UI Design Specification

### 5.1 Sidebar Language Controls

```
┌─────────────────────────────────────────┐
│  Acoustic Strategy                      │
│  ┌─────────────────────────────────┐   │
│  │ Mode ▼                           │   │
│  └─────────────────────────────────┘   │
├─────────────────────────────────────────┤
│  Language & Translation                 │
│  ┌─────────────────────────────────┐   │
│  │ Source Language ▼ [Auto] [ES]   │   │
│  │      [EN] [PT]                   │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │ Target Language ▼ [EN] [ES] [PT]│   │
│  └─────────────────────────────────┘   │
│  ☐ Enable Translation                  │
│  ┌─────────────────────────────────┐   │
│  │ Detected: Spanish (95%) ▶        │   │
│  └─────────────────────────────────┘   │
├─────────────────────────────────────────┤
```

### 5.2 Main Output Area - Bilingual View

```
┌─────────────────────────────────────────────────────────────┐
│ Transcript                              [Text] [Cues] [Bilingual] ▼
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ▼ 00:00.5 → 00:03.2  [ES]  Hola mundo                     │
│       00:00.5 → 00:03.2  [EN]  Hello world                 │
│                                                             │
│  ▼ 00:03.5 → 00:06.1  [ES]  ¿Cómo estás?                   │
│       00:03.5 → 00:06.1  [EN]  How are you?                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Visual Polish Checklist

- [ ] Consistent 8px spacing grid
- [ ] Smooth CSS transitions (150ms ease-out)
- [ ] Loading skeletons for async operations
- [ ] Toast notifications for errors/success
- [ ] Keyboard navigation (Tab order, focus states)
- [ ] Dark/light theme support
- [ ] Responsive layout (mobile-friendly)
- [ ] Accessible color contrast (WCAG AA)
- [ ] Reduced motion support

---

## 6. Testing Strategy

### 6.1 Test Audio Acquisition

| Language | Source | Duration | Content Type |
|----------|--------|----------|--------------|
| English | Existing test_wavs/en.wav | 3.8s | JFK quote |
| Spanish | Download from Common Voice / OpenSpeech | 5-10s | Conversational |
| Portuguese | Download from Common Voice / FalaBrasil | 5-10s | Conversational |
| Mixed (ES+EN) | Synthetic mix | 10s | Code-switching |

### 6.2 Test Matrix

| Test Case | Source | Target | Mode | Expected |
|-----------|--------|--------|------|----------|
| Native ES | es | - | transcribe | Spanish text |
| Native PT | pt | - | transcribe | Portuguese text |
| Native EN | en | - | transcribe | English text |
| Translate ES→EN | es | en | translate | Both columns |
| Translate PT→EN | pt | en | translate | Both columns |
| Translate EN→ES | en | es | translate | Both columns |
| Auto-detect ES | auto | - | transcribe | Detects ES, outputs ES |
| Auto-detect + Translate | auto | en | translate | Detects, outputs both |

### 6.3 Quality Metrics

| Metric | Target |
|--------|--------|
| ES WER (native) | < 8% |
| PT WER (native) | < 10% |
| ES→EN BLEU | > 35 |
| PT→EN BLEU | > 30 |
| Language detection accuracy | > 95% |
| End-to-end latency (5s audio) | < 3s |

---

## 7. File Structure Changes

```
stt/
├── __init__.py
├── config.py          # + multilingual, translation settings
├── asr.py             # + multilingual model loading
├── translation.py     # NEW: CTranslate2 + NLLB wrapper
├── audio.py           # unchanged
├── robust.py          # + language detection, multilingual transcribe
├── cli.py             # + --lang-source, --lang-target, --translate
├── ui.py              # COMPLETE REDESIGN
├── tests/
│   ├── test_multilingual.py      # NEW
│   ├── test_translation.py       # NEW
│   ├── test_language_detection.py # NEW
│   └── test_ui_e2e.py            # NEW
scripts/
├── install_models.sh      # + multilingual, translation models
└── download_test_audio.py # NEW
```

---

## 8. Risk Assessment & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Multilingual model not available for sherpa-onnx | Medium | High | Fallback to per-language models |
| CTranslate2/NLLB compatibility issues | Low | Medium | Use MarianMT as alternative |
| Translation quality insufficient | Medium | High | Add post-editing UI, confidence scores |
| UI performance with bilingual display | Low | Medium | Virtual scrolling, lazy rendering |
| Model download size (3.4 GB) | Certain | Low | Add progress, resume support |
| Language detection on short clips | Medium | Medium | Minimum duration threshold |

---

## 9. Implementation Checklist

### Backend
- [ ] `translation.py`: CTranslate2 wrapper with NLLB-200
- [ ] `asr.py`: `get_multilingual_offline()`, language detection
- [ ] `robust.py`: `transcribe_multilingual()`, `translate_segments()`
- [ ] `config.py`: New config keys, validation
- [ ] `cli.py`: New flags, bilingual output formatters

### Frontend (ui.py)
- [ ] Language selector components (source/target/auto)
- [ ] Translation toggle with visual feedback
- [ ] Bilingual segment renderer
- [ ] Language detection indicator badge
- [ ] Updated export: bilingual SRT/VTT/JSONL
- [ ] Visual polish: spacing, animations, accessibility

### Models & Data
- [ ] `scripts/install_models.sh`: Add multilingual + NLLB
- [ ] `scripts/download_test_audio.py`: Common Voice samples
- [ ] Model verification script

### Testing
- [ ] Unit tests for translation pipeline
- [ ] Integration tests for multilingual transcribe
- [ ] E2E UI tests with Playwright
- [ ] Performance benchmarks
- [ ] Quality evaluation scripts (WER, BLEU)

### Documentation
- [ ] Update README with multilingual usage
- [ ] Add translation troubleshooting guide
- [ ] Model license verification (NLLB is MIT, Parakeet is CC-BY-4.0)

---

## 10. Success Criteria

The feature is complete when:

1. ✅ User can select Spanish audio → get Spanish transcript with timestamps
2. ✅ User can select Spanish audio → get English translation with timestamps
3. ✅ User can select "Auto-detect" → system detects language → transcribes
4. ✅ All output formats (TXT, SRT, VTT, JSONL) support bilingual columns
5. ✅ Web UI is polished, responsive, accessible
5. ✅ All tests pass (unit + integration + E2E)
6. ✅ Performance meets targets (< 3s for 5s audio)
7. ✅ Documentation updated

---

## 11. Dependencies to Add

```txt
# pyproject.toml additions
ctranslate2>=4.0.0
huggingface-hub>=0.20.0  # for model downloads
sentencepiece>=0.1.99    # NLLB tokenizer
```

---

*This specification will be updated as implementation progresses.*