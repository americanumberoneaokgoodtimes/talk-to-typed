# HANDOFF — `talk-to-typed` (`stt`) Offline Speech-to-Text CLI

Updated November 2026 for any agent or engineer taking over the project. Read this first, followed by [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`README.md`](README.md).

---

## 1. What the Project Is
`talk-to-typed` (`stt`) is an industrial-grade, CPU-only offline speech recognition and audio restoration engine built on `sherpa-onnx` and INT8 quantized neural models.
- **Commands**:
  - `stt file <paths>`: Batch transcription for any multimedia format with automatic VAD-to-subfloor fallback, timestamp formatting, and subtitle sidecar generation (`.srt`, `.vtt`, `.jsonl`, `.txt`).
  - `stt mic`: Real-time streaming transcription (Nemotron / Kroko) or utterance/subfloor scanning (`--whisper`, `--distant`, `--subnoise`).
  - `stt ui`: Browser-based workspace with waveform seek bar, cue selection, drag-and-drop, and live SSE streaming.
  - `stt doctor`: Multi-platform diagnostic tool checking runtime, models, libraries, and microphone hardware.
  - `stt config`: Command-line inspection and atomic updates for configuration keys.

---

## 2. Multi-Platform Environment & Directory Standards
The codebase dynamically detects the operating platform and resolves standard directories using `pathlib.Path`:

| Operating System | Config Path | Model Storage Path | Audio Capture Backend |
| :--- | :--- | :--- | :--- |
| **Linux (Debian/Kali)** | `~/.config/stt/config.toml` | `~/.local/share/stt/models/` | PulseAudio (`parec` + reader thread) |
| **Android (Termux/Userland)** | `~/.config/stt/config.toml` | `~/.local/share/stt/models/` | Termux:API (`termux-microphone-record`) / Termux PulseAudio |
| **Windows (10/11)** | `%APPDATA%\stt\config.toml` | `%LOCALAPPDATA%\stt\models\` | DirectShow (`ffmpeg -f dshow`) / PortAudio (`sounddevice`) |

---

## 3. Core Engine Architecture

```text
stt/
├── asr.py           # Model loading, recognizer caching, single & batched stream decodes
├── audio.py         # Ingestion, zero-phase DSP, AcousticProfile, Silero VAD, DPDFNet/GTCRN
├── cli.py           # Command-line interface, interactive wizard, timeline formatting, dispatch
├── config.py        # Platform-aware config resolution (XDG vs AppData), TOML serialization
├── mic.py           # Multi-platform audio capture backends (Pulse, WASAPI, Termux:API)
├── robust.py        # Aggressiveness levels 0-5, subnoise scanning, deduplication, auto-tuning
├── translation.py   # CTranslate2 engine with NLLB-200 distilled 600M and language detection
└── ui.py            # Zero-dependency HTTP/SSE web interface with waveform seeking
```

### Critical Innovations Implemented in This System:
1. **Adaptive Acoustic Profiler & Hum Notcher ([`stt/audio.py`](stt/audio.py))**:
   - `profile_audio(samples)`: Evaluates 15th percentile noise floor, top-sustained peak energy, power-grid hum prominence (50 Hz vs 60 Hz on quiet frames up to 600 Hz), low-frequency rumble fraction (<120 Hz), and speech-band formant energy (300–3200 Hz).
   - Generates zero-phase Butterworth high-pass cutoff (80 Hz to 130 Hz).
   - Injects harmonic notch filters (`60/120/180/240/300/360/420/480/540/600 Hz` or `50/.../600 Hz`) when hum prominence > 4.0x.
   - Computes dynamic speech formant boost (+0.0 dB to +5.0 dB) and dynamic transducer blank penalty ($0.30$ to $1.05$).
2. **4-Cue Speech Presence Probability & Ephraim-Malah Log-MMSE ([`stt/audio.py`](stt/audio.py))**:
   - `compute_speech_presence_prob`: 50 Hz frame composite of post-denoiser emergence, 2-8 Hz syllabic modulation, formant spectral structure (1 - spectral flatness), and low-band periodicity.
   - `adaptive_spectral_gate`: Rational Padé approximation of Log-MMSE with MCRA noise tracking.
3. **Parallel Stream Decoding & CN-ROVER ([`stt/asr.py`](stt/asr.py), [`stt/robust.py`](stt/robust.py))**:
   - `transcribe_batch` / `decode_streams`: Concurrently decodes 7 enhancement streams (`gtcrn`, `hybrid`, `blend`, `dp4`, `formant_agc`, `spec_gate`, `hp`).
   - `cn_rover_fuse`: Time-synchronous Confusion Network ROVER with token-level confidence weighting and specialist single-stream recovery.
4. **Dual-Layer & Gap Recovery with Burial Telemetry ([`stt/robust.py`](stt/robust.py))**:
   - Recovers loud foreground speech AND scans untranscribed gaps for buried background speech.
   - Computes and logs acoustic burial depth ($\Delta_{\text{floor}}$ and $\Delta_{\text{fg}}$) with timeline timestamps `[MM:SS.mmm -> MM:SS.mmm]`, exported to terminal, `--jsonl`, and Web UI.

---

## 4. Installed Models Summary

| Model Directory | Architecture | Size | Purpose |
| :--- | :--- | :--- | :--- |
| `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` | NeMo Parakeet TDT 0.6B INT8 | 671 MB | Default offline transcriber |
| `sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25` | Nemotron Streaming INT8 | 640 MB | Default low-latency live mic |
| `sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06` | Kroko Zipformer INT8 | 71 MB | Low-resource live mic (`--model kroko`) |
| `silero_vad.onnx` | Silero VAD v4 | 1.8 MB | Voice activity detection |
| `gtcrn_simple.onnx` | GTCRN Complex Denoiser | 5 MB | High-frequency speech formant recovery |
| `dpdfnet8.onnx`, `dpdfnet4.onnx`, `dpdfnet2.onnx` | DPDFNet Denoisers | 18 MB / 9 MB / 4.5 MB | Non-stationary noise suppression |
| `nllb-200-distilled-600M` | CTranslate2 INT8 NLLB-200 | 2.46 GB | Multilingual cross-translation |

---

## 5. Verification Status

Full test suite execution: **51 tests passed**:
```text
pytest -v tests/
====== 48 passed, 2 skipped, 1 xfailed in 237.03s ======
```
- `tests/test_audio.py`: 10/10 passed in 0.73s (Butterworth filter, harmonic notch, speech-band RMS, acoustic profiling).
- `tests/test_cli.py`: 6/6 passed (timeline timestamping, SRT/VTT formatting, recursive media collection).
- `tests/test_config.py`: 3/3 passed (TOML serialization, platform dispatch, path defaults).
- `tests/test_multilingual.py`: passed (NLLB translation, language detection, bilingual formatters).
- `tests/test_pipeline.py`: passed (clean audio, synthetic white/pink noise down to 2.0 dB SNR, level progression, disk round-trip).
- `tests/test_robust.py`: 9/9 passed (auto-tuning, bigram coherence scoring, candidate deduplication).
- `tests/test_ui.py`: 4/4 passed (HTTP web server, API status, SSE streaming endpoints).

---

## 6. Golden Rules for Incoming Agents
1. **Zero New Third-Party Dependencies**: All DSP must remain pure NumPy + standard library. Do not add `scipy`, `torchaudio`, or `librosa`.
2. **Platform Parity**: Never break Linux/Android paths when editing Windows code, and vice versa. Always use `pathlib.Path` and dynamic config keys.
3. **Subprocess Hygiene**: Always register cleanup handlers on `SIGINT`, `SIGTERM`, and `atexit` for child processes (`ffmpeg`, `parec`, `pulseaudio`). Use bounded queues or reader threads to prevent pipe buffer deadlocks.
4. **Transducer Stability**: Do not raise transducer blank penalty above 1.5; excessive blank penalty causes repetitive token loops under pure noise. Rely on `score_coherence()` to reject flat-entropy noise artifacts.
