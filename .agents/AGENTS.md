# Project Rules & Architecture: STT (Speech-to-Text CLI)

## 1. Overview & Multi-Platform Support
- **Engine**: Offline speech transcription using `sherpa-onnx` (int8 quantized models, CPU-only).
- **Supported Environments**:
  - **Linux** (Debian/Ubuntu/Kali): Native PulseAudio (`parec`), virtualenv, ffmpeg.
  - **Android (Termux / Kali Userland)**: Termux:API chunk recording (`termux-microphone-record`) or Termux's PulseAudio with `module-sles-source`.
  - **Windows**: DirectShow / WASAPI audio capture via `ffmpeg` or `sounddevice`, native Windows paths (`%APPDATA%`, `%LOCALAPPDATA%`).
- **Core Models**:
  - Offline ASR: NeMo Parakeet TDT 0.6B v3 int8 (multilingual: en, es, pt, de, fr, etc.)
  - Streaming ASR: NeMo Nemotron 0.6B (160ms) / Banafo Kroko zipformer
  - VAD: Silero VAD int8
  - Speech Enhancement / Denoising: GTCRN (`gtcrn_simple.onnx`), DPDFNet (`dpdfnet8.onnx`, `dpdfnet4.onnx`, `dpdfnet2.onnx`)
  - Offline Translation: Meta NLLB-200 distilled 600M via CTranslate2 INT8

## 2. Platform Detection & Directory Standards
- Never overwrite Linux configurations when adding Windows support. Use dynamic platform dispatch:
  - Linux/Android: Config in `~/.config/stt/config.toml`, models in `~/.local/share/stt/models`.
  - Windows: Config in `%APPDATA%\stt\config.toml`, models in `%LOCALAPPDATA%\stt\models`.
  - Path resolution: always use `pathlib.Path` with platform-aware defaults.

## 3. Subprocess & Zombie Process Management
- All child processes (`ffmpeg`, `parec`, `pulseaudio`, `termux-microphone-record`) MUST:
  - Register termination hooks on `SIGINT` / `SIGTERM` / `atexit` / `finally`.
  - Register an error event listener or catch `OSError`/`FileNotFoundError` gracefully.
  - Use bounded queues or reader threads to prevent pipe buffer deadlocks.

## 4. Noisy & Sub-Noise-Floor Speech Handling
- **Pure-NumPy DSP**: No `scipy` or `torch` dependencies allowed in core DSP routines.
- **Rumble Elimination**: Files with energy >85% in <120 Hz rumble require dynamic zero-phase Butterworth high-pass filtering (`butterworth_highpass`, 80 Hz–130 Hz) to avoid dynamic range saturation.
- **Electrical Hum Notching**: Measure 50 Hz vs 60 Hz hum prominence via `profile_audio`; engage zero-phase harmonic notch rejection filters (`harmonic_notch_filter`) when prominence > 4.0x.
- **Formant Lifting & Blank Penalty**: Speech near or below the noise floor requires formant band boosting (300 Hz–3200 Hz) and empirical SNR-scaled transducer blank penalty (0.30–1.05) to prevent silent `<blank>` token drops.
- **Automatic Subfloor Escalation**: When standard VAD levels 0 through 5 detect no segments, the pipeline MUST automatically trigger subfloor formant recovery scanning (`subnoise_transcribe`) instead of abandoning the file.
- **Parallel Stream Decoding**: Multi-model enhancement candidates (`gtcrn`, `dpdfnet4`, `hpf`) MUST be parallelized using `asr.transcribe_batch` via `rec.decode_streams()` to leverage native multithreading.
- **Timeline Timestamping**: All transcribed segments MUST preserve and display audio timeline intervals `[MM:SS.mmm -> MM:SS.mmm]` for verifiable manual auditing.
