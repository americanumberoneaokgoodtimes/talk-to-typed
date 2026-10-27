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
- Files with energy >85% in <100Hz rumble require high-pass filtering before feeding VAD/denoiser.
- When speech is near or below noise floor, DPDFNet-8 alone can erase weak vocal formants.
- Use multi-model denoising (`gtcrn` + `dpdfnet4` + high-pass) and sliding-window formant scanning (`--subnoise`) when standard VAD finds no speech.
