# talk-to-typed (`stt`)

> **High-Performance, CPU-Only, Offline Speech-to-Text with Sub-Noise-Floor Formant Recovery**  
> *Native support for Linux, Android (Termux / Kali Userland), and Windows.*

---

## What Makes `talk-to-typed` Different?

Most speech-to-text tools (Whisper, Vosk, Cloud ASR APIs) share a critical blind spot: **they rely on conventional Voice Activity Detectors (VAD)**. 

When audio has heavy background noise, HVAC rumble, electrical hum, or when a speaker whispers quietly, standard VAD algorithms classify the audio as dead silence and drop the file. If you run heavy denoisers alone, they frequently erase delicate high-frequency speech formants.

`talk-to-typed` is engineered from the ground up to solve this:

1. 🎙️ **Sub-Noise-Floor Speech Recovery**: When standard VAD fails, `stt` automatically falls back to an **overlapping sliding-window formant scanner**. It bypasses silence gates, strips rumble, and extracts intelligible words from audio submerged beneath the noise floor.
2. 🎛️ **Built-In Zero-Phase Acoustic Profiling**: No need for external audio editors or DSP plugins. `stt` automatically profiles your audio:
   - Detects 50 Hz vs. 60 Hz electrical grid hum and injects zero-phase harmonic notch filters.
   - Measures low-frequency rumble fraction and applies an adaptive zero-phase Butterworth filter (80–130 Hz).
   - Dynamically boosts human speech formants (300–3200 Hz) and scales neural transducer blank penalties.
3. ⚡ **100% Offline, Private, CPU-Only**: Powered by INT8 quantized models (`sherpa-onnx`). Runs multiple times faster than real-time on standard laptop and phone CPUs. No GPUs, no NPUs, no cloud connections, and zero data leaving your device.
4. ⏱️ **Exact Audio Timeline Timestamps**: Every piece of transcribed speech is tagged with verifiable `[MM:SS.mmm -> MM:SS.mmm]` intervals so you can instantly verify what was said.
5. 🌐 **Integrated Web Interface & Offline Translation**: Includes a built-in browser UI with synchronized audio seeking, as well as offline cross-lingual translation via Meta's NLLB-200.

---

## Quick Start & Installation

### Windows (PowerShell)
```powershell
# 1. Create and activate a virtual environment
python -m venv $env:LOCALAPPDATA\stt\venv
& "$env:LOCALAPPDATA\stt\venv\Scripts\Activate.ps1"

# 2. Install talk-to-typed
pip install -e .

# 3. Download offline INT8 models (ASR, denoisers, VAD)
python scripts/install_models.py --all

# 4. Verify system health
stt doctor
```

### Linux & Android (Termux / Kali)
```sh
# 1. Install prerequisites (Debian/Ubuntu/Kali)
sudo apt update && sudo apt install -y python3-venv ffmpeg pulseaudio pulseaudio-utils wget

# 2. Create virtual environment & install
python3 -m venv ~/.local/share/stt/venv
. ~/.local/share/stt/venv/bin/activate
pip install -e .

# 3. Download models
sh scripts/install_models.sh --all

# 4. Create convenient symlink
mkdir -p ~/.local/bin
printf '#!/bin/sh\nexec "$HOME/.local/share/stt/venv/bin/stt" "$@"\n' > ~/.local/bin/stt
chmod +x ~/.local/bin/stt

# 5. Verify system health
stt doctor
```

---

## Core Capabilities & Commands

```text
stt file <audio/video> [options]    # Transcribe media files or directories
stt mic [options]                   # Live microphone transcription
stt ui [--host HOST] [--port PORT]  # Launch browser-based graphical workspace
stt doctor                          # Diagnostic check of models, runtime & hardware
stt config [get|set|show|reset]     # Manage configuration keys
```

---

### 1. Transcribing Media Files (`stt file`)

Transcribe any multimedia file readable by `ffmpeg` (`.wav`, `.mp3`, `.m4a`, `.flac`, `.mp4`, `.mkv`):

```sh
# Basic transcription (outputs timeline timestamps by default)
stt file interview.mp3

# Batch process an entire directory recursively
stt file C:\Recordings\ -r

# Export standard SubRip (.srt) or WebVTT (.vtt) subtitles
stt file presentation.mp4 --srt
stt file presentation.mp4 --vtt

# Automatically create sidecar subtitle files alongside each video (video.mp4 -> video.srt)
stt file C:\Videos\ -r --sidecar --srt

# Output clean text without timeline timestamps
stt file audio.wav --no-timestamps
```

#### What an Adaptive Subfloor Output Looks Like:
When `stt` encounters challenging audio, it analyzes the acoustic envelope and logs its deterministic reasoning:
```text
[stt auto-tune] Audio: recording.mp3 (25.0s)
[stt auto-tune] Deterministic Configuration Reasoning:
  • Detected 60 Hz electrical grid hum (prominence=61.2x); engaged notches at 60/120/180 Hz.
  • Heavy low-frequency rumble dominant (92.6% energy <120Hz); raised Butterworth HPF to 130 Hz.
  • Speech near or below noise floor (SNR=7.7dB); formant band boost set to +2.5 dB.
  • Transducer blank penalty dynamically tuned to 0.56 based on estimated SNR (7.7 dB).
  • Dynamic speech-band pre-gate threshold synthesized at 0.001475.
[stt] Subnoise mode: duration=25.0s, window=5.0s, step=1.25s, HP=130.0Hz, boost=2.5dB
[00:00.000 -> 00:05.000] Thank you.
[00:10.000 -> 00:15.000] I'm gonna be able to do it.
[00:13.750 -> 00:18.750] I'm going to go to the house.
[00:17.500 -> 00:22.500] I'm not going to be able to do it.
```

---

### 2. Live Microphone (`stt mic`)

Real-time speech recognition directly from your microphone with automatic hardware backend detection (PulseAudio on Linux, DirectShow/WASAPI on Windows, Termux:API on Android):

```sh
# Real-time low-latency streaming transcription
stt mic

# Show interim partial words as you speak
stt mic --partial

# Whispering / Noisy Room Mode: Cuts utterances with VAD, normalizes and denoises each segment
stt mic --whisper

# Distant / Very Quiet Speech: High-sensitivity VAD with automatic subfloor recovery
stt mic --distant

# Continuous Sub-Noise Surveillance: Sliding ring-buffer scan with multi-model enhancement
stt mic --subnoise

# List available input microphones
stt mic --list-devices
```

---

### 3. Graphical Web Interface (`stt ui`)

Launch the built-in browser UI for visual file management and cue seeking:

```sh
stt ui
```
- **Interactive Audio Waveform**: Scrub and click anywhere on the waveform timeline to seek playback.
- **Clickable Transcribed Cues**: Click any transcribed segment to jump playback directly to that timestamp.
- **Drag-and-Drop Ingestion**: Drop audio/video files or folders directly into the browser.
- **Real-Time SSE Streaming**: Watch text appear line-by-line as windows are processed.
- **DSP Control Sliders**: Adjust rumble cutoff, formant boost, and blank penalty in real-time.

---

### 4. Offline Multilingual Transcription & Translation

`talk-to-typed` supports offline language detection, native multilingual transcription (English, Spanish, Portuguese, French, German, etc.), and local neural machine translation via Meta's NLLB-200:

```sh
# Automatically detect spoken language and transcribe natively
stt file audio_spanish.mp3 --lang-source auto

# Transcribe Spanish audio and translate into English
stt file spanish_interview.mp3 --lang-source es --lang-target en --translate

# Generate bilingual sidecar subtitles with both source speech and English translation
stt file presentation_pt.mp4 --lang-source pt --lang-target en --translate --sidecar --srt
```

---

## Aggressiveness Tuning Levels (`-a 0-5`)

For file transcription and microphone utterance modes, you can set the noise-reduction aggressiveness level:

| Level | Strategy | Pre-Processing & Neural Enhancement | Best Suited For |
| :---: | :--- | :--- | :--- |
| **0** | Clean baseline | Standard Silero VAD, zero neural denoising | Studio recordings, podcasts, clean voice |
| **1** | Permissive VAD | Lowers VAD threshold to prevent word clipping | Soft speakers, mild room echo |
| **2** | Mild Denoise | DPDFNet-8 neural denoiser + 50% clean signal blend | Light fan noise, computer hum |
| **3** | Tuned Robust | DPDFNet-8 denoiser + 30% clean blend + 5.0s segmentation | Street noise, cafe chatter, office ambient |
| **4** | Dual Variant | Decodes both clean and denoised audio, keeps longer valid text | Variable or intermittent noise bursts |
| **5** | Extreme / Distant | Gain-normalized DPDFNet-8 vs DPDFNet-4 tournament | Distant or quiet speech; auto-escalates to subnoise |

> **Automatic Fallback**: If a selected level finds no speech, `stt` automatically escalates through higher levels up to subfloor scanning. Disable automatic escalation with `--no-fallback`.

---

## Neural Models Inventory

All models run offline via INT8 ONNX or CTranslate2:

| Role | Default Model Architecture | Format & Size |
| :--- | :--- | :--- |
| **Offline ASR** | NeMo Parakeet TDT 0.6B v3 | INT8 ONNX (671 MB) |
| **Streaming ASR** | NeMo Nemotron 0.6B (160ms) / Kroko Zipformer | INT8 ONNX (640 MB / 71 MB) |
| **Voice Activity Detection** | Silero VAD v4 | INT8 ONNX (1.8 MB) |
| **Speech Enhancement A** | DPDFNet-8 / DPDFNet-4 / DPDFNet-2 | INT8 ONNX (18 MB / 9 MB / 4.5 MB) |
| **Speech Enhancement B** | GTCRN Simple | INT8 ONNX (5 MB) |
| **Offline Translation** | Meta NLLB-200 Distilled 600M | CTranslate2 INT8 (2.46 GB) |

---

## Diagnostics & System Verification

Run `stt doctor` to inspect the local installation, models, and audio hardware:

```sh
stt doctor
```

Run the automated test suite (51 tests covering DSP, CLI, config, pipelines, and UI):

```sh
pytest -v tests/
```

---

## Technical Architecture & Developer Reference

For mathematical equations, DSP diagrams, and developer schematics, see [`docs/ARCHITECTURE.md`](file:///C:/Users/o/Desktop/stt/docs/ARCHITECTURE.md) and [`HANDOFF.md`](file:///C:/Users/o/Desktop/stt/HANDOFF.md).
