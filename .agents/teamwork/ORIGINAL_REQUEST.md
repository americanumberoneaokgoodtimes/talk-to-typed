# Original User Request

## Initial Request — 2026-10-26T09:59:41Z

This is a single self-contained fix; keep it small and focused.

Upgrade the `stt` CLI in `/home/o/stt` to autonomously calibrate windowing, filtering, and multi-model denoising based on audio duration and noise characteristics, providing deterministic configuration explanations, static progress telemetry, and an interactive parameter wizard (`--interactive` / `-i`).

Working directory: `/home/o/stt`  
Integrity mode: development  

## Requirements

### R1. Dynamic Auto-Tuning Mode (`--auto` & Default Fallback)
The system must dynamically determine optimal window length, step size (stride), acoustic filtering, and denoiser selection based on the duration of the audio file, assuming a heavy noise floor with faint vocal formants. If the user invokes `stt file` without explicit tuning flags (`-a`, `--robust`, `--subnoise`), `--auto` must activate by default.

### R2. Deterministic Configuration Reasoning Output
Before starting audio processing, the system must print a human-readable telemetry block to `sys.stderr` detailing:
- The measured audio duration and acoustic profile.
- Every configured parameter (window duration, step/overlap, rumble filter cutoff, denoiser model, and blank penalty).
- The explicit engineering reasoning behind why each setting was selected for this specific file length.

### R3. Non-Obfuscating Static Progress & ETA Telemetry
Progress percentage, elapsed time, processing speed (RTF), and ETA must update smoothly on a dedicated static terminal line (using pure standard library ANSI escape sequences on `sys.stderr`). The progress telemetry must never interleave with, overwrite, or corrupt streamed transcription text or `--jsonl` records on `sys.stdout`.

### R4. Interactive Parameter Configuration Mode (`--interactive`, `-i`)
An interactive CLI wizard invoked via `stt file -i` or `stt mic -i`:
- Guides the user step-by-step through parameter configuration before execution.
- Displays comprehensive, educational help text for each parameter explaining its acoustic tradeoffs (e.g. rumble cutoff, window length vs gain dilution, denoiser suppression strength vs formant preservation, and transducer blank penalty).
- Offers sensible auto-detected defaults at each prompt that the user can accept by pressing Enter.

### R5. Integration of Subnoise & Multi-Model Formant Recovery
Port the subnoise multi-model formant scanning algorithms developed in `experiments/spanish_deep_scan.py` and `experiments/auto_tune_formants.py` directly into the core `stt` package:
- FFT rumble rejection (<120Hz)
- Multi-model evaluation (`gtcrn_simple.onnx`, `dpdfnet4.onnx`, and normalized high-pass)
- Coherence-scored sliding window transcription that eliminates repetitive token loops and recovers faint multilingual speech (Spanish and English).

### R6. Zero New Dependencies Constraint
All terminal UI formatting, progress reporting, and interactive prompts must rely exclusively on Python standard library modules (`sys`, `time`, `shutil`, `re`, `argparse`). No third-party terminal UI libraries (such as `rich` or `prompt_toolkit`) may be added to `pyproject.toml`.

## Acceptance Criteria

### Auto-Tuning & Reasoning Telemetry
- [ ] Running `stt file "/mnt/c/Users/o/Desktop/MEDIA (documents, videos, audio)/Untitled 3.wav"` without flags triggers auto-mode and prints deterministic configuration reasoning to `sys.stderr`.
- [ ] Running `stt file` on a short clip (<30s) automatically selects dense windowing (e.g. 5.0s window, 1.5s step), while long clips (>2 min) select proportional stride (e.g. 6.0s window, 3.0s step).

### Progress & Telemetry Independence
- [ ] Long audio files display real-time percentage and ETA progress without clobbering or splitting transcribed sentences in `sys.stdout`.
- [ ] `--jsonl` output piped to a file or consumer contains 100% valid JSON without ANSI escape sequences.

### Interactive Mode
- [ ] `stt file -i <path>` prompts the user with detailed educational guidance for each setting and starts transcription with the chosen settings upon completion.
- [ ] Default values shown in prompts can be selected simply by pressing Enter.

### Sub-Noise & Multilingual Recovery
- [ ] `stt file /mnt/g/0-WAV/ishehereblake.wav` recovers Spanish speech segments (`"No, no tengo nada, no"`) without degenerate repetition loops.
- [ ] `stt file "/mnt/c/Users/o/Desktop/MEDIA (documents, videos, audio)/Untitled 3.wav"` transcribes `"Okay."` without flags.
- [ ] `stt doctor` passes with exit code 0 on all core models.
