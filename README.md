# stt — offline speech-to-text for Android (Termux / Kali), CPU only

A command-line transcriber built on [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) with int8 models. No cloud, no NPU, no root.

```
stt file a.wav talk.mp4 [--jsonl] [--out FILE] [-a 0-5 | --robust] [-v]
stt mic  [--partial] [--backend pulse|termux] [--whisper | --distant] [-a 0-5] [--threads N]
stt doctor
stt <cmd> --model list          # installed models by kind;  --model NAME picks one (repeatable)
```

## Models
| Role | Model | Used by |
|---|---|---|
| offline ASR | NeMo Parakeet TDT 0.6B v3 int8 (multilingual) | `stt file`, `--whisper`, `--distant`, Termux backend |
| streaming ASR | NeMo Nemotron streaming EN 0.6B (160 ms) int8 — or Kroko zipformer EN (`--model kroko`) | `stt mic` |
| VAD | Silero VAD | everywhere |
| denoiser | DPDFNet-8 (default), DPDFNet-2/4, GTCRN | `-a 2` and up |

## Install
Tested on aarch64 Android (Kali userland and Termux binaries side by side). Python 3.10+.
```sh
apt install python3-venv ffmpeg sox pulseaudio pulseaudio-utils wget      # (root; sox is optional)
python3 -m venv ~/.local/share/stt/venv && . ~/.local/share/stt/venv/bin/activate
pip install -e .                                                           # pulls sherpa-onnx, numpy, soundfile
sh scripts/install_models.sh            # add --all for kroko + the other denoisers
printf '#!/bin/sh\nexec "$HOME/.local/share/stt/venv/bin/stt" "$@"\n' > ~/.local/bin/stt && chmod +x ~/.local/bin/stt
stt doctor
```
A default config is written to `~/.config/stt/config.toml` on first run; every key is documented inline.

## Modes
- **`stt file`** — ffmpeg decode (any format) -> Silero VAD -> Parakeet.
- **`stt mic`** — streaming recognizer with VAD endpointing; `--partial` shows interim text.
- **`stt mic --whisper`** — the VAD cuts utterances; each is normalized/denoised and decoded by Parakeet. Text appears when you pause. Calibrates on 2 s of room noise at start.
- **`stt mic --distant`** — as `--whisper`, with the most sensitive VAD, more gain and level-5 decoding for very quiet speech. Detection, not decoding, is the limit: speech near the noise floor cannot be recovered in software.
- **Mic backends** (`--backend`, auto by default): `pulse` runs a private PulseAudio server with the OpenSL ES source and reads it with `parec`; `termux` records 5 s chunks with Termux:API. Ctrl+C stops recorders and any server stt started.

## Noisy audio: `-a / --aggressiveness` (`--robust` == 3)
| Level | What it does | Mean WER* |
|---|---|---|
| 0 | off | 0.94 |
| 1 | looser VAD | 0.62 |
| 2 | + neural denoise, 50% original blended back | 0.26 |
| 3 | tuned recipe: denoise, 30% blend, permissive VAD, 5 s segments | 0.21 |
| 4 | + second variant, keep the more complete transcript | 0.21 |
| 5 | normalize-only vs level 3, keep the more complete (used by `--distant`) | n/a |

\*10 held-out noisy 128k mp3s (white/pink/babble, SNR -3..5 dB). White noise below about -5 dB and babble below about 2 dB still fail. Higher levels are slower (levels 2+ run a denoiser).

## Performance (4-core phone SoC, idle CPU)
Parakeet offline RTF 0.2 (4 threads, short clips); Nemotron streaming RTF 0.22 with 2 threads (0.39 with 4, so prefer `--threads 2`); Kroko about 0.01. Measure on an idle device; background load easily inflates RTF 5x.

## Troubleshooting
- **Silence / no text on `stt mic`:** run `stt mic --debug --save x.wav`; it prints level, gain and VAD state each second and saves the raw audio so you can replay and tune.
- **Pulse backend can't find a source:** the distro's `pulseaudio` often lacks `module-sles-source`; stt prefers Termux's binary at `termux_bin`.
- **Termux:API recording fails:** the output path must be inside the Termux app's home (`termux_tmp`); the first call after idle can take 10+ s.
- **First words missing:** raise `mic_preroll` (default 2.0 s) or lower `mic_vad_threshold`.

## Repo
`stt/` package · `scripts/install_models.sh` · `experiments/` throwaway lab scripts (noise harness, WER scoring) · `docs/BUILD_LOG.md` full build/tuning log · `HANDOFF.md` state for picking up the project.
Model licenses are their own: see each model's page (Kroko: https://huggingface.co/Banafo/Kroko-ASR).
