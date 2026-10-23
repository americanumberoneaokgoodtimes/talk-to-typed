# HANDOFF — `stt` offline speech-to-text CLI

Written 2026-10-22 for an agent taking over. Read this first, then `BUILD_LOG.md` (chronological detail) and
`~/src/stt/README.md` (user-facing usage). Everything below was verified in-session unless marked **UNVERIFIED**.

## 1. What the project is
A command `stt` that transcribes speech fully offline with sherpa-onnx (int8 models, CPU only, no root, no NPU).
- `stt file <paths>` — any ffmpeg-readable audio/video -> text (`--jsonl`, `--out`, `-v` prints RTF).
- `stt mic` — live microphone; streaming (default), or utterance modes `--whisper` / `--distant`.
- `stt doctor` — dependency/model/mic-backend PASS/FAIL table (currently 12 PASS, 0 FAIL).
- `--model list` / `--model NAME` (repeatable, substring match) picks models by kind.
- `-a 0-5` / `--robust` (= 3) "aggressiveness" for noisy audio.

## 2. Environment — read before touching anything
- **The shell is Kali Linux (uid 100000 `kali`, HOME=`/home/kali`), NOT native Termux.** `$PREFIX` is unset, there is no `pkg`.
  The user chose to install into the Kali userland. The original spec said `$PREFIX/bin`; the wrapper is at
  `~/.local/bin/stt` instead (on PATH).
- **`~` is `/home/kali`; the Termux home is a *different* directory** `/data/data/com.termux/files/home`
  (this is also the session's working dir, holding `BUILD_LOG.md`, test audio, `captures/`, `noisy/`).
  Termux binaries live in `/data/data/com.termux/files/usr/bin` and run fine from Kali.
- **No root in the agent shell.** `apt-get install` fails with "requires superuser" here; the user installed the apt packages
  themselves via a `!` command in their own session. Do not use sudo (spec rule). Ask the user to run root commands with `! cmd`.
- Kali's `/usr/bin/pulseaudio` has **no `module-sles-source`** (the phone mic). `stt` deliberately uses **Termux's**
  `/data/data/com.termux/files/usr/bin/pulseaudio`, on its own private server (tcp 127.0.0.1:4714), killed on exit.
  A separate pulseaudio the user already had running (port 4713, pid ~12594) is **not ours — never kill it.**
- **Termux:API recording** (`termux-microphone-record`): output path must be under the *Termux* home
  (`termux_tmp` config key). First call after idle can take >10 s (timeouts are 30 s).
- The phone is shared: an unrelated `hermes` agent runs on it and competes for CPU, which corrupted several of my timing
  measurements. **Re-measure RTFs on an idle box; repeat 3x.**
- 4 cores (`nproc`), 15 GB RAM. For these models 2 threads is usually faster than 4 (big.LITTLE oversubscription).
- Shell gotchas: `pkill -f <pattern>` matches the agent's own shell command line and kills it (exit 144) — use `pgrep -a` + `kill PID`.
  Use `python -I` when running anything near downloaded files. Scratch dir: `/tmp/claude-100000/` (not the project dir).

## 3. Layout
| What | Where |
|---|---|
| wrapper | `~/.local/bin/stt` -> `~/.local/share/stt/venv/bin/stt` (editable install of `~/src/stt`) |
| venv | `~/.local/share/stt/venv` (python 3.12, sherpa-onnx 1.13.8 aarch64 wheel — no source build needed, numpy, soundfile) |
| source | `~/src/stt/stt/{cli,config,audio,asr,robust,mic}.py` |
| models | `~/.local/share/stt/models/` |
| config | `~/.config/stt/config.toml` — created on first run from `config.TEMPLATE`; keys merge over `config.DEFAULTS`. Verified fresh-install writes every key. |
| lab scripts | `~/src/stt/experiments/*.py` — throwaway analysis scripts, not part of the package. Run with `python -I -c "import sys;sys.path.insert(0,'.../experiments');import runpy;runpy.run_path(...)"` (they import `stt` from the venv). |
| user captures | `/data/data/com.termux/files/home/stt/captures/{whisper,distant}.wav` — **real recordings of the user whispering/speaking very quietly; the best tuning data you have** |
| noisy test files | `/data/data/com.termux/files/home/stt/noisy/*.mp3` (10 files, white/pink/babble noise at various SNR) and `.../stt/test_speech.mp3` (clean control) |
| log | `/data/data/com.termux/files/home/stt/BUILD_LOG.md` (keep appending; the user's spec requires it) |

Installed models (`stt <cmd> --model list`):
- `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` (offline, 671 MB, default) — `model_type="nemo_transducer"`
- `sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25` (streaming, default)
- `sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06` (streaming, 71 MB, added last; license not reviewed — see Banafo/Kroko-ASR)
- denoisers: `dpdfnet8.onnx` (default), `dpdfnet4`, `dpdfnet2`, `gtcrn_simple` (from the `speech-enhancement-models` release)
- `silero_vad.onnx`
Model URLs come from the GitHub API for tag `asr-models` / `speech-enhancement-models` of k2-fsa/sherpa-onnx; don't guess names.

## 4. How it works (architecture in one screen)
- `audio.py`: ffmpeg decode to 16 kHz mono f32; Silero VAD wrapper (`make_vad`, `segments`); `denoise` (DPDFNet/GTCRN, 30 s blocks);
  `normalize`; `AGC` (slow auto-gain, noise-floor-capped); `NoiseProfile` (used by calibration for the noise-floor RMS; its
  `subtract()` and lead-in were tested and **do not help recognition**).
- `asr.py`: recognizer builders (`model_type` chosen from the directory name: "nemo" -> `nemo_transducer`, else default),
  `transcribe` (batched offline decode), model `catalog` / `apply_model_args` for `--model`.
- `robust.py`: aggressiveness levels 0-5; `transcribe(rec, samples, cfg, level)` returns `[(start,end,text)]`.
  Level 3 = DPDFNet denoise, +30% original blended back, peak-normalize, permissive VAD (thr 0.1, max seg 5 s).
  Levels 4/5 run two variants and keep the more complete transcript. Falls back to whole-clip decode if the inner VAD finds nothing.
- `mic.py`: `PulseSource` (private pulseaudio + `parec`, **reader thread + unbounded queue** so a slow decode never overflows the pipe),
  `run_pulse` (streaming), `run_pulse_utterance` (`--whisper`/`--distant`; calibration, AGC, VAD utterances, 2 s pre-roll),
  `run_termux` (5 s Termux:API chunks, offline recognizer, worker thread), `detect`.
- `cli.py`: argparse, `--model` resolution happens in `main()` and mutates `cfg` before the subcommand runs.

## 5. Key findings (so you don't redo them)
**Offline accuracy/speed:** Parakeet RTF 0.19-0.25 (4 threads) on clean test wavs; correct on de/en/es/fr samples, mp3/ogg/mp4 inputs.

**CORRECTION to earlier log entries:** Nemotron streaming RTF is **0.22 @2 threads, 0.39 @4 threads** on an idle CPU. Early notes saying
"~1.1-1.4, not real-time" were wrong (contention + warmup). BUILD_LOG.md has a CORRECTION section; the README is consistent with it.
Kroko streaming RTF ~0.01, cased+punctuated output; accuracy on hard audio **not compared**.

**Noisy speech (`-a`):** baseline breaks at ~0 dB white, ~-5 dB pink, ~5 dB babble; brown noise never breaks it. The stock VAD (thr 0.5)
returns nothing on noisy mp3s. Held-out mean WER at levels 0..4: 0.94 / 0.62 / 0.26 / 0.21 / 0.21 (level 4 ≈ level 3; kept because cheap).
Hard floor: white <= -5 dB and babble <= 2 dB still fail — recognizer limit; no swap of models was permitted.
What did NOT work: consensus/ROVER voting (worse than best single), pick-longest across many variants (small gain, 4x cost),
hand-rolled spectral subtraction, noise-profile subtraction, noise lead-in for the denoiser, a speech-band energy gate for `--distant`
(~60% false alarms: the user's room noise is bursty — removed), real-noise padding of utterances (no gain — removed).

**User's mic:** noise ~0.002-0.003 RMS but >95% of it is <100 Hz rumble; speech-band noise is tiny. Whisper is ~0.003-0.005 RMS.
The denoiser erases speech this quiet unless it is level-normalized; the recognizer works best on normalized, not denoised, audio at the very low end.

**Live-mic bugs fixed along the way:** dropped audio from pipe overflow (reader thread); premature utterance ends (end = `vad_min_silence` 0.5 s,
not 4 blocks); clipped first words (mic VAD thr 0.35/min_speech 0.1, **pre-roll now 2.0 s** — 2 s gets 4/4 full sentences on both captures,
1 s got 2-3/4); AGC bug (envelope initialised too high so gain stuck at 1.1); inner-VAD discarding a live-accepted utterance.

## 6. State of verification
Verified: file mode (wav/mp3/ogg/mp4), `--jsonl`/`--out`, `stt doctor` all PASS, both mic backends capture real signal (RMS 0.02-0.04),
Ctrl+C leaves no `parec`/`pulseaudio`/recorder processes or temp chunks, fresh-config generation, `--model list`/`NAME`/ambiguity errors,
`--whisper` live (user: "works quite well"), `--distant` live (4/4 utterances caught, 3 lost "Don't" with the 1 s pre-roll),
replay of both captures through the live loop with the 2 s pre-roll (4/4 each).

**UNVERIFIED (needs the user's voice):**
1. `--distant` live with the new 2.0 s pre-roll (only verified by replaying the saved capture).
2. `--model kroko` in a real `stt mic` session (only run through a simulated source).
3. The Termux:API backend with the user speaking (`stt mic --backend termux`) — only silence/ambient tested.
4. Whether default streaming `stt mic` feels responsive now that the reader-thread fix and true RTF are known (the user last tried it before the fix).
There is no automated test suite. No git repository exists for `~/src/stt` (suggest `git init` + commit).

## 7. Suggested next steps (priority order)
1. Ask the user to run the four unverified live checks above (commands in section 8). Use `--debug --save` and **tune from the saved wav** — this method found every real mic problem.
2. If the user wants streaming accuracy compared: score `kroko` vs `nemotron` on the noisy mp3s / captures (lab scripts in `experiments/` show the WER harness: `noise_lab.py`).
3. Add a minimal pytest suite (decode, VAD segmentation, `--model` resolution, config template round-trip, `robust.transcribe` fallback) and `git init`.
4. Optional: remove unused experiment scripts; review the Kroko license before recommending it.
5. Known small items: `--lang` is accepted but a no-op (both models auto-detect or are English-only); `-a/--robust` is ignored by the streaming pulse backend (prints a note);
   `--partial` has no effect in `--whisper/--distant`.

## 8. Commands
```
stt file test_speech.mp3 -v                       # RTF to stderr
stt file -a 3 noisy/s11_pink_-3.mp3               # noisy-audio mode
stt mic --partial --threads 2                     # streaming, interim text
stt mic --model kroko --partial --threads 2       # faster, punctuated streaming
stt mic --whisper --threads 2                     # whispering / noisy room
stt mic --distant --threads 2 --debug --save /data/data/com.termux/files/home/stt/captures/distant2.wav
stt mic --backend termux -a 3                     # Termux:API chunk backend
stt <cmd> --model list ; stt doctor
```
Run live-mic tests with the user's `!` prefix in their prompt so they can speak; agent-side runs only capture ambient noise.

## 9. Working agreements with the user (from the original spec and the session)
- Keep `BUILD_LOG.md` updated with every command that mattered and its outcome.
- If a step fails twice with the same error, stop, log it and report — don't switch stacks.
- Don't silently swap models/libraries; explain any swap in the log. (Adding selectable models via `--model` is fine and documented.)
- Report outcomes faithfully, including failed experiments and corrections to earlier claims (see the streaming-RTF correction above).
- Never use sudo; never write outside `$HOME`-style user dirs; ask the user to run root commands with `!`.
- Use they/them for anyone whose pronouns aren't stated.
