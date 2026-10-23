# BUILD_LOG

## Phase 0: Recon (2026-10-21)
- uname -m: aarch64
- Python (Kali, /usr/bin): 3.12.13
- Free in $HOME: ~10 GB (/dev/block/dm-81, 99% used of 936G) -> above 3 GB threshold
- RAM: 15.4 GB total, ~5.3 GB available
- CPU cores: 4 (nproc)
- Termux:API: /data/data/com.termux/files/usr/bin/termux-microphone-record present; `-i` returns {"isRecording": false} (responds)

### Environment discrepancy
The shell is NOT native Termux: it is Kali GNU/Linux (uid 100000 `kali`), HOME=/home/kali, PREFIX unset,
no `pkg`/`termux-*` on PATH, apt present but no root/sudo. Termux's prefix is reachable at
/data/data/com.termux/files/usr (pkg, pulseaudio, ffmpeg, cmake, termux-microphone-record exist there)
and /data/data/com.termux/files/home is a separate dir from /home/kali.
Kali has ffmpeg and parec but no sox/pulseaudio; /dev/snd is permission-denied.
Install target (Termux prefix vs Kali userland) needs a user decision before Phase 1.

## Decisions
- Target: Kali userland (user confirmed apt usable; root-only dpkg needed user's `!` shell). Install paths use /home/kali, wrapper in ~/.local/bin (on PATH), not $PREFIX.
- Termux binaries (pulseaudio w/ module-sles-source, termux-microphone-record) are called from /data/data/com.termux/files/usr/bin. Kali's own pulseaudio lacks module-sles-source -> stt prefers Termux's.

## Phase 1
- apt-get install (as root via user): ffmpeg, sox, pulseaudio, pulseaudio-utils, python3-venv etc. (sox/pulseaudio verified present afterwards)
- venv ~/.local/share/stt/venv; `pip install sherpa-onnx numpy soundfile` -> aarch64 wheel OK, sherpa_onnx 1.13.8, numpy 2.5.4, soundfile 0.14.0. No source build needed.

## Phase 2 (asset names from GitHub API, tag asr-models)
- sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 (641M extracted) — offline
- sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25 (633M) — streaming (published by sherpa-onnx; 160ms chunk variant chosen)
- silero_vad.onnx (632K)
- Tarballs deleted after extraction. Both load with model_type="nemo_transducer" (from_transducer).

## Phase 3/4
- Package ~/src/stt (stt/{cli,config,audio,asr,mic}.py), `pip install -e`, wrapper ~/.local/bin/stt. Config written to ~/.config/stt/config.toml on first run.
- Bug found+fixed: mic pulse backend picked Kali's /usr/bin/pulseaudio (no sles module) -> now uses Termux's binary.
- Termux:API recording path must be under /data/data/com.termux/files/home (Kali ~ is different dir).

## Phase 5
- stt file on parakeet test wavs de/en/es/fr: correct text; RTF 0.19-0.25 (4 threads).
- ogg and mp4 (h264+aac) inputs decode and transcribe identically.
- Mic 5s capture RMS: termux 0.022, pulse 0.036 (both nonzero = real signal). Ctrl+C leaves no parec/pulseaudio/recorder processes, temp chunks cleaned.
- stt doctor: all PASS.
- Streaming (nemotron 160ms int8) real-time factor via chunked feed: ~1.1-1.4 at best (1-3 threads, noisy), 1.7-3.9 at 4+ threads (oversubscription on big.LITTLE hurts). NOT real-time on this CPU; text is correct. Model NOT swapped (spec asks for Nemotron); lower-latency alternative noted: sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06 (57 MB) — untested.
- Offline Parakeet RTF: 0.19-0.25 @4 threads.
- Pending: user-spoken live `stt mic` test.

## Robust (noisy-speech) mode
- Lab: ~/src/stt/experiments/noise_lab.py mixes white/pink/brown/babble noise into test_speech.mp3 at a set SNR (speech-active RMS) and scores WER vs the clean transcript.
- Baseline breakpoints (wav): white ~0 dB, pink ~-5 dB, babble ~5 dB; brown noise never fails (below speech band). On mp3-encoded noisy files the stock VAD returned nothing in most cases.
- Findings: (1) Silero VAD at 0.5 rejects speech in noise -> empty output; (2) recognizer drops the tail of long noisy segments; (3) denoisers help but distort — blending original back fixes much of that; (4) consensus/ROVER-style voting over variants was WORSE than the best single pipeline; (5) hand-rolled spectral subtraction plateaued, sherpa-onnx's DPDFNet-8 beat GTCRN, DPDFNet-2/4.
- Models added (speech-enhancement-models release): dpdfnet8.onnx (14.6MB; also dpdfnet2/4, gtcrn_simple in models dir, unused).
- Final recipe (--robust): dpdfnet8 denoise -> +0.3*original -> peak-normalize 0.7 -> Silero VAD thr 0.1, min_speech 0.1s, min_silence 0.5s, max_speech 5s -> Parakeet.
- Held-out test (new noise seeds 11/12, 128k mp3): base vs robust WER — see final report. Limits: white <= -5 dB, babble <= 2 dB still fail (recognizer limit).

## Aggressiveness levels (stt file/mic -a 0-4; --robust == -a 3; config key `aggressiveness`)
Held-out 10 noisy mp3 (seeds 11/12), mean WER: L0 0.94, L1 0.62, L2 0.26, L3 0.21, L4 0.21. RTF measured in one noisy run: 0.17 / 0.46 / 2.7 / 3.1 / 3.7 (earlier single-file runs showed ~0.8 for L3; timing varies with thermals/threads).
L4 gave no gain over L3 on this set. Code: stt/robust.py.

## Live-mic fixes (user testing)
- Fragmented/dropped words: reader thread drains parec (no pipe overflow when decode < real time); utterance end = vad_min_silence (0.5s) not 4 blocks.
- Missed first words: mic-specific VAD (thr .35, min_speech .1) and 1.0s pre-roll (mic_preroll).
- Whispering (user capture captures/whisper.wav: noise floor ~0.002 RMS, whisper ~0.003-0.005): streaming model drops words; offline Parakeet on raw also drops first words; with -a 3 (denoise+normalize) all 4 utterances decode. Added `stt mic --whisper` (pulse VAD utterances -> robust.transcribe). AGC fixed (env init was too high -> gain stuck at 1.1; now ramps to ~12x). robust.transcribe falls back to whole-clip decode if inner VAD finds nothing (<30s).
- Added `--save WAV` / `--debug` to stt mic for diagnosing levels.

## --distant (very quiet speech) + calibration
- User mic noise (captures/whisper.wav): ~0.002-0.003 RMS but >95% of it is <100 Hz rumble; speech band noise tiny. Whisper ~9.5 dB above in-band noise; 1/4 of whisper ~ -1 dB, 1/8 ~ -7 dB.
- DPDFNet erases speech this quiet (treats as noise) even with pre-gain; noise-lead and profile spectral subtraction did NOT help (WER 0.38->0.48-0.56 at 1/4 level). Simple normalize + loose VAD did best single; level 5 = pick-longest of {normalize-only, L3}: WER 0.28 @1/4, 0.47 @1/8 (L3: 0.38, 0.59).
- Live-style sim (calibration + VAD gating): --whisper finds 0/4 utts at 1/4 level, --distant (Silero thr .25, 1.5s preroll, AGC noise cap .05, calibrated AGC floor) finds 2/4. Detection, not decoding, is the limit.
- Speech-band energy gate (calibrated, 0.5s average) tried and REMOVED: ~60% false alarms because the room noise is bursty.
- Calibration ("Calibrating, please don't speak..." / "OK, noise profile applied") seeds the AGC noise floor; 2s, config mic_calibrate_seconds, --no-calibrate to skip.

## Live captures + model selection
- distant.wav (user, live): --distant caught 4/4 utterances but dropped "Don't" on 3; replay of both captures through the live loop: mic_preroll 2.0 + zero pad = 4/4 full sentences on distant.wav and whisper.wav (1.0 s: 2/4 and 3/4). Real-noise padding did not help (removed). mic_preroll default now 2.0 (distant uses 1.5x).
- `--model list` shows installed models by kind (asr-offline/asr-streaming/denoiser/vad); `--model NAME` (repeatable, prefix match) sets the matching default by kind. gtcrn_simple uses the gtcrn config, dpdfnet* the dpdfnet one. Code: asr.catalog/apply_model_args.

## CORRECTION: streaming speed
Earlier entries claim Nemotron streaming RTF ~1.1-1.4 ("not real-time"). Re-measured with idle CPU: RTF 0.22 @2 threads, 0.39 @4 threads (3 repeats each, stable). The earlier numbers were inflated by CPU contention (an unrelated hermes agent runs on this phone) and warmup; the live-mode lag in the first user test was mostly the dropped-audio bug (fixed with a reader thread). Default thread count 4 is slower than 2 for streaming on this big.LITTLE CPU.
- Added kroko streaming model (sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06, 71MB): RTF ~0.01, cased + punctuated output, correct on test_speech.mp3. Accuracy on hard audio NOT compared. License: see Banafo/Kroko-ASR (not reviewed). Select with `--model kroko`.
- README.md added at ~/src/stt/README.md; removed duplicate noisy/lab_scripts copy.
- HANDOFF.md written (working dir + ~/src/stt) for agent takeover.
