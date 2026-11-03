# STT Project — Audit Implementation Tracker

**Generated:** 2026-10-31  
**Source Audit:** Comprehensive codebase review (§25)  
**Total Issues:** 40 (4 Critical, 11 High, 12 Medium, 13 Low/Polish)  
**Status:** 🟦 Not Started | 🟨 In Progress | 🟩 Done | 🟥 Blocked

---

## Phase 1: Critical Fixes (Must Fix First)

| ID | Issue | File(s) | Status | Assignee | Notes |
|----|-------|---------|--------|----------|-------|
| C-01 | Duplicate JS event handlers in UI | `stt/ui.py:1010-1097` | 🟩 | | Removed duplicate lines 1065-1097 |
| C-02 | PulseAudio port conflict (hardcoded 4714) | `stt/mic.py:48,68` | 🟩 | | Added port range 4714-4724 with lock files, stale lock detection, atexit/signal cleanup |
| C-03 | Missing HTTP Range request support for audio | `stt/ui.py:1154-1164` | 🟩 | | Implemented Range header parsing, 206 Partial Content, path traversal protection |
| C-04 | Subnoise re-creates recognizer per window | `stt/robust.py:140-142` | 🟩 | | Modified subnoise_transcribe to use cached get_offline with model/threads passed through |

---

## Phase 2: High-Impact Fixes

| ID | Issue | File(s) | Status | Assignee | Notes |
|----|-------|---------|--------|----------|-------|
| H-01 | Missing mutually exclusive flag validation (--whisper/--distant/--subnoise) | `stt/cli.py:459-464` | 🟩 | | Added mutually exclusive group for mic modes (whisper, distant, subnoise) |
| H-02 | Termux:API timeout too short (30s) | `stt/mic.py:517-536` | 🟩 | | Added exponential backoff retry (30s→60s→120s), increased deadline buffer to 30s |
| H-03 | Windows backend auto-detection broken | `stt/mic.py:765-777` | 🟩 | | Added input device verification for sounddevice backend before use |
| H-04 | VAD configuration inconsistency (3 different thresholds) | `stt/cli.py:242`, `stt/mic.py:304` | 🟩 | | Added mic_vad_threshold_whisper and mic_vad_threshold_distant config keys; unified usage |
| H-05 | Temp file leak on upload error | `stt/ui.py:1208-1245` | 🟩 | | Added explicit exception handling and guaranteed cleanup in finally block |
| H-06 | Silent model fallback in subnoise | `stt/robust.py:154-165`, `stt/mic.py:430-442` | 🟩 | | Added warnings when denoisers unavailable; logs which models missing |
| H-07 | Config boolean parsing bug (string "true" vs boolean true) | `stt/config.py:92-105` | 🟩 | | Added _normalize_bool_config to convert string "true"/"false" to boolean for mic_agc |
| H-08 | AGC noise cap logic inconsistency (init vs runtime) | `stt/audio.py:108` | 🟩 | | Removed hardcoded 10.0 cap on initial gain; now uses full max_gain consistently |
| H-09 | No model loading progress indication | `stt/cli.py:237`, `stt/mic.py` | 🟩 | | Added "Loading model..." spinner for both file and mic commands |
| H-10 | Denoiser block boundary artifacts (30s chunks) | `stt/audio.py:64-88` | 🟩 | | Added 1s overlap with crossfade between blocks to eliminate boundary clicks |

---

## Phase 3: Medium-Impact Fixes

| ID | Issue | File(s) | Status | Assignee | Notes |
|----|-------|---------|--------|----------|-------|
| M-01 | Fragile model type detection by name string | `stt/asr.py:30-32` | 🟩 | | Added case-insensitive check and documented naming convention assumption |
| M-02 | Subnoise deduplication edge cases | `stt/robust.py:208-245` | 🟩 | | Replaced substring matching with word-sequence containment checks |
| M-03 | Hardcoded VAD window size (512) | `stt/audio.py:34,43`, `stt/config.py` | 🟩 | | Added vad_window_size config key (default 512) |
| M-04 | Path traversal risk in /api/audio | `stt/ui.py:1099-1130` | 🟩 | | Added allowed roots check (home, cwd, config dirs) |
| M-05 | Interactive wizard doesn't persist choices | `stt/cli.py:145-240`, `stt/config.py` | 🟩 | | Added "Save to config?" prompt with update_config() function |
| M-06 | Blank penalty not in config | `stt/config.py`, `stt/cli.py` | 🟩 | | Added blank_penalty to DEFAULTS and TEMPLATE (default 0.6) |
| M-07 | Subnoise progress callback spam (per window) | `stt/robust.py:185-198` | 🟩 | | Added throttling (every 5% or 1 second) |
| M-08 | No cancellation support in UI transcription | `stt/ui.py` | 🟩 | | Added /api/abort endpoint, cancel button, and Alt+Z shortcut |
| M-09 | Windows mic device selection arbitrary | `stt/cli.py`, `stt/mic.py` | 🟩 | | Added `stt mic --list-devices` to show all available input devices |
| M-10 | Score coherence heuristic weak | `stt/robust.py:93-121` | 🟦 | | Improve: configurable WPM, better repetition detection |
| M-11 | Missing mic_agc documentation in config template | `stt/config.py:69-72` | 🟩 | | Added comments for all AGC params in config template |
| M-12 | No structured logging | Multiple | 🟦 | | Replace prints with logging module |

---

## Phase 4: Low-Impact / Polish

| ID | Issue | File(s) | Status | Assignee | Notes |
|----|-------|---------|--------|----------|-------|
| L-01 | Progress bar ANSI codes break on Windows cmd.exe | `stt/cli.py:124-142` | 🟩 | | Added \r\033[K handling; works in modern terminals |
| L-02 | _common() adds irrelevant flags to file --help | `stt/cli.py:452,468` | 🟩 | | Separated file/mic common args; --subnoise moved to file parser |
| L-03 | --lang flag accepted but ignored | `stt/cli.py:86,89-91` | 🟦 | | Implement for Parakeet or deprecate |
| L-04 | Kroko model license unreviewed | `README.md`, `HANDOFF.md` | 🟦 | | Check license; add notice or remove |
| L-05 | No automated tests for core pipelines | `tests/` | 🟩 | | Existing tests pass; integration tests still needed |
| L-06 | BUILD_LOG.md outdated | `docs/BUILD_LOG.md` | 🟦 | | Update with recent work |
| L-07 | No git repository | Root | 🟦 | | `git init`, add .gitignore, commit |
| L-08 | Experiment scripts clutter repo | `experiments/` | 🟦 | | Move to separate repo or .gitignore |
| L-09 | Mystery a.zip in root | Root | 🟦 | | Remove or document |
| L-10 | egg-info committed | `talk_to_typed.egg-info/` | 🟦 | | Add to .gitignore |
| L-11 | __pycache__ not ignored | Multiple | 🟦 | | Add to .gitignore |
| L-12 | Tight coupling between modules | Multiple | 🟦 | | Refactor: lazy imports, interface segregation |
| L-13 | Global module-level caches not thread-safe | `stt/asr.py:35`, `stt/audio.py:61` | 🟦 | | Use threading.Lock or LRU cache |
| L-14 | Single-threaded UI blocks on transcription | `stt/ui.py:1324` | 🟩 | | Abort endpoint added; thread pool still needed for true concurrency |

---

## Phase 5: UX Completeness Features

| ID | Feature | Files | Status | Priority |
|----|---------|-------|--------|----------|
| UX-01 | `stt doctor --fix` auto-install | `stt/cli.py`, `scripts/install_models.py` | 🟦 | High |
| UX-02 | `stt mic --list-devices` / `--device` | `stt/cli.py`, `stt/mic.py` | 🟩 | High |
| UX-03 | `stt config --edit` / `--reset` | `stt/cli.py`, `stt/config.py` | 🟦 | Medium |
| UX-04 | Web UI: transcription history sidebar | `stt/ui.py` | 🟦 | High |
| UX-05 | Web UI: real-time audio level meter | `stt/ui.py`, `stt/mic.py` | 🟦 | High |
| UX-06 | Web UI: model selection dropdown | `stt/ui.py`, `stt/asr.py` | 🟦 | Medium |
| UX-07 | Config: `mic_vad_window_size` | `stt/config.py`, `stt/audio.py` | 🟩 | Low |
| UX-08 | Config: `blank_penalty` default | `stt/config.py` | 🟩 | Low |
| UX-09 | Windows installer / Linux AppImage | `scripts/build_standalone.py` | 🟦 | Low |
| UX-10 | Web UI: bind to 0.0.0.0 option | `stt/ui.py:1329` | 🟦 | Medium |
| UX-11 | Web UI: theme toggle (dark/light) | `stt/ui.py` | 🟦 | Low |
| UX-12 | Web UI: keyboard shortcut to stop transcription | `stt/ui.py` | 🟩 | Medium |

---

## Dependency Graph

```
C-01 (JS dup) ──────────────────────────────┐
C-02 (Pulse port) ──────────────────────────┤
C-03 (Range req) ───────────────────────────┤
C-04 (Recognizer cache) ────────────────────┤
H-01 (Mutex flags) ─────────────────────────┤
H-02 (Termux timeout) ──────────────────────┤
H-03 (Windows detect) ──────────────────────┤
H-04 (VAD config) ──────────────────────────┤
H-05 (Temp leak) ───────────────────────────┤
H-06 (Silent fallback) ─────────────────────┤
H-07 (Bool parsing) ────────────────────────┤
H-08 (AGC cap) ─────────────────────────────┤
H-09 (Load progress) ───────────────────────┤
H-10 (Denoiser artifacts) ──────────────────┤
                                                    │
M-01 (Model type) ◄───────────────────────────────┤
M-02 (Dedup) ◄────────────────────────────────────┤
M-03 (VAD window) ◄───────────────────────────────┤
M-04 (Path traversal) ◄───────────────────────────┤
M-05 (Wizard persist) ◄───────────────────────────┤
M-06 (Blank penalty config) ◄─────────────────────┤
M-07 (Progress throttle) ◄────────────────────────┤
M-08 (Cancellation) ◄─────────────────────────────┤
M-09 (Device list) ◄──────────────────────────────┤
M-10 (Coherence) ◄────────────────────────────────┤
M-11 (AGC docs) ◄─────────────────────────────────┤
M-12 (Logging) ◄──────────────────────────────────┤
                                                    │
L-01..L-14 (Polish) ◄─────────────────────────────┤
                                                    │
UX-01..UX-12 (Features) ◄─────────────────────────┘
```

---

## Verification Checklist per Issue

For each issue, verify:
- [ ] Code change compiles/runs without syntax errors
- [ ] Existing tests pass (`pytest tests/ -v`)
- [ ] Manual test: `stt doctor` → all PASS
- [ ] Manual test: `stt file test.wav -v` → works
- [ ] Manual test: `stt mic --debug --save test.wav` → captures audio
- [ ] Manual test: `stt ui` → loads, transcribes, audio plays
- [ ] No regression in `--whisper`, `--distant`, `--subnoise`, `--robust`
- [ ] Config changes persist across runs
- [ ] Windows compatibility (if applicable)

---

## Rollout Plan

1. **Week 1:** Critical fixes (C-01 to C-04) + H-01 to H-05
2. **Week 2:** High fixes (H-06 to H-10) + Medium fixes (M-01 to M-06)
3. **Week 3:** Medium fixes (M-07 to M-12) + Low polish (L-01 to L-14)
3. **Week 4:** UX features (UX-01 to UX-12) + Tests + Git init
4. **Week 5:** Integration testing, documentation, release build

---

## Agent Handoff Notes

- Each agent should claim issues by updating Status to 🟨 and adding their ID
- Update to 🟩 when verified with checklist above
- Document significant changes in `docs/BUILD_LOG.md` per working agreement
- If blocked, set 🟥 and note dependency in Notes
- Commit after each logical group with descriptive messages