# Comprehensive Audit Report — `talk-to-typed` (stt)

**Date:** 2026-11-10  
**Auditor Model:** claude-sonnet-4-6  
**Repository:** `J:/git/stt` → `https://github.com/americanumberoneaokgoodtimes/talk-to-typed.git`  
**Branch:** `main` (clean, up to date with origin)

---

## Project Context for Incoming Agents

`talk-to-typed` (`stt`) is a production-grade, 100% offline, CPU-only speech-to-text CLI + Web UI built on `sherpa-onnx` with INT8 quantized models. It is **not** a demo — it is a field-ready forensic audio recovery tool.

**Core differentiators:**
- Sub-noise-floor speech recovery via 4-cue Speech Presence Probability (SPP) + 7-stream CN-ROVER hypothesis fusion
- Zero-phase pure-NumPy DSP (no scipy/torch): Butterworth HPF, harmonic notch filters, Ephraim-Malah spectral gating, formant AGC, keyboard click suppression
- Multi-platform: Linux (PulseAudio), Android Termux, Windows (DirectShow/WASAPI)
- Web UI with SSE streaming, drag-and-drop, waveform audio player, keyboard shortcuts

**Key source files:** `stt/cli.py` (1182L), `stt/ui.py` (1914L), `stt/robust.py` (1467L), `stt/audio.py` (1930L), `stt/asr.py` (362L), `stt/mic.py` (996L), `stt/config.py` (210L), `stt/translation.py` (375L), `stt/matrix.py` (471L)

**Test suite:** `tests/` — 10+ test files covering DSP, CLI, config, pipeline, robust, mic, UI, matrix

**Zero-dependency rule:** All DSP must remain pure NumPy + stdlib. No `scipy`, `torchaudio`, or `librosa`.

---

## Summary Counts

| Severity | Count |
|---|---|
| **Critical** | 3 |
| **High** | 9 |
| **Medium** | 10 |
| **Low / Polish** | 12 |
| **TOTAL** | **34** |

---

## Critical Issues

- [ ] **C-01 — `progress_report` callback signature mismatch (TypeError crash in standard→subnoise escalation)**
  - **File:** `stt/cli.py:607-611`
  - **Description:** In `cmd_file`, the standard (non-subnoise) transcription path defines `progress_report()` with **zero parameters** and passes it as `progress_cb` to `robust.transcribe()`. When all standard levels 0–5 find no speech, `robust.transcribe` escalates to `subnoise_transcribe`, which calls `progress_cb(idx + 1, total_wins, t1, dur)` — **4 arguments**. This triggers `TypeError: progress_report() takes 0 positional arguments but 4 were given`, crashing file transcription of any audio where standard VAD completely fails.
  - **Fix:** Change `def progress_report():` to `def progress_report(*_args):` or align its signature with `(current_step, total_steps, current_sec, total_sec)`.
  - **Affected path:** `stt file <audio>` on any file where VAD finds no speech segments at any aggressiveness level.

- [ ] **C-02 — `DEPLOYMENT_STATUS.md` contains corrupted AI agent tool-call artifacts**
  - **File:** `DEPLOYMENT_STATUS.md:48-52`
  - **Description:** Lines 48–52 contain raw, unformatted previous AI agent output leaking `</think>`, `<tool_call>`, `<function=shell>`, and `<parameter=command>` XML tags directly in the markdown. This is a corrupted document artifact from an earlier agent session. It will confuse any agent or human reading it.
  - **Fix:** Remove the artifact lines (48–52). The document ends cleanly at line 47. The document should end with the FINAL STEP section.

- [ ] **C-03 — Web UI audio player broken for drag-and-drop file uploads**
  - **File:** `stt/ui.py` — `handle_transcribe_upload()` / JS audio player setup
  - **Description:** When a user uploads files via drag-and-drop or the "Choose File" button, the server writes them to temp files, transcribes, then deletes them in `finally`. The JS audio player (`<audio id="audioPlayer">`) is populated with `/api/audio?path=<original_tmp_path>` after transcription — but the temp file is already deleted. The audio player appears but serves a 404, so seekable playback doesn't work for uploaded files. Only server-path transcription (via the "Or enter local file/directory path" text field) has a working audio player.
  - **Fix:** Either (a) keep temp files in a session cache for the audio player's lifetime (clean up on new transcription or server restart), or (b) hide the audio player when upload mode is used and show it only for server-path mode.

---

## High-Impact Issues

- [ ] **H-01 — Web UI missing `--dynamic` mode (CN-ROVER 4-model ensemble)**
  - **File:** `stt/ui.py:583-589` — strategy `<select>` dropdown
  - **Description:** The strategy dropdown has: `auto`, `subnoise`, `standard`, `robust`, `distant`. It is missing `dynamic` (the 4-model `--dynamic` CN-ROVER ensemble with cohesive sentence reconstitution), which is the most powerful mode for recovering faint speech. Task 2.4 in `docs/TASK_TRACKER.md` explicitly calls this out as incomplete.
  - **Fix:** Add `<option value="dynamic">Dynamic CN-ROVER Ensemble (4-Model)</option>` to the strategy dropdown and handle `strategy == "dynamic"` in `stream_transcription_sse()`.

- [ ] **H-02 — Web UI missing `--all` multi-model matrix comparison mode**
  - **File:** `stt/ui.py` — no `/api/transcribe_matrix` endpoint or `--all` strategy option
  - **Description:** The CLI's `stt file --all` runs 18 model/enhancement combinations with color-coded output. This mode (implemented in `stt/matrix.py`) is not exposed in the web UI at all. `docs/TASK_TRACKER.md` Task 2.4 covers this.
  - **Fix:** Add a "Matrix (All Models)" strategy option to the UI, or a separate "Compare All Models" button that calls a new `/api/transcribe_matrix` endpoint.

- [ ] **H-03 — Web UI missing `--range` time-range slicing**
  - **File:** `stt/ui.py` — no time-range input field
  - **Description:** `stt file --range` supports millisecond-exact slice transcription (e.g., `00h00m07s635z-00h00m10s000z`). The CLI parses this via `audio.parse_time_range`. The web UI provides no way to transcribe a time slice, limiting it to full-file processing only.
  - **Fix:** Add a "Time Range" input row (optional) to the sidebar. When non-empty, pass `start_sec` and `end_sec` to `audio.decode()` in `stream_transcription_sse`.

- [ ] **H-04 — Web UI: no transcription history sidebar (UX-04)**
  - **File:** `stt/ui.py` — no history panel
  - **Description:** Previous transcription sessions are lost on new file load. The README's feature list implies persistent session history. No session history sidebar or list exists in the current UI.
  - **Fix:** Add a collapsible sidebar panel listing past transcriptions (filename, segment count, timestamp) with ability to reload them into the output area.

- [ ] **H-05 — Web UI: no real-time audio level meter (UX-05)**
  - **File:** `stt/ui.py` — no VU meter or level canvas
  - **Description:** The UI provides no visual feedback on input audio level / signal quality before or during transcription. Users processing faint audio have no way to gauge whether their file will trigger sub-noise recovery.
  - **Fix:** Add a small level-meter canvas (or a static SNR badge from the auto-tune reasoning block) showing the detected SNR and noise floor after auto-profiling.

- [ ] **H-06 — Web UI: no model selection dropdown (UX-06)**
  - **File:** `stt/ui.py:1742-1744` — model is always `None`
  - **Description:** `asr.get_offline(cfg, model=None, ...)` is always called with `model=None`, using the default from config. Users cannot select between installed ASR models (Parakeet vs. Nemotron vs. Kroko), denoiser variants, or custom model paths from the UI.
  - **Fix:** Populate a `<select id="modelSelect">` from `/api/status` or a new `/api/models` endpoint that calls `asr.catalog(cfg)`. Pass the selected model name to `stream_transcription_sse`.

- [ ] **H-07 — `stt doctor --fix` auto-install missing (UX-01)**
  - **File:** `stt/cli.py:896-947` — `cmd_doctor`
  - **Description:** `stt doctor` only diagnoses failures. When a model is FAIL (e.g. not downloaded), users must manually run `scripts/install_models.py`. There is no `stt doctor --fix` that automatically downloads missing models and runtime components.
  - **Fix:** Add a `--fix` flag to the doctor command that invokes `scripts/install_models.py --all` for missing FAIL items.

- [ ] **H-08 — `stt config --edit` subcommand missing (UX-03 partial)**
  - **File:** `stt/cli.py:1133-1157` — config subparser
  - **Description:** `stt config` only has `show`, `set`, `reset`. There is no `--edit` subcommand to open the config file in `$EDITOR`, which is the standard pattern for config management. Users must manually navigate to the platform-specific config path.
  - **Fix:** Add `stt config edit` that opens `config.CONFIG_PATH` in `$EDITOR` (or `notepad.exe` on Windows), falling back to printing the path if `$EDITOR` is not set.

- [ ] **H-09 — Progress bar not shown for standard-mode transcription in Web UI**
  - **File:** `stt/ui.py:1829-1833`
  - **Description:** In `stream_transcription_sse`, `progress_cb` is only passed to `robust.transcribe` when `is_subnoise=True`. For `strategy in ("standard", "robust", "distant")`, `progress_cb=None` is passed, so no progress events are sent to the browser. The progress bar stays at 0% and hidden during all standard-mode transcription jobs.
  - **Fix:** Always pass `progress_cb` regardless of mode, or at minimum send start/done events to show the progress bar.

---

## Medium-Impact Issues

- [ ] **M-01 — `--lang` flag accepted but ignored (L-03)**
  - **File:** `stt/cli.py:220-221`, `stt/cli.py:244`
  - **Description:** `--lang LANG` is in the help text with note "ignored by models that auto-detect or are English-only" but is never connected to any pipeline. It should either be wired into the multilingual pipeline as an alias for `--lang-source`, or removed from the argument parser with a deprecation message.
  - **Fix:** Deprecate `--lang` or alias it to `--lang-source`.

- [ ] **M-02 — Coherence scoring heuristic lacks configurable WPM cap and refined repetition detection (M-10)**
  - **File:** `stt/robust.py:322-361` — `score_coherence`
  - **Description:** The WPM cap (`wps_max = 8.0 / 5.0`) is hardcoded. The bigram repetition check only fires for sequences ≥ 4 words. The function doesn't handle cross-sentence repetition (e.g. "Thank you. Thank you. Thank you." across multiple calls). No config key exposes the WPM threshold.
  - **Fix:** Expose `max_wps` as a config key. Add a unigram n-gram check across the full transcript history for prior-collapse common phrases.

- [ ] **M-03 — No structured logging; all output uses `print()` and `sys.stderr.write()` (M-12)**
  - **File:** Multiple — `stt/cli.py`, `stt/robust.py`, `stt/audio.py`, `stt/mic.py`, `stt/ui.py`
  - **Description:** The codebase uses raw `print(..., file=sys.stderr)` and `sys.stderr.write()` throughout. This prevents log level filtering, makes testing stderr output fragile, and prevents integration with log aggregators. A `--verbose` / `--quiet` flag exists but is not propagated to modules.
  - **Fix:** Replace stderr prints with `logging.getLogger("stt")` with appropriate levels (DEBUG, INFO, WARNING). Keep the `[stt]` prefix convention in format strings.

- [ ] **M-04 — Thread-safety of module-level caches incomplete (L-13)**
  - **File:** `stt/asr.py:60-62` — `_offline_cache` has a lock; `stt/audio.py` — VAD model/denoiser loading has no lock
  - **Description:** `asr.py` correctly uses `_offline_cache_lock` for the LRU recognizer cache. However, `audio.py` loads Silero VAD and DPDFNet denoisers into module-level singletons (e.g., `_vad_model`, `_denoise_cache`) without threading locks. The Web UI's `ThreadedHTTPServer` with `daemon_threads=True` can have concurrent transcription requests that race on these singletons.
  - **Fix:** Add `threading.Lock()` guards around lazy-loading of VAD and denoiser model objects in `audio.py`.

- [ ] **M-05 — `--lang-source` / `--lang-target` limited to 3 languages (en/es/pt)**
  - **File:** `stt/cli.py:222-226`
  - **Description:** `--lang-source` and `--lang-target` only accept `auto`, `en`, `es`, `pt`. Meta's NLLB-200 supports 200 languages. The UI's language dropdowns also only show these 3 options. The `translation.py` module accepts any NLLB Flores-200 language code (e.g., `fra_Latn`, `deu_Latn`).
  - **Fix:** Expand choices to cover common European and Asian languages, or accept any string and validate at runtime against the NLLB-200 vocabulary.

- [ ] **M-06 — Web UI: no `--dynamic` mode awareness in sidebar info text**
  - **File:** `stt/ui.py:579-591`
  - **Description:** The sidebar explains "Auto Dynamic Sub-Noise (Recommended)" but doesn't explain the distinction between `auto` (full subnoise scan with profiling), `dynamic` (CN-ROVER 4-model ensemble), and `subnoise` (force-aggressive scan). The label descriptions are ambiguous, which will confuse first-time users.
  - **Fix:** When `dynamic` is added (H-01), add label descriptions clarifying each mode's tradeoffs in the sidebar.

- [ ] **M-07 — `experiments/` directory contains 25+ ephemeral exploration scripts committed to main**
  - **File:** `experiments/` (25 files tracked in git: `auto_tune_formants.py`, `distant_lab*.py`, `explore*.py`, etc.)
  - **Description:** These scripts are R&D exploration artifacts that should not be in the main production branch. They increase repo size and confuse contributors about which code is canonical.
  - **Fix:** Either add `experiments/` to `.gitignore` and `git rm --cached experiments/`, or move to a separate `dev` branch. The core algorithms have already been ported to `stt/`.

- [ ] **M-08 — Web UI: `Alt+Z` cancel shortcut not listed in hotkey guide modal**
  - **File:** `stt/ui.py:1436-1438` — keydown handler; `stt/ui.py:793-807` — modal table
  - **Description:** `Alt+Z` is wired to the Cancel button (and listed in its `title` tooltip) but is missing from the hotkey guide modal table. Users pressing `?` to discover shortcuts won't find Cancel.
  - **Fix:** Add `<tr><td><kbd>Alt+Z</kbd></td><td>Cancel Transcription</td></tr>` to the hotkey table in the modal.

- [ ] **M-09 — `README.md` and `HANDOFF.md` reference stale path `C:/Users/o/Desktop/stt`**
  - **File:** `README.md:230`, `HANDOFF.md:44-57` — file:// links
  - **Description:** Internal documentation file:// links point to the old development path (`C:/Users/o/Desktop/stt`) which no longer exists. The repo now lives at `J:/git/stt`. These links break in editors and confuse agents.
  - **Fix:** Replace all `file:///C:/Users/o/Desktop/stt/` occurrences with relative links (e.g., `./stt/audio.py`), or remove the file:// links and use markdown-relative paths.

- [ ] **M-10 — `scripts/health_monitor.py` exists but is not integrated into `stt doctor`**
  - **File:** `scripts/health_monitor.py` — standalone script
  - **Description:** A `health_monitor.py` script exists but it's not called from `stt doctor` or any CLI command. The `DEPLOYMENT_STATUS.md` claims "Health monitoring active (stt/health_monitor.py)" but it's in `scripts/`, not `stt/`, and there's no invocation path.
  - **Fix:** Either integrate health_monitor checks into `stt doctor` output or document how to use it separately.

---

## Low-Impact / Polish Issues

- [ ] **L-01 — Kroko model license unreviewed (L-04)**
  - **File:** `README.md`, `HANDOFF.md` — model inventory table
  - **Description:** The Banafo Kroko Zipformer model (71 MB, used by `stt mic --model kroko`) is listed in the model inventory but its license has never been reviewed. If its license is incompatible with the project's MIT license, it cannot be bundled or redistributed.
  - **Fix:** Check the Banafo Kroko license at Hugging Face. Add a license notice in README or remove from docs if incompatible.

- [ ] **L-02 — `docs/BUILD_LOG.md` outdated (L-06)**
  - **File:** `docs/BUILD_LOG.md`
  - **Description:** The BUILD_LOG.md has not been updated to reflect any of the significant recent work: sub-noise detection, CN-ROVER, whisper formant tilt EQ, multi-scale SMI scanning, acoustic burial telemetry, or streaming declicker.
  - **Fix:** Append entries for commits since `b854fae` (last update) through `5a83e80` (current HEAD).

- [ ] **L-03 — `benchmarks/` README links non-existent or moved scripts**
  - **File:** `benchmarks/README.md`
  - **Description:** `benchmarks/README.md` documents scripts that were migrated from the root directory. Some references may be stale or the scripts may reference audio file paths that only exist on the original dev machine.
  - **Fix:** Audit benchmark scripts for hardcoded paths and update README.

- [ ] **L-04 — `stt ui` server binds to 127.0.0.1 only; LAN access requires knowing the `--host` flag (UX-10)**
  - **File:** `stt/cli.py:1114` — `--host` default `127.0.0.1`
  - **Description:** No documentation in the help text or README explains how to expose the UI on a local network for access from another device (e.g., a tablet or phone). The flag exists but is undiscoverable.
  - **Fix:** Add a note in `stt ui` help and README: "To expose on LAN: `stt ui --host 0.0.0.0`".

- [ ] **L-05 — Web UI: no dark/light theme toggle (UX-11)**
  - **File:** `stt/ui.py` — CSS uses hardcoded `--bg-dark: #0f1117`
  - **Description:** The UI is permanently dark-themed with no light theme option. While dark is preferred for accessibility in a monitoring context, users in bright environments may need a light theme.
  - **Fix:** Add a theme toggle button to the header that swaps CSS variables between dark and light palettes. Store preference in `localStorage`.

- [ ] **L-06 — No favicon for `stt ui` server (minor but noticeable)**
  - **File:** `stt/ui.py` — `<head>` section
  - **Description:** The web UI has no `<link rel="icon">` tag. Browsers display a broken-icon tab favicon. This signals "unfinished software" to users.
  - **Fix:** Add a simple inline SVG favicon or serve a 32×32 PNG via a `/favicon.ico` route.

- [ ] **L-07 — `stt config edit` / `--edit` subcommand undocumented (UX-03 remainder)**
  - **File:** `stt/cli.py:1133-1157`
  - **Description:** The config subcommand has `show`, `set`, `reset` but no `edit` to open the file in `$EDITOR`. The AUDIT_TRACKER UX-03 marks this as medium priority.
  - **Fix:** Add `stt config edit` that invokes `$EDITOR $CONFIG_PATH` (or `notepad` on Windows).

- [ ] **L-08 — CI pipeline (`ci.yml`) installs `sherpa-onnx` which requires C++ build on some envs**
  - **File:** `.github/workflows/ci.yml:28-33`
  - **Description:** `pip install -e .` will attempt to install `sherpa-onnx` from PyPI. If pre-built wheels are unavailable for a matrix combination (e.g. Python 3.13 + Windows), the CI will fail during dependency install rather than test execution. Currently no wheel availability check or mock is in place.
  - **Fix:** Pin `sherpa-onnx` version to one with known pre-built wheels, or add `--only-binary :all:` flag for CI, or mock `sherpa_onnx` in the test suite.

- [ ] **L-09 — `stt/matrix.py` `--all` mode doesn't expose a Web UI route**
  - **File:** `stt/ui.py` — no `/api/transcribe_matrix` endpoint
  - **Description:** The color-coded multi-model comparison matrix (18 combinations) in `stt/matrix.py` is CLI-only. The `run_matrix()` function prints ANSI-colored output to stdout which is not SSE-streamable in its current form.
  - **Fix:** Refactor `matrix.run_matrix()` to yield structured `{model, text, start, end}` dicts. Add a `/api/transcribe_matrix` SSE endpoint that calls this generator. (This is part of H-02.)

- [ ] **L-10 — Windows installer / Linux AppImage not implemented (UX-09)**
  - **File:** `scripts/build_standalone.py`
  - **Description:** `scripts/build_standalone.py` exists as a placeholder but the Windows installer and Linux AppImage builds are not implemented. There's no PyInstaller spec, Inno Setup script, or AppImageBuilder config.
  - **Fix:** Implement `scripts/build_standalone.py` using PyInstaller + Inno Setup (Windows) and AppImageTool (Linux). Out of scope for immediate work but should be tracked.

- [ ] **L-11 — `talk_to_typed.egg-info/` untracked but present on disk (L-10)**
  - **File:** `.gitignore` — already includes `*.egg-info/`
  - **Description:** `talk_to_typed.egg-info/` is on disk (from `pip install -e .`) but correctly excluded by `.gitignore`. No action needed for git, but it clutters the working directory for new contributors.
  - **Fix:** Already handled by `.gitignore`. Optionally add a note in the development setup instructions.

- [ ] **L-12 — `SPEC_MULTILINGUAL_TRANSLATION.md` and `MULTILINGUAL_IMPLEMENTATION_SUMMARY.md` are stale root-level docs**
  - **File:** `SPEC_MULTILINGUAL_TRANSLATION.md`, `MULTILINGUAL_IMPLEMENTATION_SUMMARY.md`
  - **Description:** These large spec documents (18KB, 5KB) live at the root level but are implementation-complete. They should be moved to `docs/` to reduce root clutter.
  - **Fix:** `git mv SPEC_MULTILINGUAL_TRANSLATION.md docs/` and `git mv MULTILINGUAL_IMPLEMENTATION_SUMMARY.md docs/`.

---

## Feature Completeness Gaps (Roadmap Items)

The following are documented pending features from `AUDIT_TRACKER.md` and `docs/TASK_TRACKER.md` that have not been started:

- [ ] **F-01 — `stt doctor --fix` auto-install models** (UX-01, Task 2.3)
  - Add `--fix` flag to download missing models automatically.

- [ ] **F-02 — SentencePiece BPE hotword biasing** (Task 2.3)
  - Wire user-specified hotword phrases through BPE tokenization into the transducer beam search bias.

- [ ] **F-03 — Terminal ASCII spectrogram (`stt file --inspect`)** (Task 3.2)
  - Unicode Braille character spectrogram in PowerShell/Bash without external tools.

- [ ] **F-04 — Interactive Textual TUI waveform inspector (`stt tui`)** (Task 3.3)
  - Full-terminal UI with keyboard-navigable waveform, VAD intervals, and matrix comparison.

- [ ] **F-05 — Non-autoregressive CTC fallback engine** (Task 3.4)
  - CTC model for prior-collapse sanity checking in CN-ROVER fusion.

---

## Git & Repository Hygiene

- [ ] **G-01 — 276 MB MP3 file `2026-11-09_11-19-20_0001.mp3` present in working directory**
  - **Status:** Correctly gitignored (`*.mp3` in `.gitignore`), **not tracked**. No action needed for git.
  - **Note:** This is a local test/development file and doesn't affect the repository.

- [ ] **G-02 — `experiments/` directory (25 files) tracked in git**
  - **Status:** `git ls-files experiments/ | wc -l` → 25 files ARE committed.
  - **Fix:** Decide: keep in `experiments/` branch or untrack. If untracking: add `experiments/` to `.gitignore`, then `git rm -r --cached experiments/`.

---

## Verification Checklist (for agents implementing fixes)

For every fix implemented, verify:

- [ ] `pytest tests/ -v` → 0 failures
- [ ] `python -m stt file <test.wav>` → transcribes without error
- [ ] `python -m stt file <noisy.wav> --dynamic` → CN-ROVER ensemble runs
- [ ] `python -m stt mic --debug` → captures audio (Linux/Windows)
- [ ] `python -m stt ui` → browser UI loads, file transcription works, audio player works
- [ ] `python -m stt doctor` → all core models PASS
- [ ] `python -m stt config show` → displays current config
- [ ] No ANSI escape sequences in `--jsonl` output
- [ ] Windows path compatibility (use `pathlib.Path`, not hardcoded slashes)

---

## Prioritized Implementation Order

### Sprint 1 — Critical Fixes (must fix before any release)
1. **C-01** — Fix `progress_report` callback signature in `cli.py`
2. **C-02** — Remove corrupted artifact from `DEPLOYMENT_STATUS.md`
3. **C-03** — Fix audio player for drag-and-drop uploads (keep temp file in session cache)

### Sprint 2 — High-Impact UX Completeness
4. **H-01** — Add `dynamic` mode to Web UI strategy dropdown
5. **H-09** — Pass `progress_cb` for all modes in Web UI SSE handler
6. **M-08** — Add Alt+Z to hotkey modal table
7. **M-09** — Fix stale `file://` paths in README.md and HANDOFF.md
8. **H-04** — Web UI transcription history sidebar
9. **H-05** — Web UI audio level meter / SNR badge

### Sprint 3 — Medium Improvements
10. **H-02** — Web UI `--all` matrix mode
11. **H-03** — Web UI `--range` time slicing
12. **H-06** — Web UI model selection dropdown
13. **M-01** — Wire or deprecate `--lang` flag
14. **M-04** — Thread-safety for audio.py VAD/denoiser singleton loads
15. **M-07** — Untrack `experiments/` from git

### Sprint 4 — Infrastructure & Polish
16. **H-07** — `stt doctor --fix` auto-install
17. **H-08** — `stt config edit` subcommand
18. **L-08** — CI sherpa-onnx wheel pinning
19. **L-02** — Update `docs/BUILD_LOG.md`
20. **L-01** — Review Kroko model license
