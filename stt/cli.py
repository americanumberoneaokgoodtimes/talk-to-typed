import argparse
import glob
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

from . import __version__, config, robust

MEDIA_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac",
    ".wma", ".mp4", ".mkv", ".avi", ".mov", ".webm", ".alac", ".aiff"
}


def collect_media_paths(raw_paths: list[str], recursive: bool = False) -> list[str]:
    """Expand file and directory paths, optionally recursing into subdirectories."""
    collected = []
    seen = set()

    def add_file(p: Path):
        abs_str = str(p.resolve())
        if abs_str not in seen and p.suffix.lower() in MEDIA_EXTENSIONS:
            seen.add(abs_str)
            collected.append(str(p))

    for p_str in raw_paths:
        p = Path(p_str)
        if p.is_dir():
            if recursive:
                for root, _, files in os.walk(p):
                    for fname in sorted(files):
                        add_file(Path(root) / fname)
            else:
                for child in sorted(p.iterdir()):
                    if child.is_file():
                        add_file(child)
        elif p.is_file():
            add_file(p)
        else:
            matches = sorted(glob.glob(p_str))
            for m in matches:
                mp = Path(m)
                if mp.is_file():
                    add_file(mp)
                elif mp.is_dir() and recursive:
                    for root, _, files in os.walk(mp):
                        for fname in sorted(files):
                            add_file(Path(root) / fname)
                elif mp.is_dir():
                    for child in sorted(mp.iterdir()):
                        if child.is_file():
                            add_file(child)
    return collected


def _level(args, cfg) -> int:
    if args.aggressiveness is not None:
        return args.aggressiveness
    return 3 if args.robust else int(cfg["aggressiveness"])


def _common(p):
    p.add_argument("--threads", type=int, help="inference threads (default from config, 4)")
    p.add_argument("--model", action="append", metavar="NAME",
                   help="choose a model by name; its kind (offline/streaming ASR, denoiser, VAD) is detected. "
                        "Repeatable. `--model list` shows what is installed")
    p.add_argument("-a", "--aggressiveness", type=int, choices=robust.LEVELS, metavar="0-5",
                   help="noisy-audio effort: 0 off, 1 loose VAD, 2 +denoise, 3 tuned (same as --robust), "
                        "4 +second variant, keep the more complete, 5 = --distant. Higher = slower. File and termux-mic only")
    p.add_argument("--robust", action="store_true", help="shortcut for --aggressiveness 3")
    p.add_argument("--subnoise", action="store_true",
                   help="aggressive multi-model denoising + formant scanning for low-energy speech near or below noise floor")
    p.add_argument("--auto", action="store_true",
                   help="dynamically determine window, step, and filtering based on duration and heavy noise floor")
    p.add_argument("-i", "--interactive", action="store_true",
                   help="interactive parameter wizard with educational guidance before running")
    p.add_argument("--window-size", type=float, help="sliding window length in seconds (subnoise mode)")
    p.add_argument("--step-size", type=float, help="sliding window stride in seconds (subnoise mode)")
    p.add_argument("--rumble-cutoff", type=float, help="high-pass filter cutoff in Hz (default 120)")
    p.add_argument("--formant-boost", type=float, help="formant equalization boost in dB (default 3.0)")
    p.add_argument("--blank-penalty", type=float, help="transducer blank penalty (0.0 to 1.5, default 0.6)")
    p.add_argument("--lang", help="language hint; ignored by models that auto-detect or are English-only")


def _lang_note(args):
    if args.lang:
        print(f"note: --lang {args.lang} ignored (selected models auto-detect / are English-only)", file=sys.stderr)


def make_progress_bar():
    start_time = time.time()
    def progress_cb(current_step, total_steps, current_sec, total_sec):
        now = time.time()
        elapsed = now - start_time
        pct = int((current_step / total_steps) * 100) if total_steps > 0 else 0
        rtf = elapsed / max(current_sec, 0.001)
        rem_sec = max(0.0, total_sec - current_sec)
        eta = rem_sec * rtf
        cur_m, cur_s = divmod(int(current_sec), 60)
        tot_m, tot_s = divmod(int(total_sec), 60)
        eta_m, eta_s = divmod(int(eta), 60)
        bar_len = 20
        filled = int(bar_len * (pct / 100.0))
        bar = "=" * filled + (">" if filled < bar_len else "") + " " * max(0, bar_len - filled - (1 if filled < bar_len else 0))
        line = f"\r\033[K[stt] [{bar}] {pct:3d}% | {cur_m:02d}:{cur_s:02d}/{tot_m:02d}:{tot_s:02d} | Speed: {rtf:.2f}x | ETA: {eta_m:02d}:{eta_s:02d}"
        sys.stderr.write(line)
        sys.stderr.flush()
    return progress_cb


def interactive_wizard(args, cfg, is_file=True):
    print("\n" + "=" * 65, file=sys.stderr)
    print("  STT Interactive Configuration Wizard", file=sys.stderr)
    print("=" * 65, file=sys.stderr)
    print("Configure acoustic and transcription parameters with in-depth guidance.", file=sys.stderr)
    print("Press [Enter] at any prompt to accept the recommended default.\n", file=sys.stderr)

    # 1. Strategy
    if is_file:
        print("[1/5] Processing Strategy & Noise Handling:", file=sys.stderr)
        print("  - 'auto'    : Dynamic sub-noise scanning with rumble removal (best for heavy noise)", file=sys.stderr)
        print("  - 'standard': Fast standard Silero VAD segmentation (best for clean audio)", file=sys.stderr)
        print("  - 'robust'  : Neural denoising + blending (aggressiveness level 3)", file=sys.stderr)
        choice = input("  Select strategy [auto]: ").strip().lower() or "auto"
        if choice == "auto":
            args.auto = True
        elif choice == "robust":
            args.robust = True
        elif choice == "standard":
            args.aggressiveness = 0
            args.auto = False
    else:
        print("[1/5] Microphone Mode & Acoustic Strategy:", file=sys.stderr)
        print("  - 'streaming': Real-time low-latency streaming tokens (Nemotron / Kroko)", file=sys.stderr)
        print("  - 'subnoise' : Continuous sliding ring-buffer scan with multi-model denoising (faint speech)", file=sys.stderr)
        print("  - 'whisper'  : Utterance segmentation on pauses + offline Parakeet (level 3)", file=sys.stderr)
        print("  - 'distant'  : Sensitive VAD for quiet/far-away speech (level 5)", file=sys.stderr)
        choice = input("  Select mic mode [streaming]: ").strip().lower() or "streaming"
        if choice == "subnoise":
            args.subnoise = True
        elif choice == "whisper":
            args.whisper = True
        elif choice == "distant":
            args.distant = True

    # 2. Rumble Rejection High-Pass Filter
    print("\n[2/5] Low-Frequency Rumble Filter (High-pass cutoff in Hz):", file=sys.stderr)
    print("  Mic recordings often have >85% of energy below 100 Hz (HVAC, handling rumble).", file=sys.stderr)
    print("  Stripping this rumble restores dynamic range and prevents VAD saturation.", file=sys.stderr)
    print("  Recommended: 120 Hz (cuts rumble without touching human speech formants).", file=sys.stderr)
    hp_in = input("  High-pass cutoff Hz [120]: ").strip()
    if hp_in.isdigit():
        args.rumble_cutoff = float(hp_in)
    else:
        args.rumble_cutoff = 120.0

    # 3. Formant Equalization Boost
    print("\n[3/5] Speech Formant Equalization Boost (dB):", file=sys.stderr)
    print("  Gently boosts the 300-3200 Hz speech frequency band before denoising,", file=sys.stderr)
    print("  lifting faint vowel and consonant formants above the neural threshold.", file=sys.stderr)
    print("  Recommended: 3.5 dB for quiet/sub-noise speech, 0.0 dB for clean speech.", file=sys.stderr)
    boost_in = input("  Formant boost dB [3.5]: ").strip()
    try:
        args.formant_boost = float(boost_in) if boost_in else 3.5
    except ValueError:
        args.formant_boost = 3.5

    # 4. Transducer Blank Penalty
    print("\n[4/5] Transducer Blank Penalty (0.0 to 1.5):", file=sys.stderr)
    print("  In neural transducers, faint speech causes blank tokens. A small penalty", file=sys.stderr)
    print("  forces phoneme emission. Setting >1.5 may cause repetitive word loops.", file=sys.stderr)
    print("  Recommended: 0.7 for faint speech, 0.0 for standard speech.", file=sys.stderr)
    bp_in = input("  Blank penalty [0.7]: ").strip()
    try:
        args.blank_penalty = float(bp_in) if bp_in else 0.7
    except ValueError:
        args.blank_penalty = 0.7

    # 5. Inference Threads
    print("\n[5/5] Inference Threads:", file=sys.stderr)
    print("  Number of CPU threads allocated to sherpa-onnx inference.", file=sys.stderr)
    def_threads = cfg.get("threads", 4)
    thr_in = input(f"  CPU threads [{def_threads}]: ").strip()
    if thr_in.isdigit():
        args.threads = int(thr_in)

    print("\n" + "=" * 65, file=sys.stderr)
    print("  Configuration complete. Proceeding to execution...", file=sys.stderr)
    print("=" * 65 + "\n", file=sys.stderr)


def cmd_file(args, cfg):
    from . import asr, audio

    _lang_note(args)
    if getattr(args, "interactive", False):
        interactive_wizard(args, cfg, is_file=True)

    resolved_paths = collect_media_paths(args.paths, recursive=getattr(args, "recursive", False))
    if not resolved_paths:
        sys.exit("[stt] error: no supported audio/video files found in given paths")

    is_auto = getattr(args, "auto", False) or (args.aggressiveness is None and not args.robust and not args.subnoise)
    level = _level(args, cfg)
    rec = asr.offline(cfg, args.model, args.threads)
    out = open(args.out, "w") if args.out else sys.stdout
    total_audio = total_proc = 0.0
    total_files = len(resolved_paths)

    try:
        for idx, path in enumerate(resolved_paths):
            if total_files > 1 and not args.jsonl:
                print(f"\n[stt file {idx + 1}/{total_files}] Processing: {path}", file=sys.stderr)

            try:
                samples = audio.decode(path)
            except RuntimeError as e:
                print(f"error: {e}", file=sys.stderr)
                continue

            dur = len(samples) / audio.SR
            t = time.time()

            if is_auto or getattr(args, "subnoise", False):
                auto_params = robust.auto_tune_settings(dur)
                if getattr(args, "rumble_cutoff", None) is not None:
                    auto_params["low_hz"] = float(args.rumble_cutoff)
                if getattr(args, "formant_boost", None) is not None:
                    auto_params["boost_db"] = float(args.formant_boost)
                if getattr(args, "blank_penalty", None) is not None:
                    auto_params["blank_penalty"] = float(args.blank_penalty)
                if getattr(args, "window_size", None) is not None:
                    auto_params["win_sec"] = float(args.window_size)
                if getattr(args, "step_size", None) is not None:
                    auto_params["step_sec"] = float(args.step_size)

                print(f"\n[stt auto-tune] Audio: {path} ({dur:.1f}s)", file=sys.stderr)
                print("[stt auto-tune] Deterministic Configuration Reasoning:", file=sys.stderr)
                for reason in auto_params["reasoning"]:
                    print(f"  • {reason}", file=sys.stderr)

                progress_cb = make_progress_bar()
                triples = robust.transcribe(
                    rec, samples, cfg, level=0, subnoise=True,
                    params=auto_params, progress_cb=progress_cb
                )
                sys.stderr.write("\r\033[K")
                sys.stderr.flush()
            else:
                triples = robust.transcribe(rec, samples, cfg, level, subnoise=False)

            dt = time.time() - t
            total_audio += dur
            total_proc += dt
            if args.verbose:
                print(f"{path}: {dur:.1f}s audio, {dt:.2f}s processing, RTF {dt / max(dur, 1e-9):.3f}", file=sys.stderr)

            lines = []
            for s, e, text in triples:
                if args.jsonl:
                    out.write(json.dumps({"file": path, "start": round(s, 2), "end": round(e, 2), "text": text}, ensure_ascii=False) + "\n")
                else:
                    lines.append(text)
            if not args.jsonl:
                if total_files > 1:
                    out.write(f"== {path}\n")
                if lines:
                    out.write(" ".join(lines) + "\n")
        out.flush()
    finally:
        if args.out:
            out.close()
    if args.verbose and total_audio:
        print(f"total RTF {total_proc / total_audio:.3f}", file=sys.stderr)


def cmd_mic(args, cfg):
    from . import mic

    _lang_note(args)
    if getattr(args, "interactive", False):
        interactive_wizard(args, cfg, is_file=False)
    if _level(args, cfg):
        print("[stt] aggressiveness applies to the termux backend; the streaming pulse backend ignores it", file=sys.stderr)
    backend = args.backend or cfg["mic_backend"]
    avail = mic.detect(cfg)
    if backend == "auto":
        order = [b for b in ("pulse", "windows", "termux") if b in avail and avail[b][0]]
        if not order:
            sys.exit("no mic backend available; run `stt doctor`")
    else:
        order = [backend]
    t0 = time.time()

    def emit(text, start, end):
        if args.jsonl:
            print(json.dumps({"start": round(start - t0, 2), "end": round(end - t0, 2), "text": text}, ensure_ascii=False), flush=True)
        else:
            print(text, flush=True)

    # Ctrl+C / SIGTERM -> KeyboardInterrupt so `finally` blocks reap child processes.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    if getattr(args, "subnoise", False):
        pulse_runner = mic.run_pulse_subnoise
    elif args.whisper or args.distant:
        pulse_runner = mic.run_pulse_utterance
    else:
        pulse_runner = mic.run_pulse

    runners = {
        "pulse": pulse_runner,
        "windows": lambda c, a, e: pulse_runner(c, a, e, src=mic.get_windows_source(c)),
        "termux": mic.run_termux,
    }
    for i, b in enumerate(order):
        try:
            print(f"[stt] backend: {b} — listening, Ctrl+C to stop", file=sys.stderr)
            runners[b](cfg, args, emit)
            return
        except KeyboardInterrupt:
            print("\n[stt] stopped", file=sys.stderr)
            return
        except Exception as e:
            print(f"[stt] backend {b} failed: {e}", file=sys.stderr)
            if i == len(order) - 1:
                sys.exit(1)


def cmd_ui(args, cfg):
    from . import ui
    ui.launch_ui(cfg, host=args.host, port=args.port, open_browser=not args.no_browser)


def cmd_doctor(args, cfg):
    from . import mic

    rows = []

    def add(name, ok, detail="", optional=False):
        rows.append((name, "PASS" if ok else ("N/A" if optional else "FAIL"), detail))

    add("python", sys.version_info >= (3, 10), sys.version.split()[0])
    try:
        import sherpa_onnx
        add("sherpa-onnx", True, sherpa_onnx.__version__)
    except Exception as e:
        add("sherpa-onnx", False, str(e))
    for mod in ("numpy", "soundfile"):
        try:
            m = __import__(mod)
            add(mod, True, getattr(m, "__version__", ""))
        except Exception as e:
            add(mod, False, str(e))
    add("ffmpeg", bool(shutil.which("ffmpeg")), shutil.which("ffmpeg") or "not on PATH")
    add("config", config.CONFIG_PATH.exists(), str(config.CONFIG_PATH))
    md = Path(cfg["models_dir"])
    for key in ("offline_model", "streaming_model"):
        d = md / cfg[key]
        have = all(any((d / f"{s}{x}").exists() for x in (".int8.onnx", ".onnx")) for s in ("encoder", "decoder", "joiner")) and (d / "tokens.txt").exists()
        add(key, have, f"{d}")
    rd = md / cfg["robust_denoiser"]
    add("robust_denoiser", rd.exists(), str(rd), optional=True)
    vad = md / cfg["vad_model"]
    add("vad_model", vad.exists(), str(vad))
    avail = mic.detect(cfg)
    any_mic = False
    for b, (ok, detail) in avail.items():
        any_mic |= ok
        add(f"mic backend: {b}", ok, detail, optional=True)
    w = max(len(r[0]) for r in rows)
    for n, s, d in rows:
        print(f"{n:<{w}}  {s:<4}  {d}")
    all_ok = all(s != "FAIL" for _, s, _ in rows)
    if all_ok and not any_mic:
        print("\nNote: No active microphone backend found. File transcription is fully operational.")
    sys.exit(0 if all_ok else 1)


def main(argv=None):
    cfg = config.load()
    ap = argparse.ArgumentParser(prog="stt", description="Offline speech-to-text (sherpa-onnx, int8, CPU)")
    ap.add_argument("--version", action="version", version=f"stt {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("file", help="transcribe audio/video files")
    f.add_argument("paths", nargs="+", help="audio/video file or directory paths")
    f.add_argument("-r", "--recursive", action="store_true", help="recursively search directories for audio/video files")
    f.add_argument("--jsonl", action="store_true", help="JSON lines with start/end/text")
    f.add_argument("--out", help="write output to this file")
    f.add_argument("-v", "--verbose", action="store_true", help="print timing / real-time factor to stderr")
    _common(f)
    f.set_defaults(fn=cmd_file)

    m = sub.add_parser("mic", help="live microphone transcription")
    m.add_argument("--partial", action="store_true", help="show interim text (pulse backend)")
    m.add_argument("--backend", choices=["pulse", "termux", "windows"], help="override auto-detection")
    m.add_argument("--jsonl", action="store_true")
    m.add_argument("--whisper", action="store_true",
                   help="whispering / noisy room: cut utterances with the VAD, denoise + normalize each, decode with the "
                        "offline model (text appears when you pause; implies -a 3 unless -a is given)")
    m.add_argument("--distant", action="store_true",
                   help="very quiet / far-away speech: like --whisper but with the most sensitive VAD, larger gain "
                        "and a normalize-or-denoise decode (level 5)")
    m.add_argument("--no-calibrate", action="store_true", help="skip the 2 s room-noise calibration at start")
    m.add_argument("--save", metavar="WAV", help="also write the raw captured audio (pulse backend) for tuning")
    m.add_argument("--debug", action="store_true", help="print mic level / AGC gain / VAD state once a second (pulse backend)")
    _common(m)
    m.set_defaults(fn=cmd_mic)

    u = sub.add_parser("ui", help="launch graphical web front-end with file picker and hotkeys")
    u.add_argument("--host", default="127.0.0.1", help="server listen address (default 127.0.0.1)")
    u.add_argument("--port", type=int, default=8080, help="server listen port (default 8080)")
    u.add_argument("--no-browser", action="store_true", help="do not auto-open browser on launch")
    u.set_defaults(fn=cmd_ui)

    d = sub.add_parser("doctor", help="check dependencies, models, mic backends")
    d.set_defaults(fn=cmd_doctor)

    args = ap.parse_args(argv)
    names = getattr(args, "model", None) or []
    if "list" in names:
        from . import asr
        print(asr.format_catalog(cfg))
        return
    if names:
        from . import asr
        cfg = asr.apply_model_args(cfg, names)
        args.model = None  # now carried by cfg
    args.fn(args, cfg)
