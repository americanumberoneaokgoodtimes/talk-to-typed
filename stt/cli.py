import argparse
import glob
import json
import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path

from . import __version__, config, robust

if sys.platform == "win32":
    try:
        os.system("")
    except Exception:
        pass


def _color_enabled() -> bool:
    """Check if ANSI color output should be enabled based on environment and TTY status."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class ColorHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """High-contrast ANSI color help formatter with clean columns and syntax highlighting."""

    def __init__(self, prog, indent_increment=2, max_help_position=30, width=105):
        super().__init__(prog, indent_increment=indent_increment, max_help_position=max_help_position, width=width)
        self.use_color = _color_enabled()

    def format_help(self):
        help_text = super().format_help()
        if not self.use_color:
            return help_text
        return self._colorize(help_text)

    def _colorize(self, text: str) -> str:
        lines = []
        in_examples = False
        for line in text.splitlines():
            # Suppress bare subparser placeholder lines like "  {file,mic,ui,doctor,config}"
            if re.match(r"^\s*\{[a-z0-9_,\-]+\}\s*$", line):
                continue

            # Section: EXAMPLES / SYNTAX / EXAMPLES & USAGE
            if re.match(r"^(EXAMPLES|SYNTAX|EXAMPLES & USAGE):", line.strip(), re.IGNORECASE):
                in_examples = True
                lines.append(f"\033[1;33m{line}\033[0m")
                continue

            if in_examples:
                if re.match(r"^\s+stt\b", line):
                    # Command invocation: highlight command in bold green, flags in cyan
                    cmd_line = re.sub(r"(-{1,2}[a-zA-Z0-9_\-]+)", r"\033[1;36m\1\033[0m", line)
                    cmd_line = re.sub(r"(\bstt(?:\s+[a-z]+)?\b)", r"\033[1;32m\1\033[0m", cmd_line)
                    lines.append(cmd_line)
                elif line.strip().startswith("#"):
                    lines.append(f"\033[90m{line}\033[0m")
                else:
                    lines.append(line)
                continue

            # 1. Usage line
            if re.match(r"^usage:", line, re.IGNORECASE):
                line = re.sub(r"^usage:", "\033[1;32mUSAGE:\033[0m", line, flags=re.IGNORECASE)
                line = re.sub(r"(\bstt(?:\s+[a-z]+)?\b)", r"\033[1;37m\1\033[0m", line)
                lines.append(line)
                continue

            # 2. Section Headings (e.g., 'POSITIONAL ARGUMENTS:', 'OPTIONS:', etc.)
            clean_strip = line.strip()
            if re.match(r"^[A-Z0-9 &_\-\/]+:$", clean_strip) or clean_strip.lower() in ("options:", "positional arguments:", "subcommands:"):
                lines.append(f"\033[1;36m{clean_strip.upper()}\033[0m")
                continue

            # 3. Option definitions: lines starting with indentation and flags
            if line.startswith("  -"):
                m = re.match(r"^(\s{2,})(.*?)(\s{2,}\S.*)$", line)
                if m:
                    indent, inv, desc = m.groups()
                    c_inv = self._colorize_invocation(inv)
                    c_desc = re.sub(r"(\(default[^)]*\))", r"\033[90m\1\033[0m", desc, flags=re.IGNORECASE)
                    lines.append(f"{indent}{c_inv}{c_desc}")
                else:
                    m2 = re.match(r"^(\s{2,})(.*?)$", line)
                    if m2:
                        lines.append(f"{m2.group(1)}{self._colorize_invocation(m2.group(2))}")
                    else:
                        lines.append(line)
                continue

            # 4. Continuation lines in options (descriptions indented >= 24 chars)
            if re.match(r"^\s{24,}\S", line):
                line = re.sub(r"(\(default[^)]*\))", r"\033[90m\1\033[0m", line, flags=re.IGNORECASE)
                lines.append(line)
                continue

            # 5. Positional arguments / subcommands
            m_pos = re.match(r"^(\s{2,})([a-zA-Z0-9_\-]+)(\s{2,})(.*)$", line)
            if m_pos and not line.strip().startswith("-"):
                indent, name, sp, desc = m_pos.groups()
                c_name = f"\033[1;32m{name}\033[0m" if name in ("file", "mic", "ui", "doctor", "config", "show", "set", "reset") else f"\033[33m{name}\033[0m"
                desc_col = re.sub(r"(\(default[^)]*\))", r"\033[90m\1\033[0m", desc, flags=re.IGNORECASE)
                lines.append(f"{indent}{c_name}{sp}{desc_col}")
                continue

            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _colorize_invocation(inv: str) -> str:
        parts = []
        tokens = re.split(r"(-{1,2}[a-zA-Z0-9_\-]+|\{[^}]+\}|[A-Z0-9_\-]+)", inv)
        for tok in tokens:
            if tok.startswith("-"):
                parts.append(f"\033[1;36m{tok}\033[0m")
            elif tok.startswith("{") or (tok and tok[0].isalnum() and not tok.isspace()):
                parts.append(f"\033[33m{tok}\033[0m")
            else:
                parts.append(tok)
        return "".join(parts)


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


def _add_shared_groups(p, is_file: bool = True):
    g_dsp = p.add_argument_group("ACOUSTIC ENHANCEMENT & SUBFLOOR RECOVERY")
    g_dsp.add_argument("--dynamic", action="store_true",
                       help="run 4-model ensemble (gtcrn, hybrid, dpdfnet4, hp) with CN-ROVER fusion and cohesive sentence reconstitution")
    if is_file:
        g_dsp.add_argument("--all", action="store_true",
                           help="run audio through ALL model combinations, enhancements, and parameters with color-coded comparative output")
        g_dsp.add_argument("--subnoise", action="store_true",
                           help="aggressive multi-model denoising + formant scanning for low-energy speech near or below noise floor")
    g_dsp.add_argument("--auto", action="store_true",
                       help="dynamically determine window, step, and filtering based on duration and heavy noise floor")
    g_dsp.add_argument("--rumble-cutoff", type=float, metavar="HZ",
                       help="zero-phase Butterworth high-pass cutoff in Hz (default: 120)")
    g_dsp.add_argument("--formant-boost", type=float, metavar="DB",
                       help="speech formant equalization boost in dB (default: 3.0)")
    g_dsp.add_argument("--blank-penalty", type=float, metavar="FLOAT",
                       help="transducer blank penalty (0.0 to 1.5, default: 0.6)")
    g_dsp.add_argument("--window-size", type=float, metavar="SEC",
                       help="sliding window length in seconds (subnoise mode, default: 5.0)")
    g_dsp.add_argument("--step-size", type=float, metavar="SEC",
                       help="sliding window stride in seconds (subnoise mode, default: 1.25)")
    g_dsp.add_argument("--declick", action=argparse.BooleanOptionalAction, default=True,
                       help="adaptive impulse and keyboard click removal using 2nd-order curvature MAD z-scores "
                            "and Hermite spline inpainting (default: true; disable with --no-declick)")
    g_dsp.add_argument("--declick-threshold", type=float, metavar="FLOAT", default=4.8,
                       help="MAD curvature z-score sensitivity threshold for click detection "
                            "(default: 4.8; lower values like 3.5 are more aggressive)")

    g_noise = p.add_argument_group("NOISE SUPPRESSION & AGGRESSIVENESS")
    g_noise.add_argument("-a", "--aggressiveness", type=int, choices=robust.LEVELS, metavar="0-5",
                         help="noisy-audio effort: 0 off, 1 loose VAD, 2 +denoise, 3 tuned (same as --robust), "
                              "4 +second variant, keep the more complete, 5 = --distant. Higher = slower")
    g_noise.add_argument("--robust", action="store_true", help="shortcut for --aggressiveness 3")
    g_noise.add_argument("--no-fallback", action="store_true",
                         help="disable progressive fallback escalation when initial VAD detects no speech")

    g_trans = p.add_argument_group("MULTILINGUAL & TRANSLATION")
    g_trans.add_argument("--lang", metavar="LANG", help="language hint; ignored by models that auto-detect or are English-only")
    g_trans.add_argument("--lang-source", choices=["auto", "en", "es", "pt"], metavar="{auto,en,es,pt}", default=None,
                         help="source language: auto (detect), en, es, pt")
    g_trans.add_argument("--lang-target", choices=["en", "es", "pt"], metavar="{en,es,pt}", default=None,
                         help="target language for translation: en, es, pt")
    g_trans.add_argument("--translate", action="store_true",
                         help="enable translation from source to target language")

    g_hw = p.add_argument_group("HARDWARE & ENGINE CONTROLS")
    g_hw.add_argument("--threads", type=int, metavar="N", help="CPU inference threads (default from config, 4)")
    g_hw.add_argument("--model", action="append", metavar="NAME",
                      help="choose model by name/path; repeatable. `--model list` shows what is installed")

    g_diag = p.add_argument_group("DIAGNOSTICS & WIZARD")
    g_diag.add_argument("-i", "--interactive", action="store_true",
                        help="interactive parameter wizard with educational guidance before running")


def _common(p):
    _add_shared_groups(p, is_file=True)


def _lang_note(args):
    if args.lang:
        print(f"note: --lang {args.lang} ignored (selected models auto-detect / are English-only)", file=sys.stderr)


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS.mmm or MM:SS.mmm for timeline displays."""
    millis = int(round(seconds * 1000))
    hours, remainder = divmod(millis, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    secs, ms = divmod(remainder, 1000)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


def format_timestamp_srt(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, remainder = divmod(millis, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, remainder = divmod(millis, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _is_bilingual(triples):
    """Check if triples contain bilingual data (4-tuples)."""
    return triples and len(triples[0]) == 4


def format_srt(triples: list[tuple]) -> str:
    """Format as SRT. Supports both 3-tuple (start, end, text) and 4-tuple (start, end, source, target)."""
    blocks = []
    bilingual = _is_bilingual(triples)
    for idx, item in enumerate(triples, 1):
        if bilingual:
            s, e, src, tgt = item
            text = f"{src}\n{tgt}" if tgt else src
        else:
            s, e, text = item
        blocks.append(f"{idx}\n{format_timestamp_srt(s)} --> {format_timestamp_srt(e)}\n{text}\n")
    return "\n".join(blocks)


def format_vtt(triples: list[tuple]) -> str:
    """Format as VTT. Supports both 3-tuple and 4-tuple."""
    blocks = ["WEBVTT\n"]
    bilingual = _is_bilingual(triples)
    for idx, item in enumerate(triples, 1):
        if bilingual:
            s, e, src, tgt = item
            text = f"{src}\n{tgt}" if tgt else src
        else:
            s, e, text = item
        blocks.append(f"{idx}\n{format_timestamp_vtt(s)} --> {format_timestamp_vtt(e)}\n{text}\n")
    return "\n".join(blocks)


def format_jsonl(triples: list[tuple], file: str = "") -> str:
    """Format as JSONL. Supports both 3-tuple and 4-tuple."""
    lines = []
    bilingual = _is_bilingual(triples)
    for item in triples:
        if bilingual:
            s, e, src, tgt = item
            obj = {"file": file, "start": round(s, 2), "end": round(e, 2), "text": src, "translated": tgt}
        else:
            s, e, text = item
            obj = {"file": file, "start": round(s, 2), "end": round(e, 2), "text": text}
        lines.append(json.dumps(obj, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")


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

    # Offer to save to config
    print("\n[Save] Configuration:", file=sys.stderr)
    print("  Save these settings to config file for future runs?", file=sys.stderr)
    save_choice = input("  Save to config? [y/N]: ").strip().lower()
    if save_choice in ('y', 'yes'):
        save_wizard_config(args, cfg, is_file)

    print("\n" + "=" * 65, file=sys.stderr)
    print("  Configuration complete. Proceeding to execution...", file=sys.stderr)
    print("=" * 65 + "\n", file=sys.stderr)


def save_wizard_config(args, cfg, is_file: bool):
    """Save wizard choices to config file."""
    from . import config
    updates = {}
    
    if is_file:
        if getattr(args, "auto", False):
            updates["aggressiveness"] = 0  # auto uses level 0 with subnoise
        elif getattr(args, "robust", False):
            updates["aggressiveness"] = 3
        elif getattr(args, "aggressiveness", None) is not None:
            updates["aggressiveness"] = args.aggressiveness
    else:
        # Mic mode settings
        if getattr(args, "subnoise", False):
            updates["aggressiveness"] = 0  # subnoise uses its own logic
        elif getattr(args, "whisper", False):
            updates["aggressiveness"] = 3
        elif getattr(args, "distant", False):
            updates["aggressiveness"] = 5
    
    if getattr(args, "blank_penalty", None) is not None:
        updates["blank_penalty"] = args.blank_penalty
    if getattr(args, "threads", None) is not None:
        updates["threads"] = args.threads
    
    if updates:
        config.update_config(updates)
        print(f"[stt] Saved {len(updates)} setting(s) to {config.CONFIG_PATH}", file=sys.stderr)
    else:
        print("[stt] No settings to save", file=sys.stderr)


def cmd_file(args, cfg):
    from . import asr, audio

    _lang_note(args)
    if getattr(args, "interactive", False):
        interactive_wizard(args, cfg, is_file=True)

    resolved_paths = collect_media_paths(args.paths, recursive=getattr(args, "recursive", False))
    if not resolved_paths:
        sys.exit("[stt] error: no supported audio/video files found in given paths")

    is_auto = getattr(args, "auto", False)
    level = _level(args, cfg)
    bp = getattr(args, "blank_penalty", None)
    if bp is None:
        bp = cfg.get("blank_penalty")
    
    # Multilingual & Translation args
    lang_source = getattr(args, "lang_source", None) or cfg.get("source_language", "auto")
    lang_target = getattr(args, "lang_target", None) or cfg.get("target_language", "en")
    do_translate = getattr(args, "translate", False) or cfg.get("translation_enabled", False)
    use_multilingual = do_translate or (lang_source not in ("auto", "en"))
    
    is_all = getattr(args, "all", False)
    if not is_all:
        if not args.jsonl:
            print("[stt] Loading model...", file=sys.stderr, flush=True)
        
        if use_multilingual:
            # Load multilingual model for translation or non-English source
            rec = asr.get_multilingual_offline(cfg, args.model, args.threads, blank_penalty=bp)
        else:
            rec = asr.offline(cfg, args.model, args.threads, blank_penalty=bp)
        
        if not args.jsonl:
            print("\r\033[K[stt] Model loaded", file=sys.stderr, flush=True)
    else:
        rec = None
    out = open(args.out, "w") if args.out else sys.stdout
    total_audio = total_proc = 0.0
    total_files = len(resolved_paths)

    try:
        for idx, path in enumerate(resolved_paths):
            if total_files > 1 and not args.jsonl:
                print(f"\n[stt file {idx + 1}/{total_files}] Processing: {path}", file=sys.stderr)

            time_offset = 0.0
            range_arg = getattr(args, "range", None)
            start_sec = None
            end_sec = None
            if range_arg:
                try:
                    start_sec, end_sec = audio.parse_time_range(range_arg)
                    time_offset = start_sec
                except ValueError as ve:
                    sys.exit(f"[stt] error: invalid --range {range_arg!r}: {ve}")

            try:
                samples = audio.decode(path, start_sec=start_sec, end_sec=end_sec)
            except RuntimeError as e:
                print(f"error: {e}", file=sys.stderr)
                continue
            if samples is None:
                continue

            dur = len(samples) / audio.SR
            t = time.time()

            if is_all:
                from . import matrix
                matrix.run_matrix(samples, cfg, audio_path=path, threads=args.threads, time_offset=time_offset)
                total_audio += dur
                total_proc += time.time() - t
                continue

            is_dynamic = getattr(args, "dynamic", False)
            if is_auto or getattr(args, "subnoise", False) or is_dynamic:
                auto_params = robust.auto_tune_settings(dur, samples=samples)
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
                if getattr(args, "declick", None) is not None:
                    auto_params["declick"] = bool(args.declick)
                if getattr(args, "declick_threshold", None) is not None:
                    auto_params["declick_threshold"] = float(args.declick_threshold)

                if is_dynamic:
                    print(f"\n[stt dynamic] Audio: {path} ({dur:.1f}s)", file=sys.stderr)
                    print("[stt dynamic] Multi-Model Dynamic Ensemble: gtcrn, hybrid, dpdfnet4, hp", file=sys.stderr)
                    print("[stt dynamic] Hypotheses Fusion: Confusion Network ROVER + Cohesive Sentence Reconstitution", file=sys.stderr)
                else:
                    print(f"\n[stt auto-tune] Audio: {path} ({dur:.1f}s)", file=sys.stderr)
                    print("[stt auto-tune] Deterministic Configuration Reasoning:", file=sys.stderr)
                    for reason in auto_params["reasoning"]:
                        print(f"  • {reason}", file=sys.stderr)

                progress_cb = make_progress_bar()
                triples = robust.transcribe(
                    rec, samples, cfg, level=0, subnoise=True,
                    params=auto_params, progress_cb=progress_cb,
                    model=args.model, threads=args.threads
                )
                sys.stderr.write("\r\033[K")
                sys.stderr.flush()
            else:
                # Standard transcription
                total_chunks = max(1, len(samples) // (30 * audio.SR))
                print(f"[stt file] Processing {dur:.0f}s audio (~{total_chunks} chunks)...", file=sys.stderr)
                
                chunk_progress = [0]
                def progress_report():
                    chunk_progress[0] += 1
                    pct = min(100, int(chunk_progress[0] / max(total_chunks, 1) * 100))
                    bar = "=" * (pct // 5) + ">" + " " * (20 - pct // 5 - 1)
                    sys.stderr.write(f"\r[stt] [{bar}] {pct:3d}% | Chunk {chunk_progress[0]}/{total_chunks}")
                    sys.stderr.flush()
                
                no_fallback = getattr(args, "no_fallback", False)
                cfg_file = dict(cfg)
                if getattr(args, "declick", None) is not None:
                    cfg_file["declick"] = bool(args.declick)
                if getattr(args, "declick_threshold", None) is not None:
                    cfg_file["declick_threshold"] = float(args.declick_threshold)

                if use_multilingual and (do_translate or lang_source != "en"):
                    # Multilingual transcription with optional translation
                    triples = robust.transcribe_multilingual(
                        rec, samples, cfg_file,
                        source_lang=lang_source,
                        target_lang=lang_target if do_translate else None,
                        translate=do_translate,
                        level=level,
                        progress_cb=progress_report if not no_fallback else None,
                    )
                else:
                    # Standard English-only transcription
                    triples = robust.transcribe(rec, samples, cfg_file, level, subnoise=False, no_fallback=no_fallback)

            if time_offset > 0 and triples:
                if len(triples[0]) == 4:
                    triples = [(s + time_offset, e + time_offset, src, tgt) for s, e, src, tgt in triples]
                else:
                    triples = [(s + time_offset, e + time_offset, text) for s, e, text in triples]

            dt = time.time() - t
            total_audio += dur
            total_proc += dt
            if args.verbose:
                print(f"{path}: {dur:.1f}s audio, {dt:.2f}s processing, RTF {dt / max(dur, 1e-9):.3f}", file=sys.stderr)

            # Sidecar files
            if getattr(args, "sidecar", False):
                p_obj = Path(path)
                if getattr(args, "srt", False):
                    sidecar_path = p_obj.with_suffix(".srt")
                    sidecar_path.write_text(format_srt(triples), encoding="utf-8")
                elif getattr(args, "vtt", False):
                    sidecar_path = p_obj.with_suffix(".vtt")
                    sidecar_path.write_text(format_vtt(triples), encoding="utf-8")
                elif getattr(args, "jsonl", False):
                    sidecar_path = p_obj.with_suffix(".jsonl")
                    sidecar_path.write_text(format_jsonl(triples, path), encoding="utf-8")
                else:
                    sidecar_path = p_obj.with_suffix(".txt")
                    show_ts = getattr(args, "timestamps", True)
                    if show_ts:
                        if triples and len(triples[0]) == 4:
                            lines = [f"[{format_timestamp(s)} -> {format_timestamp(e)}] {src} / {tgt}" if tgt else f"[{format_timestamp(s)} -> {format_timestamp(e)}] {src}" for s, e, src, tgt in triples]
                        else:
                            lines = [f"[{format_timestamp(s)} -> {format_timestamp(e)}] {t}" for s, e, t in triples]
                        sidecar_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    else:
                        if triples and len(triples[0]) == 4:
                            lines = [f"{src} / {tgt}" if tgt else src for _, _, src, tgt in triples]
                        else:
                            lines = [t for _, _, t in triples]
                        sidecar_path.write_text(" ".join(lines) + "\n", encoding="utf-8")
                if args.verbose:
                    print(f"[stt] saved sidecar: {sidecar_path}", file=sys.stderr)

            # Main output
            if getattr(args, "srt", False):
                if total_files > 1 and not args.out:
                    out.write(f"== {path}\n")
                out.write(format_srt(triples))
            elif getattr(args, "vtt", False):
                if total_files > 1 and not args.out:
                    out.write(f"== {path}\n")
                out.write(format_vtt(triples))
            elif getattr(args, "jsonl", False):
                out.write(format_jsonl(triples, path))
            else:
                if total_files > 1:
                    out.write(f"== {path}\n")
                show_ts = getattr(args, "timestamps", True)
                if show_ts:
                    if triples and len(triples[0]) == 4:
                        lines = [f"[{format_timestamp(s)} -> {format_timestamp(e)}] {src} / {tgt}" if tgt else f"[{format_timestamp(s)} -> {format_timestamp(e)}] {src}" for s, e, src, tgt in triples]
                    else:
                        lines = [f"[{format_timestamp(s)} -> {format_timestamp(e)}] {t}" for s, e, t in triples]
                    if lines:
                        out.write("\n".join(lines) + "\n")
                else:
                    # For bilingual, include both source and translation
                    if triples and len(triples[0]) == 4:
                        lines = [f"{src} / {tgt}" if tgt else src for _, _, src, tgt in triples]
                    else:
                        lines = [t for _, _, t in triples]
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
    
    # Handle --list-devices
    if getattr(args, "list_devices", False):
        print("[stt] Available audio input devices:")
        print()
        
        # Pulse/Termux (Linux/Android)
        if sys.platform != "win32":
            print("Linux/Android (PulseAudio/Termux):")
            pa = mic.which(cfg, "pulseaudio")
            parec = mic.which(cfg, "parec")
            if pa and parec:
                print(f"  PulseAudio: {pa}")
                print(f"  parec: {parec}")
            termux_rec = mic.which(cfg, "termux-microphone-record")
            if termux_rec:
                print(f"  Termux:API: {termux_rec}")
            print()
        
        # Windows
        if sys.platform == "win32" or True:  # Also show on other platforms for cross-compile
            print("Windows:")
            # DirectShow
            dshow_devs = mic.list_dshow_audio_devices()
            if dshow_devs:
                print("  DirectShow (ffmpeg):")
                for i, dev in enumerate(dshow_devs):
                    print(f"    [{i}] {dev}")
            else:
                print("  DirectShow: no devices found")
            
            # sounddevice
            try:
                import sounddevice as sd
                devs = sd.query_devices()
                in_devs = [(i, d) for i, d in enumerate(devs) if d.get('max_input_channels', 0) > 0]
                if in_devs:
                    print("  sounddevice (PortAudio/WASAPI):")
                    for i, dev in in_devs:
                        print(f"    [{i}] {dev.get('name')} (channels: {dev.get('max_input_channels')})")
                else:
                    print("  sounddevice: no input devices found")
            except ImportError:
                print("  sounddevice: not installed")
            except Exception as e:
                print(f"  sounddevice: error - {e}")
        return
    
    if getattr(args, "interactive", False):
        interactive_wizard(args, cfg, is_file=False)
    if _level(args, cfg) and not args.whisper and not args.distant and not getattr(args, "subnoise", False):
        print("[stt] note: -a/--aggressiveness applies to the termux backend and --whisper/--distant; the streaming pulse backend ignores it", file=sys.stderr)
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
        elif getattr(args, "timestamps", True):
            s_rel = max(0.0, start - t0)
            e_rel = max(0.0, end - t0)
            print(f"[{format_timestamp(s_rel)} -> {format_timestamp(e_rel)}] {text}", flush=True)
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


def cmd_config(args, cfg):
    action = args.config_action
    if action == "show":
        print(f"Config file: {config.CONFIG_PATH}")
        if config.CONFIG_PATH.exists():
            print()
            print(config.CONFIG_PATH.read_text())
        else:
            print("(not yet created; defaults shown below)")
            for k, v in sorted(config.DEFAULTS.items()):
                print(f"  {k} = {v!r}")
    elif action == "reset":
        if config.CONFIG_PATH.exists():
            config.CONFIG_PATH.unlink()
            print(f"[stt] Deleted {config.CONFIG_PATH}; defaults will be written on next run.")
        else:
            print(f"[stt] Config file not found: {config.CONFIG_PATH}")
    elif action == "set":
        if len(args.kv) < 2:
            sys.exit("usage: stt config set KEY VALUE")
        key = args.kv[0]
        raw = " ".join(args.kv[1:])
        if key not in config.DEFAULTS:
            known = ", ".join(sorted(config.DEFAULTS))
            sys.exit(f"[stt] unknown config key '{key}'. Known keys:\n  {known}")
        default_val = config.DEFAULTS[key]
        try:
            if isinstance(default_val, bool):
                value = raw.lower() in ("true", "1", "yes")
            elif isinstance(default_val, int):
                value = int(raw)
            elif isinstance(default_val, float):
                value = float(raw)
            else:
                value = raw
        except ValueError:
            sys.exit(f"[stt] invalid value for '{key}': {raw!r} (expected {type(default_val).__name__})")
        config.update_config({key: value})
        print(f"[stt] {key} = {value!r}  →  {config.CONFIG_PATH}")


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


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser for the STT CLI."""
    ap = argparse.ArgumentParser(
        prog="stt",
        description="Offline speech-to-text (sherpa-onnx, int8 quantized models, CPU-only)",
        formatter_class=ColorHelpFormatter,
        epilog="""EXAMPLES & USAGE:
  stt file audio.mp3
      Transcribe an audio file using default offline ASR.

  stt file recording.wav --dynamic
      Run multi-model dynamic ensemble for faint, distant, or noisy speech.

  stt file recording.wav --all
      Run through ALL installed models & DSP enhancements with color-coded comparison.

  stt file audio.mp3 --range 07s635z-10s --all
      Transcribe a millisecond-exact audio slice across all model matrix configurations.

  stt file video.mp4 --srt --sidecar
      Generate video.srt subtitles alongside the video file.

  stt mic --whisper
      Live mic transcription with VAD utterance cutting and offline Parakeet.

  stt ui
      Launch local web UI on http://127.0.0.1:8080 with drag-and-drop.

  stt doctor
      Verify model files, Python dependencies, ffmpeg, and mic audio backends.

  stt config show
      Display active configuration and model paths.
"""
    )
    ap.add_argument("--version", action="version", version=f"stt {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True, title="SUBCOMMANDS")

    # --- stt file ---
    f = sub.add_parser(
        "file",
        help="transcribe audio/video files",
        formatter_class=ColorHelpFormatter,
        description="Transcribe audio and video files using offline neural ASR and DSP speech enhancement.",
        epilog="""EXAMPLES & USAGE:
  stt file interview.mp3
      Standard transcription with automatic noise profiling.

  stt file audio.wav --dynamic
      4-model multi-stream ensemble (gtcrn, hybrid, dpdfnet4, hp) with CN-ROVER
      fusion and cohesive sentence reconstitution for low-energy speech.

  stt file audio.wav --all
      Run audio through ALL models (offline & streaming), enhancement filters
      (GTCRN, DPDFNet, hybrid crossover, formant EQ), and parameter permutations
      with color-coded side-by-side terminal output and summary matrix.

  stt file video.mp4 --srt --sidecar
      Create SubRip subtitle file (video.srt) directly beside input media.

  stt file -r ./recordings/ --sidecar --timestamps
      Recursively transcribe all audio/video files in a directory.

  stt file noisy.wav --subnoise --rumble-cutoff 140 --formant-boost 4.0 --blank-penalty 0.55
      Acoustic restoration pipeline targeting speech buried near or below noise floor.

  stt file audio_es.wav --translate --lang-source es --lang-target en
      Transcribe Spanish speech and translate directly into English (NLLB-200).

  stt file audio.mp3 --range 00h00m07s635z-00h00m10s000z --all
      Transcribe only the specified time slice (e.g. 7s-10s) through all model combinations.
"""
    )
    g_f_pos = f.add_argument_group("POSITIONAL ARGUMENTS")
    g_f_pos.add_argument("paths", nargs="+", metavar="PATH", help="audio/video file or directory paths")

    g_f_io = f.add_argument_group("INPUT & OUTPUT OPTIONS")
    g_f_io.add_argument("-r", "--recursive", action="store_true", help="recursively search directories for audio/video files")
    g_f_io.add_argument("--range", metavar="SPEC",
                        help="transcribe only a time range (e.g. 00h00m07s635z-00h00m10s000z, 7s-10s, 01:30-02:00, 7.5-12)")
    g_f_io.add_argument("--out", metavar="FILE", help="save transcript output to specified file")
    g_f_io.add_argument("--jsonl", action="store_true", help="output JSON lines with start/end/text")
    g_f_io.add_argument("--srt", action="store_true", help="output standard SubRip (.srt) subtitles")
    g_f_io.add_argument("--vtt", action="store_true", help="output WebVTT (.vtt) subtitles")
    g_f_io.add_argument("--sidecar", action="store_true",
                        help="save sidecar subtitle/transcript file next to each input file (e.g. video.mp4 -> video.srt)")
    g_f_io.add_argument("-t", "--timestamps", action=argparse.BooleanOptionalAction, default=True,
                        help="display timestamps for each transcribed speech segment (default: true; disable with --no-timestamps)")

    _add_shared_groups(f, is_file=True)

    g_f_diag = f.add_argument_group("FILE DIAGNOSTICS & LOGGING")
    g_f_diag.add_argument("-v", "--verbose", action="store_true", help="print timing / real-time factor (RTF) to stderr")

    f.set_defaults(fn=cmd_file)

    # --- stt mic ---
    m = sub.add_parser(
        "mic",
        help="live microphone transcription",
        formatter_class=ColorHelpFormatter,
        description="Capture live microphone audio and transcribe in real-time or segmented utterances.",
        epilog="""EXAMPLES & USAGE:
  stt mic
      Real-time streaming speech transcription with low latency (Nemotron / Kroko).

  stt mic --whisper
      High-accuracy utterance decoding on pauses (aggressiveness 3 + Parakeet).

  stt mic --distant
      Ultra-sensitive VAD for quiet or distant speech from across a room (level 5).

  stt mic --subnoise
      Sliding ring-buffer acoustic scan with multi-model enhancement.

  stt mic --backend windows --list-devices
      List available Windows audio capture devices.

  stt mic --partial
      Show interim transcription tokens as words are spoken.
"""
    )
    g_m_mode = m.add_argument_group("MICROPHONE CAPTURE MODES")
    mic_mode = g_m_mode.add_mutually_exclusive_group()
    mic_mode.add_argument("--whisper", action="store_true",
                          help="whispering / noisy room: cut utterances with VAD, denoise + normalize each, decode with offline model (implies -a 3)")
    mic_mode.add_argument("--distant", action="store_true",
                          help="very quiet / far-away speech: like --whisper with most sensitive VAD, higher gain (level 5)")
    mic_mode.add_argument("--subnoise", action="store_true",
                          help="continuous sliding ring-buffer scan with multi-model denoising for sub-noise speech")

    g_m_stream = m.add_argument_group("STREAMING & OUTPUT OPTIONS")
    g_m_stream.add_argument("--partial", action="store_true", help="show interim text as words are spoken (pulse backend)")
    g_m_stream.add_argument("--jsonl", action="store_true", help="output JSON lines with start/end/text")
    g_m_stream.add_argument("-t", "--timestamps", action=argparse.BooleanOptionalAction, default=True,
                            help="display timestamps for each transcribed speech segment (default: true; disable with --no-timestamps)")

    _add_shared_groups(m, is_file=False)

    g_m_hw = m.add_argument_group("MICROPHONE HARDWARE & DRIVER CONTROLS")
    g_m_hw.add_argument("--backend", choices=["pulse", "termux", "windows"], metavar="{pulse,termux,windows}",
                        help="override auto-detection of audio capture backend")
    g_m_hw.add_argument("--list-devices", action="store_true", help="list available audio input devices and exit")
    g_m_hw.add_argument("--no-calibrate", action="store_true", help="skip the 2 s room-noise calibration at start")
    g_m_hw.add_argument("--save", metavar="WAV", help="also write raw captured audio to WAV file (pulse backend) for tuning")
    g_m_hw.add_argument("--debug", action="store_true", help="print mic level / AGC gain / VAD state once a second (pulse backend)")

    m.set_defaults(fn=cmd_mic)

    # --- stt ui ---
    u = sub.add_parser(
        "ui",
        help="launch graphical web front-end with file picker and hotkeys",
        formatter_class=ColorHelpFormatter,
        description="Launch local browser-based GUI with file dropzone, live mic controls, and waveform viewer.",
        epilog="""EXAMPLES & USAGE:
  stt ui
      Launch web UI on http://127.0.0.1:8080 and open default web browser.

  stt ui --port 9000 --no-browser
      Listen on port 9000 without automatically opening browser.
"""
    )
    g_u = u.add_argument_group("SERVER OPTIONS")
    g_u.add_argument("--host", default="127.0.0.1", metavar="HOST", help="server listen address (default: 127.0.0.1)")
    g_u.add_argument("--port", type=int, default=8080, metavar="PORT", help="server listen port (default: 8080)")
    g_u.add_argument("--no-browser", action="store_true", help="do not auto-open browser on launch")
    u.set_defaults(fn=cmd_ui)

    # --- stt doctor ---
    d = sub.add_parser(
        "doctor",
        help="check dependencies, models, mic backends",
        formatter_class=ColorHelpFormatter,
        description="Inspect and diagnose Python packages, sherpa-onnx, neural models, ffmpeg, and audio capture backends.",
        epilog="""EXAMPLES & USAGE:
  stt doctor
      Run complete diagnostic report to verify system readiness.
"""
    )
    d.set_defaults(fn=cmd_doctor)

    # --- stt config ---
    cfg_p = sub.add_parser(
        "config",
        help="view or edit configuration",
        formatter_class=ColorHelpFormatter,
        description="Inspect, set, or reset persistent configuration settings.",
        epilog="""EXAMPLES & USAGE:
  stt config show
      Display configuration file path and all active values.

  stt config set threads 4
      Set default CPU inference thread count to 4.

  stt config set blank_penalty 0.65
      Set default transducer blank penalty.

  stt config reset
      Delete configuration file and restore factory defaults.
"""
    )
    cfg_sub = cfg_p.add_subparsers(dest="config_action", required=True, title="CONFIG ACTIONS")
    cfg_sub.add_parser("show", help="print config file path and current values", formatter_class=ColorHelpFormatter)
    cfg_sub.add_parser("reset", help="delete config file (defaults restored on next run)", formatter_class=ColorHelpFormatter)
    cfg_set = cfg_sub.add_parser("set", help="set a config key (e.g. stt config set threads 2)", formatter_class=ColorHelpFormatter)
    cfg_set.add_argument("kv", nargs="+", metavar="KEY VALUE", help="config key followed by desired value")
    cfg_p.set_defaults(fn=cmd_config)

    return ap


_build_parser = build_parser


def main(argv=None):
    cfg = config.load()
    ap = build_parser()
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


if __name__ == "__main__":
    main()
