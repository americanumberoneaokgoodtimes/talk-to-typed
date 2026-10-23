import argparse
import json
import shutil
import signal
import sys
import time
from pathlib import Path

from . import __version__, config, robust


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
    p.add_argument("--lang", help="language hint; ignored by models that auto-detect or are English-only")


def _lang_note(args):
    if args.lang:
        print(f"note: --lang {args.lang} ignored (selected models auto-detect / are English-only)", file=sys.stderr)


def cmd_file(args, cfg):
    from . import asr, audio

    _lang_note(args)
    level = _level(args, cfg)
    rec = asr.offline(cfg, args.model, args.threads)
    out = open(args.out, "w") if args.out else sys.stdout
    total_audio = total_proc = 0.0
    try:
        for path in args.paths:
            try:
                samples = audio.decode(path)
            except RuntimeError as e:
                print(f"error: {e}", file=sys.stderr)
                continue
            t = time.time()
            triples = robust.transcribe(rec, samples, cfg, level)
            dt = time.time() - t
            dur = len(samples) / audio.SR
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
                if len(args.paths) > 1:
                    out.write(f"== {path}\n")
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
    if _level(args, cfg):
        print("[stt] aggressiveness applies to the termux backend; the streaming pulse backend ignores it", file=sys.stderr)
    backend = args.backend or cfg["mic_backend"]
    avail = mic.detect(cfg)
    if backend == "auto":
        order = [b for b in ("pulse", "termux") if avail[b][0]]
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
    runners = {"pulse": mic.run_pulse_utterance if (args.whisper or args.distant) else mic.run_pulse, "termux": mic.run_termux}
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
    sys.exit(0 if all(s != "FAIL" for _, s, _ in rows) and any_mic else 1)


def main(argv=None):
    cfg = config.load()
    ap = argparse.ArgumentParser(prog="stt", description="Offline speech-to-text (sherpa-onnx, int8, CPU)")
    ap.add_argument("--version", action="version", version=f"stt {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("file", help="transcribe audio/video files")
    f.add_argument("paths", nargs="+")
    f.add_argument("--jsonl", action="store_true", help="JSON lines with start/end/text")
    f.add_argument("--out", help="write output to this file")
    f.add_argument("-v", "--verbose", action="store_true", help="print timing / real-time factor to stderr")
    _common(f)
    f.set_defaults(fn=cmd_file)

    m = sub.add_parser("mic", help="live microphone transcription")
    m.add_argument("--partial", action="store_true", help="show interim text (pulse backend)")
    m.add_argument("--backend", choices=["pulse", "termux"], help="override auto-detection")
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
