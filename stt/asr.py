"""Recognizer construction. Model dirs hold encoder/decoder/joiner .onnx + tokens.txt."""
from pathlib import Path

import numpy as np
import sherpa_onnx

from .audio import SR


def _resolve(cfg: dict, model: str | None, key: str) -> Path:
    name = model or cfg[key]
    p = Path(name)
    if not p.is_dir():
        p = Path(cfg["models_dir"]) / name
    if not p.is_dir():
        raise FileNotFoundError(f"model directory not found: {p}")
    return p


def _files(d: Path) -> dict:
    def pick(stem):
        for suffix in (".int8.onnx", ".onnx"):
            if (d / (stem + suffix)).exists():
                return str(d / (stem + suffix))
        raise FileNotFoundError(f"{stem}.onnx not found in {d}")

    return dict(encoder=pick("encoder"), decoder=pick("decoder"), joiner=pick("joiner"), tokens=str(d / "tokens.txt"))


def _model_type(d: Path) -> str:
    """NeMo/Nemotron exports need model_type nemo_transducer; plain zipformer transducers (e.g. kroko) use the default."""
    return "nemo_transducer" if "nemo" in d.name else ""


_offline_cache: dict = {}


def offline(cfg: dict, model: str | None = None, threads: int | None = None, blank_penalty: float | None = None):
    d = _resolve(cfg, model, "offline_model")
    kwargs = dict(num_threads=threads or cfg["threads"], model_type=_model_type(d))
    if blank_penalty is not None:
        kwargs["blank_penalty"] = float(blank_penalty)
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        **_files(d), **kwargs
    )


def get_offline(cfg: dict, model: str | None = None, threads: int | None = None, blank_penalty: float | None = None):
    bp_key = round(float(blank_penalty), 2) if blank_penalty is not None else None
    key = (model or cfg.get("offline_model"), threads or cfg.get("threads", 4), bp_key)
    if key not in _offline_cache:
        _offline_cache[key] = offline(cfg, model=model, threads=threads, blank_penalty=blank_penalty)
    return _offline_cache[key]


def streaming(cfg: dict, model: str | None = None, threads: int | None = None):
    d = _resolve(cfg, model, "streaming_model")
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        **_files(d), num_threads=threads or cfg["threads"], model_type=_model_type(d)
    )


def transcribe(rec, chunks: list[np.ndarray]) -> list[str]:
    """Batch-decode a list of 16 kHz float32 arrays with an offline recognizer."""
    streams = []
    for c in chunks:
        s = rec.create_stream()
        s.accept_waveform(SR, c)
        streams.append(s)
    if streams:
        rec.decode_streams(streams)
    return [s.result.text.strip() for s in streams]


DENOISER_PREFIXES = ("gtcrn", "dpdfnet")


def catalog(cfg: dict) -> list[dict]:
    """Installed models under models_dir: [{name, kind, size_mb, active}]."""
    out = []
    md = Path(cfg["models_dir"])
    for p in sorted(md.iterdir()):
        if p.is_dir() and (p / "tokens.txt").exists():
            kind = "asr-streaming" if "streaming" in p.name else "asr-offline"
            active = p.name in (cfg["streaming_model"] if kind == "asr-streaming" else cfg["offline_model"],)
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        elif p.suffix == ".onnx" and p.name.startswith(DENOISER_PREFIXES):
            kind, active, size = "denoiser", p.name == cfg["robust_denoiser"], p.stat().st_size
        elif p.name.startswith("silero_vad"):
            kind, active, size = "vad", p.name == cfg["vad_model"], p.stat().st_size
        else:
            continue
        out.append(dict(name=p.name, kind=kind, size_mb=size / 1e6, active=active))
    return out


def format_catalog(cfg: dict) -> str:
    rows = catalog(cfg)
    w = max(len(r["name"]) for r in rows)
    lines = [f"{'':2}{'NAME':<{w}}  {'KIND':<14}SIZE"]
    for r in sorted(rows, key=lambda r: (r["kind"], r["name"])):
        lines.append(f"{'*' if r['active'] else ' ':2}{r['name']:<{w}}  {r['kind']:<14}{r['size_mb']:.0f} MB")
    lines.append("\n* = current default. Use --model NAME (repeatable, e.g. --model dpdfnet4 --model <asr-name>).")
    lines.append("Any unambiguous part of a name works (e.g. --model parakeet, --model gtcrn).")
    return "\n".join(lines)


def apply_model_args(cfg: dict, names: list[str]) -> dict:
    """Resolve each --model value by kind and return a config with the matching defaults overridden."""
    cfg = dict(cfg)
    rows = catalog(cfg)
    for n in names:
        p = Path(n)
        if p.is_dir() and (p / "tokens.txt").exists():  # explicit path to an ASR model directory
            kind = "asr-streaming" if "streaming" in p.name else "asr-offline"
            cfg["streaming_model" if kind == "asr-streaming" else "offline_model"] = str(p)
            continue
        hits = [r for r in rows if r["name"] == n or r["name"] == n + ".onnx"] or [r for r in rows if n in r["name"]]
        if len(hits) != 1:
            avail = ", ".join(r["name"] for r in rows)
            raise SystemExit(f"--model {n}: {'no such model' if not hits else 'ambiguous'}; installed: {avail}\n(try --model list)")
        r = hits[0]
        key = {"asr-streaming": "streaming_model", "asr-offline": "offline_model", "denoiser": "robust_denoiser", "vad": "vad_model"}[r["kind"]]
        cfg[key] = r["name"]
    return cfg
