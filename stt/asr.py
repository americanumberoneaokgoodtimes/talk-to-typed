"""Recognizer construction. Model dirs hold encoder/decoder/joiner .onnx + tokens.txt."""
import threading
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
    """NeMo/Nemotron exports need model_type nemo_transducer; plain zipformer (e.g., kroko) use default.
    
    Detection is based on directory name convention from sherpa-onnx releases.
    NeMo models have 'nemo' in their release name (e.g., sherpa-onnx-nemo-parakeet-*, sherpa-onnx-nemotron-*).
    """
    return "nemo_transducer" if "nemo" in d.name.lower() else ""


_offline_cache: dict = {}
_offline_cache_lock = threading.Lock()


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
    with _offline_cache_lock:
        if key not in _offline_cache:
            _offline_cache[key] = offline(cfg, model=model, threads=threads, blank_penalty=blank_penalty)
        return _offline_cache[key]


def streaming(cfg: dict, model: str | None = None, threads: int | None = None):
    d = _resolve(cfg, model, "streaming_model")
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        **_files(d), num_threads=threads or cfg["threads"], model_type=_model_type(d)
    )


def transcribe(rec, chunks: list[np.ndarray]) -> list[str]:
    """Decode a list of 16 kHz float32 arrays with an offline recognizer (process each chunk individually)."""
    texts = []
    for c in chunks:
        s = rec.create_stream()
        s.accept_waveform(SR, c)
        rec.decode_stream(s)  # Process each stream individually to avoid OOM
        texts.append(s.result.text.strip())
    return texts


def transcribe_batch(rec, chunks: list[np.ndarray], max_batch_size: int = 4) -> list[str]:
    """Batch-decode a small list of candidate audio chunks in parallel using decode_streams."""
    if not chunks:
        return []
    texts = []
    for i in range(0, len(chunks), max_batch_size):
        batch = chunks[i : i + max_batch_size]
        streams = []
        for c in batch:
            s = rec.create_stream()
            s.accept_waveform(SR, c)
            streams.append(s)
        try:
            rec.decode_streams(streams)
            for s in streams:
                texts.append(s.result.text.strip())
        except Exception:
            # Fallback to single decode if batch decode is unsupported or fails
            for s in streams:
                try:
                    rec.decode_stream(s)
                    texts.append(s.result.text.strip())
                except Exception:
                    texts.append("")
    return texts


# Multilingual support
_multilingual_offline_cache: dict = {}
_multilingual_offline_cache_lock = threading.Lock()


def get_multilingual_offline(cfg: dict, model: str | None = None, threads: int | None = None, blank_penalty: float | None = None):
    """Get or create a multilingual offline recognizer.
    
    Multilingual models (like Parakeet TDT 1.1B or Parakeet TDT 0.6B) support multiple languages
    and can auto-detect the spoken language.
    """
    name = model or cfg.get("multilingual_model")
    try:
        d = _resolve(cfg, name, "multilingual_model")
    except (FileNotFoundError, ValueError):
        name = model or cfg.get("offline_model")
        d = _resolve(cfg, name, "offline_model")
    
    bp_key = round(float(blank_penalty), 2) if blank_penalty is not None else None
    key = (name, threads or cfg.get("threads", 4), bp_key)
    
    with _multilingual_offline_cache_lock:
        if key not in _multilingual_offline_cache:
            kwargs = dict(num_threads=threads or cfg["threads"], model_type=_model_type(d))
            if blank_penalty is not None:
                kwargs["blank_penalty"] = float(blank_penalty)
            _multilingual_offline_cache[key] = sherpa_onnx.OfflineRecognizer.from_transducer(
                **_files(d), **kwargs
            )
        return _multilingual_offline_cache[key]


def detect_language_from_audio(rec, samples: np.ndarray, min_duration: float = 2.0) -> tuple[str, float]:
    """Detect language from audio using the multilingual model.
    
    Returns (language_code, confidence)
    
    Note: This is a simple heuristic - runs recognition on first min_duration
    seconds and uses text-based language detection.
    """
    from . import translation
    
    if len(samples) == 0:
        return "en", 0.0
    
    # Use first min_duration seconds for detection
    detect_samples = int(min_duration * SR)
    chunk = samples[:min(detect_samples, len(samples))]
    
    # Transcribe
    texts = transcribe(rec, [chunk])
    text = texts[0] if texts else ""
    
    # Use text-based detection
    if text:
        return translation.detect_language_simple(text)
    return "en", 0.0


def transcribe_multilingual(rec, chunks: list[np.ndarray]) -> list[tuple[str, str]]:
    """Transcribe with language detection per chunk.
    
    Returns list of (text, detected_language) tuples.
    """
    results = []
    for c in chunks:
        s = rec.create_stream()
        s.accept_waveform(SR, c)
        rec.decode_stream(s)
        text = s.result.text.strip()
        
        # Detect language from the transcribed text
        lang, conf = detect_language_from_audio(rec, c, min_duration=0.5)
        results.append((text, lang))
    
    return results


DENOISER_PREFIXES = ("gtcrn", "dpdfnet")


def catalog(cfg: dict) -> list[dict]:
    """Installed models under models_dir: [{name, kind, size_mb, active}]."""
    out = []
    md = Path(cfg["models_dir"])
    if not md.exists():
        return out
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
    if not rows:
        return f"(no models found in {cfg['models_dir']} — run scripts/install_models.sh or .ps1)"
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
