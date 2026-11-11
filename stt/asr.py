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


def _has_joiner(d: Path) -> bool:
    return (d / "joiner.int8.onnx").exists() or (d / "joiner.onnx").exists()


def _pick_onnx(d: Path, stem: str) -> str:
    for suffix in (".int8.onnx", ".onnx"):
        if (d / (stem + suffix)).exists():
            return str(d / (stem + suffix))
    raise FileNotFoundError(f"{stem}.onnx not found in {d}")


def _files(d: Path) -> dict:
    return dict(
        encoder=_pick_onnx(d, "encoder"),
        decoder=_pick_onnx(d, "decoder"),
        joiner=_pick_onnx(d, "joiner"),
        tokens=str(d / "tokens.txt"),
    )


def _is_supported_asr_dir(d: Path) -> bool:
    if not d.is_dir() or not (d / "tokens.txt").exists():
        return False
    if _has_joiner(d):
        return True
    if "cohere" in d.name.lower() and hasattr(sherpa_onnx.OfflineRecognizer, "from_cohere_transcribe"):
        return (d / "encoder.int8.onnx").exists() or (d / "encoder.onnx").exists()
    return False


def _model_type(d: Path) -> str:
    """NeMo/Nemotron exports need model_type nemo_transducer; plain zipformer (e.g., kroko) use default.
    
    Detection is based on directory name convention from sherpa-onnx releases.
    NeMo models have 'nemo' in their release name (e.g., sherpa-onnx-nemo-parakeet-*, sherpa-onnx-nemotron-*).
    """
    return "nemo_transducer" if "nemo" in d.name.lower() else ""


_offline_cache: dict = {}
_offline_cache_lock = threading.Lock()
_MAX_OFFLINE_CACHE = 6


def offline(
    cfg: dict,
    model: str | None = None,
    threads: int | None = None,
    blank_penalty: float | None = None,
    decoding_method: str = "greedy_search",
    max_active_paths: int = 4,
):
    d = _resolve(cfg, model, "offline_model")
    if not _has_joiner(d) and "cohere" in d.name.lower() and hasattr(sherpa_onnx.OfflineRecognizer, "from_cohere_transcribe"):
        return sherpa_onnx.OfflineRecognizer.from_cohere_transcribe(
            encoder=_pick_onnx(d, "encoder"),
            decoder=_pick_onnx(d, "decoder"),
            tokens=str(d / "tokens.txt"),
            num_threads=threads or cfg["threads"],
        )
    kwargs = dict(
        num_threads=threads or cfg["threads"],
        model_type=_model_type(d),
        decoding_method=decoding_method,
    )
    if decoding_method == "modified_beam_search":
        kwargs["max_active_paths"] = max_active_paths
    if blank_penalty is not None:
        kwargs["blank_penalty"] = float(blank_penalty)
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        **_files(d), **kwargs
    )


_DEFAULT_OFFLINE_FN = offline


def get_offline(
    cfg: dict,
    model: str | None = None,
    threads: int | None = None,
    blank_penalty: float | None = None,
    decoding_method: str = "greedy_search",
    max_active_paths: int = 4,
):
    if offline is not _DEFAULT_OFFLINE_FN:
        return offline(
            cfg,
            model=model,
            threads=threads,
            blank_penalty=blank_penalty,
            decoding_method=decoding_method,
            max_active_paths=max_active_paths,
        )
    bp_key = round(float(blank_penalty), 2) if blank_penalty is not None else None
    key = (
        str(cfg.get("models_dir", "")),
        model or cfg.get("offline_model"),
        threads or cfg.get("threads", 4),
        bp_key,
        decoding_method,
        max_active_paths,
    )
    evicted = []
    with _offline_cache_lock:
        if key in _offline_cache:
            val = _offline_cache.pop(key)
            _offline_cache[key] = val
            return val
        while len(_offline_cache) >= _MAX_OFFLINE_CACHE:
            oldest_key = next(iter(_offline_cache))
            evicted.append(_offline_cache.pop(oldest_key))
        _offline_cache[key] = offline(
            cfg,
            model=model,
            threads=threads,
            blank_penalty=blank_penalty,
            decoding_method=decoding_method,
            max_active_paths=max_active_paths,
        )
        result = _offline_cache[key]
    # Release evicted recognizers and trigger GC outside the lock to avoid
    # potential deadlock with ONNX Runtime's internal thread pool finalizers.
    if evicted:
        del evicted
        import gc
        gc.collect()
    return result


def streaming(cfg: dict, model: str | None = None, threads: int | None = None):
    d = _resolve(cfg, model, "streaming_model")
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        **_files(d), num_threads=threads or cfg["threads"], model_type=_model_type(d)
    )


def transcribe(rec, chunks: list[np.ndarray]) -> list[str]:
    """Decode a list of 16 kHz float32 arrays using batch decoding (2 at a time) for throughput."""
    return [d["text"] for d in transcribe_batch_detailed(rec, chunks, max_batch_size=2)]


def transcribe_batch(rec, chunks: list[np.ndarray], max_batch_size: int = 2) -> list[str]:
    """Batch-decode a small list of candidate audio chunks in parallel using decode_streams."""
    detailed = transcribe_batch_detailed(rec, chunks, max_batch_size=max_batch_size)
    return [d["text"] for d in detailed]


def transcribe_batch_detailed(rec, chunks: list[np.ndarray], max_batch_size: int = 2) -> list[dict]:
    """Batch-decode candidate chunks in parallel and return rich acoustic/neural decoding metadata."""
    if not chunks:
        return []
    results = []
    for i in range(0, len(chunks), max_batch_size):
        batch = chunks[i : i + max_batch_size]
        streams = []
        for c in batch:
            s = rec.create_stream()
            s.accept_waveform(SR, c)
            streams.append(s)
        try:
            rec.decode_streams(streams)
        except Exception:
            for s in streams:
                try:
                    rec.decode_stream(s)
                except Exception:
                    pass
        for s in streams:
            res = s.result
            tokens = list(res.tokens) if hasattr(res, "tokens") else []
            log_probs = list(res.ys_log_probs) if hasattr(res, "ys_log_probs") else []
            token_confs = [float(np.clip(np.exp(lp), 0.0, 1.0)) for lp in log_probs]
            conf = float(np.mean(token_confs)) if token_confs else (0.75 if tokens else 0.0)
            durations = list(res.durations) if hasattr(res, "durations") else []
            zero_skips = sum(1 for d in durations if d == 0)
            zero_skip_ratio = float(zero_skips / max(len(durations), 1)) if durations else 0.0
            results.append({
                "text": res.text.strip(),
                "words": list(res.words) if hasattr(res, "words") and res.words else res.text.strip().split(),
                "tokens": tokens,
                "timestamps": list(res.timestamps) if hasattr(res, "timestamps") else [],
                "durations": durations,
                "log_probs": log_probs,
                "token_confs": token_confs,
                "confidence": conf,
                "zero_skip_ratio": zero_skip_ratio,
            })
    return results


# Multilingual support (delegates to unified _offline_cache to prevent duplicate 670 MB model loads)
_multilingual_offline_cache = _offline_cache
_multilingual_offline_cache_lock = _offline_cache_lock


def get_multilingual_offline(
    cfg: dict,
    model: str | None = None,
    threads: int | None = None,
    blank_penalty: float | None = None,
    decoding_method: str = "greedy_search",
    max_active_paths: int = 4,
):
    """Get or create a multilingual offline recognizer, sharing the unified recognizer cache."""
    name = model or cfg.get("multilingual_model")
    try:
        _resolve(cfg, name, "multilingual_model")
    except (FileNotFoundError, ValueError, KeyError):
        name = model or cfg.get("offline_model")
        _resolve(cfg, name, "offline_model")
    return get_offline(
        cfg,
        model=name,
        threads=threads,
        blank_penalty=blank_penalty,
        decoding_method=decoding_method,
        max_active_paths=max_active_paths,
    )


def detect_language_from_audio(rec, samples: np.ndarray, min_duration: float = 3.0) -> tuple[str, float]:
    """Detect spoken language from the highest speech-band energy spans of an audio signal."""
    from . import audio, translation
    
    if len(samples) == 0:
        return "en", 0.0
    
    win_len = max(int(max(min_duration, 2.5) * SR), SR)
    if len(samples) <= win_len:
        candidate_chunks = [samples]
    else:
        step = max(SR, win_len // 2)
        scored = []
        for start in range(0, len(samples) - win_len + 1, step):
            sl = samples[start : start + win_len]
            scored.append((audio.speech_band_energy(sl), start, sl))
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = []
        used_starts = []
        for _, st_idx, sl in scored:
            if all(abs(st_idx - u) >= win_len // 2 for u in used_starts):
                chosen.append(sl)
                used_starts.append(st_idx)
            if len(chosen) >= 3:
                break
        candidate_chunks = chosen or [samples[:win_len]]
    
    collected = []
    for ch in candidate_chunks:
        txts = transcribe(rec, [ch])
        if txts and txts[0].strip():
            collected.append(txts[0].strip())
        if sum(len(t.split()) for t in collected) >= 6:
            break
    
    full_text = " ".join(collected).strip()
    if full_text:
        return translation.detect_language_simple(full_text)
    return "en", 0.0


def transcribe_multilingual(rec, chunks: list[np.ndarray]) -> list[tuple[str, str]]:
    """Transcribe chunks once and detect language directly from each decoded transcript."""
    from . import translation

    results = []
    for c in chunks:
        s = rec.create_stream()
        s.accept_waveform(SR, c)
        rec.decode_stream(s)
        text = s.result.text.strip()
        lang, _ = translation.detect_language_simple(text) if text else ("en", 0.0)
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
        if p.is_dir() and _is_supported_asr_dir(p):
            kind = "asr-streaming" if "streaming" in p.name else "asr-offline"
            active = p.name in (cfg.get("streaming_model") if kind == "asr-streaming" else cfg.get("offline_model"),)
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        elif p.is_dir() and (p / "model.bin").exists():
            kind = "translation"
            active = p.name == cfg.get("translation_model")
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        elif p.suffix == ".onnx" and p.name.startswith(DENOISER_PREFIXES):
            kind, active, size = "denoiser", p.name == cfg.get("robust_denoiser"), p.stat().st_size
        elif p.name.startswith("silero_vad"):
            kind, active, size = "vad", p.name == cfg.get("vad_model"), p.stat().st_size
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
        if p.is_dir() and _is_supported_asr_dir(p):  # explicit path to an ASR model directory
            kind = "asr-streaming" if "streaming" in p.name else "asr-offline"
            cfg["streaming_model" if kind == "asr-streaming" else "offline_model"] = str(p)
            continue
        hits = [r for r in rows if r["name"] == n or r["name"] == n + ".onnx"] or [r for r in rows if n in r["name"]]
        if len(hits) != 1:
            avail = ", ".join(r["name"] for r in rows)
            raise SystemExit(f"--model {n}: {'no such model' if not hits else 'ambiguous'}; installed: {avail}\n(try --model list)")
        r = hits[0]
        key = {
            "asr-streaming": "streaming_model",
            "asr-offline": "offline_model",
            "denoiser": "robust_denoiser",
            "vad": "vad_model",
            "translation": "translation_model",
        }[r["kind"]]
        cfg[key] = r["name"]
    return cfg
