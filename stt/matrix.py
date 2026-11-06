"""Multi-model and multi-enhancement matrix transcription (--all).

Runs input audio through all installed offline and streaming models, speech enhancement models,
formant equalizers, and decoding configurations with high-contrast color-coded output.
"""
import sys
import os
import time
from pathlib import Path
import numpy as np

from . import audio, asr, robust, config

# High-contrast 16/256-color palette for distinct visual contrast
PALETTE = [
    ("\033[1;36m", "Cyan"),
    ("\033[1;32m", "Green"),
    ("\033[1;33m", "Yellow"),
    ("\033[1;35m", "Magenta"),
    ("\033[38;5;209m", "Coral"),
    ("\033[38;5;75m", "Sky Blue"),
    ("\033[38;5;120m", "Mint"),
    ("\033[38;5;177m", "Orchid"),
    ("\033[38;5;214m", "Orange"),
    ("\033[38;5;43m", "Teal"),
    ("\033[1;34m", "Blue"),
    ("\033[38;5;154m", "Lime"),
    ("\033[38;5;213m", "Pink"),
    ("\033[38;5;222m", "Gold"),
    ("\033[1;37m", "White"),
]
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def _color_enabled() -> bool:
    """Check if color formatting should be rendered based on environment and tty."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def format_ts(sec: float) -> str:
    """Format seconds into MM:SS.mmm timeline timestamp."""
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:06.3f}"


def get_available_combinations(cfg: dict) -> list[dict]:
    """Inspect installed models in models_dir and build comprehensive matrix configurations."""
    models_dir = Path(cfg["models_dir"])
    combos = []

    # 1. NeMo Parakeet TDT Offline Transducer (Primary Offline ASR)
    offline_model = cfg.get("offline_model", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8")
    if (models_dir / offline_model).is_dir():
        combos.extend([
            {
                "id": "parakeet_gtcrn_beam",
                "name": "Parakeet-TDT | GTCRN | Beam-16 (BP=0.55)",
                "category": "Offline Beam Search",
                "model": offline_model,
                "sig": "gtcrn",
                "bp": 0.55,
                "beam": True,
                "type": "offline",
            },
            {
                "id": "parakeet_hybrid_beam",
                "name": "Parakeet-TDT | Hybrid Crossover | Beam-16 (BP=0.55)",
                "category": "Offline Beam Search",
                "model": offline_model,
                "sig": "hybrid",
                "bp": 0.55,
                "beam": True,
                "type": "offline",
            },
            {
                "id": "parakeet_clean_beam",
                "name": "Parakeet-TDT | Clean HP+Notch | Beam-16 (BP=0.40)",
                "category": "Offline Beam Search",
                "model": offline_model,
                "sig": "clean_hp",
                "bp": 0.40,
                "beam": True,
                "type": "offline",
            },
            {
                "id": "parakeet_gtcrn_eq_beam",
                "name": "Parakeet-TDT | GTCRN + Formant EQ | Beam-16 (BP=0.85)",
                "category": "Offline Beam Search",
                "model": offline_model,
                "sig": "gtcrn_eq",
                "bp": 0.85,
                "beam": True,
                "type": "offline",
            },
            {
                "id": "parakeet_formant_eq_beam",
                "name": "Parakeet-TDT | Formant EQ (+4dB) | Beam-16 (BP=0.70)",
                "category": "Offline Beam Search",
                "model": offline_model,
                "sig": "formant_eq",
                "bp": 0.70,
                "beam": True,
                "type": "offline",
            },
            {
                "id": "parakeet_dp4_beam",
                "name": "Parakeet-TDT | DPDFNet-4 | Beam-16 (BP=0.50)",
                "category": "Offline Beam Search",
                "model": offline_model,
                "sig": "dp4",
                "bp": 0.50,
                "beam": True,
                "type": "offline",
            },
            {
                "id": "parakeet_dp8_beam",
                "name": "Parakeet-TDT | DPDFNet-8 | Beam-16 (BP=0.50)",
                "category": "Offline Beam Search",
                "model": offline_model,
                "sig": "dp8",
                "bp": 0.50,
                "beam": True,
                "type": "offline",
            },
            {
                "id": "parakeet_gtcrn_greedy",
                "name": "Parakeet-TDT | GTCRN | Greedy (BP=0.00)",
                "category": "Offline Greedy",
                "model": offline_model,
                "sig": "gtcrn",
                "bp": 0.0,
                "beam": False,
                "type": "offline",
            },
            {
                "id": "parakeet_raw_control",
                "name": "Parakeet-TDT | Raw Baseline | Greedy (BP=0.00)",
                "category": "Control Baseline",
                "model": offline_model,
                "sig": "raw",
                "bp": 0.0,
                "beam": False,
                "type": "offline",
            },
            {
                "id": "dynamic_ensemble",
                "name": "Multi-Model Dynamic Ensemble | CN-ROVER Fusion",
                "category": "Ensemble Fusion",
                "type": "dynamic",
            },
        ])

    # 2. Nemotron Streaming Model
    nemotron_name = "sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25"
    if (models_dir / nemotron_name).is_dir():
        combos.extend([
            {
                "id": "nemotron_gtcrn",
                "name": "Nemotron-Stream 0.6B | GTCRN",
                "category": "Streaming Transducer",
                "model": nemotron_name,
                "sig": "gtcrn",
                "type": "streaming",
            },
            {
                "id": "nemotron_hybrid",
                "name": "Nemotron-Stream 0.6B | Hybrid Crossover",
                "category": "Streaming Transducer",
                "model": nemotron_name,
                "sig": "hybrid",
                "type": "streaming",
            },
            {
                "id": "nemotron_clean",
                "name": "Nemotron-Stream 0.6B | Clean HP+Notch",
                "category": "Streaming Transducer",
                "model": nemotron_name,
                "sig": "clean_hp",
                "type": "streaming",
            },
        ])

    # 3. Kroko Zipformer Streaming Model
    kroko_name = "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"
    if (models_dir / kroko_name).is_dir():
        combos.extend([
            {
                "id": "kroko_gtcrn",
                "name": "Kroko-Zipformer | GTCRN",
                "category": "Streaming Zipformer",
                "model": kroko_name,
                "sig": "gtcrn",
                "type": "streaming",
            },
            {
                "id": "kroko_clean",
                "name": "Kroko-Zipformer | Clean HP+Notch",
                "category": "Streaming Zipformer",
                "model": kroko_name,
                "sig": "clean_hp",
                "type": "streaming",
            },
        ])

    return combos


def run_matrix(
    samples: np.ndarray,
    cfg: dict,
    audio_path: str = "",
    threads: int | None = None,
    progress_cb=None,
) -> list[dict]:
    """Execute matrix transcription across all combinations and output color-coded streams."""
    sr = audio.SR
    dur_sec = len(samples) / sr
    use_color = _color_enabled()

    # 1. Physical Acoustic Profiling & DSP Pre-filters
    prof = audio.profile_audio(samples)
    clean_hp = audio.butterworth_highpass(samples, cutoff_hz=prof.hpf_cutoff_hz)
    if prof.notch_freqs:
        clean_hp = audio.harmonic_notch_filter(clean_hp, freqs=prof.notch_freqs)

    # 2. Pre-compute enhancement layers once
    gtcrn = None
    try:
        gtcrn = audio.denoise(clean_hp, dict(cfg, robust_denoiser="gtcrn_simple.onnx"))
    except Exception as e:
        print(f"[stt matrix] Note: GTCRN unavailable ({e}), using HP fallback", file=sys.stderr)
        gtcrn = clean_hp

    hybrid = audio.spectral_crossover_blend(gtcrn, clean_hp)

    dp4 = None
    try:
        dp4 = audio.denoise(clean_hp, dict(cfg, robust_denoiser="dpdfnet4.onnx"))
    except Exception:
        dp4 = gtcrn

    dp8 = None
    try:
        dp8 = audio.denoise(clean_hp, dict(cfg, robust_denoiser="dpdfnet8.onnx"))
    except Exception:
        dp8 = clean_hp

    formant_eq = audio.speech_formant_filter(clean_hp, boost_db=4.0)
    gtcrn_eq = audio.speech_formant_filter(gtcrn, boost_db=4.0)

    enhancements = {
        "raw": samples,
        "clean_hp": clean_hp,
        "gtcrn": gtcrn,
        "hybrid": hybrid,
        "dp4": dp4,
        "dp8": dp8,
        "formant_eq": formant_eq,
        "gtcrn_eq": gtcrn_eq,
    }

    combos = get_available_combinations(cfg)
    total_combos = len(combos)

    print("\n" + "=" * 88, file=sys.stderr)
    title = f"=== MULTI-MODEL MATRIX TRANSCRIPTION (--all) [{total_combos} COMBINATIONS] ==="
    if use_color:
        print(f"\033[1;37m{title}\033[0m", file=sys.stderr)
    else:
        print(title, file=sys.stderr)
    print(f"File: {audio_path} ({dur_sec:.1f}s) | Rumble: {prof.rumble_fraction*100:.1f}% | Hum: {prof.hum_freq or 'None'}", file=sys.stderr)
    print("=" * 88 + "\n", file=sys.stderr)

    vad_cfg = dict(cfg, vad_threshold=0.35, vad_min_speech=0.10, vad_max_speech=10.0, vad_min_silence=0.25)
    matrix_results = []

    # Cache streaming models to avoid reloading
    streaming_cache = {}

    for idx, combo in enumerate(combos):
        color_code, color_name = PALETTE[idx % len(PALETTE)]
        c_pfx = color_code if use_color else ""
        c_sfx = RESET if use_color else ""

        mode_hdr = f"[MODE {idx+1:2d}/{total_combos:2d}] {combo['name']}"
        print(f"{c_pfx}{BOLD}{mode_hdr}{c_sfx}", flush=True)

        t0 = time.time()
        results = []

        if combo["type"] == "dynamic":
            # Run the multi-model dynamic ensemble with CN-ROVER fusion
            try:
                auto_params = robust.auto_tune_settings(dur_sec, samples=samples)
                triples = robust.subnoise_transcribe(
                    None, samples, cfg, params=auto_params, threads=threads
                )
                for s, e, t in triples:
                    results.append({"start": s, "end": e, "text": t})
            except Exception as e:
                print(f"{c_pfx}  [Ensemble error: {e}]{c_sfx}", flush=True)

        else:
            sig = enhancements.get(combo.get("sig", "clean_hp"), clean_hp)
            norm_sig = audio.normalize(sig)
            segs = list(audio.segments(norm_sig, vad_cfg))

            # If VAD finds segments, use them; if noisy audio blinds VAD, use 5s sliding windows
            chunks_with_time = []
            if segs:
                for s, e, c in segs:
                    chunks_with_time.append((s, e, c))
            else:
                win_samples = int(5.0 * sr)
                step_samples = int(2.5 * sr)
                for i in range(0, len(sig) - win_samples + 1, step_samples):
                    chunks_with_time.append((i / sr, (i + win_samples) / sr, sig[i : i + win_samples]))

            if combo["type"] == "offline":
                method = "modified_beam_search" if combo.get("beam") else "greedy_search"
                rec = asr.get_offline(
                    cfg,
                    model=combo["model"],
                    threads=threads or 2,
                    blank_penalty=combo.get("bp"),
                    decoding_method=method,
                    max_active_paths=16,
                )
                for s, e, c in chunks_with_time:
                    c_norm = audio.normalize(c)
                    st = rec.create_stream()
                    st.accept_waveform(sr, c_norm)
                    rec.decode_stream(st)
                    t = st.result.text.strip()
                    if t:
                        results.append({"start": s, "end": e, "text": t})

            elif combo["type"] == "streaming":
                import sherpa_onnx
                m_name = combo["model"]
                if m_name not in streaming_cache:
                    m_dir = Path(cfg["models_dir"]) / m_name
                    files = asr._files(m_dir)
                    streaming_cache[m_name] = sherpa_onnx.OnlineRecognizer.from_transducer(
                        **files,
                        num_threads=threads or 2,
                        model_type=asr._model_type(m_dir),
                    )
                rec = streaming_cache[m_name]

                for s, e, c in chunks_with_time:
                    c_norm = audio.normalize(c)
                    st = rec.create_stream()
                    st.accept_waveform(sr, c_norm)
                    st.input_finished()
                    while rec.is_ready(st):
                        rec.decode_stream(st)
                    t = rec.get_result(st).strip()
                    if t:
                        results.append({"start": s, "end": e, "text": t})

        elapsed = time.time() - t0

        # Print color-coded results
        if results:
            for r in results:
                s_str = format_ts(r["start"])
                e_str = format_ts(r["end"])
                print(f"{c_pfx}  [{s_str} -> {e_str}] {r['text']}{c_sfx}", flush=True)
        else:
            print(f"{c_pfx}{DIM}  (no speech detected){c_sfx}", flush=True)

        print(flush=True)

        matrix_results.append({
            "index": idx + 1,
            "id": combo["id"],
            "name": combo["name"],
            "category": combo.get("category", combo["type"]),
            "color_name": color_name,
            "color_code": color_code,
            "elapsed": elapsed,
            "results": results,
            "num_segments": len(results),
            "total_words": sum(len(r["text"].split()) for r in results),
        })

    # 3. Print Final Comparative Synthesis Summary Grid
    print("=" * 88, file=sys.stderr)
    summary_hdr = "=== MATRIX COMPARATIVE SUMMARY TABLE ==="
    if use_color:
        print(f"\033[1;37m{summary_hdr}\033[0m", file=sys.stderr)
    else:
        print(summary_hdr, file=sys.stderr)
    print("=" * 88, file=sys.stderr)
    print(f"{'Mode':<7} | {'Model & DSP Configuration':<46} | {'Segs':<5} | {'Words':<5} | {'Representative Output':<22}", file=sys.stderr)
    print("-" * 88, file=sys.stderr)

    for item in matrix_results:
        c_pfx = item["color_code"] if use_color else ""
        c_sfx = RESET if use_color else ""
        rep_text = item["results"][0]["text"] if item["results"] else "(none)"
        if len(rep_text) > 22:
            rep_text = rep_text[:19] + "..."
        mode_str = f"#{item['index']:02d}"
        cfg_str = item["name"]
        if len(cfg_str) > 46:
            cfg_str = cfg_str[:43] + "..."
        print(
            f"{c_pfx}{mode_str:<7} | {cfg_str:<46} | {item['num_segments']:<5} | {item['total_words']:<5} | {rep_text:<22}{c_sfx}",
            file=sys.stderr,
        )

    print("=" * 88 + "\n", file=sys.stderr)
    return matrix_results
