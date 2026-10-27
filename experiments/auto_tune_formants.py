#!/usr/bin/env python3
"""Autonomous parameter optimizer for sub-noise speech and formant extraction.

Auto-tweaks:
  - Acoustic rumble cutoffs (100Hz, 140Hz, 180Hz)
  - Speech formant band equalization (300-3200 Hz boost: 0dB to 6dB)
  - Denoising models (GTCRN, DPDFNet-4, DPDFNet-2, and raw filtered) + blend ratios
  - Transducer blank penalty (dynamically adjusting to avoid repetitive degeneration)

Scores hypotheses with a Linguistic Coherence & Lexical Diversity fitness function:
  - Heavily penalizes repetitive word loops (e.g. "no no no" or "thank you thank you")
  - Rewards diverse, natural vocabulary (Type-Token Ratio)
  - Filters out unplausibly fast speech hallucination (> 5 words/sec)

Usage:
  python experiments/auto_tune_formants.py [WAV_FILE] [--start SEC] [--dur SEC] [--step SEC]
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf
import subprocess

import stt.config as config
import stt.audio as audio

SR = 16000


def get_audio(path: str, start_sec: float | None = None, dur_sec: float | None = None) -> np.ndarray:
    """Decode any audio path via ffmpeg mono 16kHz float32."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start_sec is not None:
        cmd.extend(["-ss", str(start_sec)])
    if dur_sec is not None:
        cmd.extend(["-t", str(dur_sec)])
    cmd.extend(["-i", path, "-vn", "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"])
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr.decode(errors='replace')}")
    return np.frombuffer(p.stdout, dtype=np.float32)


def apply_formant_filter(x: np.ndarray, low_hz=140.0, high_hz=5500.0, boost_db=3.5) -> np.ndarray:
    """FFT bandpass + gentle speech formant boost (300 - 3200 Hz)."""
    if len(x) == 0:
        return x
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / SR)

    # Rejection bands
    fft[freqs < low_hz] = 0.0
    fft[freqs > high_hz] = 0.0

    # Formant band boost
    if boost_db > 0.0:
        mask = (freqs >= 300.0) & (freqs <= 3200.0)
        fft[mask] *= 10.0 ** (boost_db / 20.0)

    return np.fft.irfft(fft, n=len(x)).astype(np.float32)


def score_coherence(text: str, dur_sec: float) -> tuple[float, str]:
    """Score transcript based on lexical diversity, grammar plausibility, and non-repetition."""
    words = [w.strip(".,?!;:\"'()").lower() for w in text.split() if w.strip(".,?!;:\"'()")]
    if not words:
        return 0.0, "empty"

    # Plausibility check: speech rate (words per second)
    wps = len(words) / max(dur_sec, 0.5)
    if wps > 5.0:
        return -5.0, f"too fast / hallucination ({wps:.1f} wps)"

    # Repetition check (unigram dominance)
    counts = Counter(words)
    most_common_word, most_common_cnt = counts.most_common(1)[0]
    dom_ratio = most_common_cnt / len(words)

    # Degenerate loop detection
    if len(words) >= 3 and dom_ratio >= 0.5:
        return 0.05, f"repetitive token '{most_common_word}' ({dom_ratio*100:.0f}%)"

    # Bigram repetition check (e.g. "thank you thank you thank you")
    if len(words) >= 4:
        bigrams = list(zip(words[:-1], words[1:]))
        bi_counts = Counter(bigrams)
        most_bi, bi_cnt = bi_counts.most_common(1)[0]
        if (bi_cnt / len(bigrams)) >= 0.4:
            return 0.05, f"repetitive phrase '{' '.join(most_bi)}'"

    # Trigram repetition check
    if len(words) >= 6:
        trigrams = list(zip(words[:-2], words[1:-1], words[2:]))
        tri_counts = Counter(trigrams)
        most_tri, tri_cnt = tri_counts.most_common(1)[0]
        if (tri_cnt / len(trigrams)) >= 0.35:
            return 0.05, f"repetitive phrase '{' '.join(most_tri)}'"

    # Diversity metric: Type-Token Ratio (TTR)
    ttr = len(counts) / len(words)

    # Combined fitness score
    fitness = (len(counts) * 2.5) + (ttr * 6.0) - (dom_ratio * 4.0)
    return round(fitness, 2), "coherent"


class AdaptiveFormantTuner:
    """Auto-tunes acoustic filtering, denoising, and ASR decoding parameters."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.recognizers = {}
        self._init_recognizers()

    def _init_recognizers(self):
        d = Path(self.cfg["models_dir"]) / self.cfg["offline_model"]
        files = {
            "encoder": str(d / "encoder.int8.onnx"),
            "decoder": str(d / "decoder.int8.onnx"),
            "joiner": str(d / "joiner.int8.onnx"),
            "tokens": str(d / "tokens.txt"),
        }
        # Pre-build recognizers for candidate blank penalties
        for bp in (0.2, 0.5, 0.8, 1.1):
            key = f"greedy_{bp}"
            self.recognizers[key] = sherpa_onnx.OfflineRecognizer.from_transducer(
                **files,
                num_threads=min(self.cfg.get("threads", 4), 4),
                model_type="nemo_transducer",
                decoding_method="greedy_search",
                blank_penalty=bp,
            )

    def optimize_chunk(self, chunk: np.ndarray, dur_sec: float) -> dict:
        """Autonomously search parameter space to maximize coherent word formation."""
        norm = lambda c: c * (0.8 / max(float(np.abs(c).max()), 1e-6))
        best_candidate = {
            "score": 0.0,
            "text": "",
            "reason": "empty",
            "params": {}
        }

        # Parameter Search Grid
        filter_configs = [
            {"low_hz": 120.0, "boost_db": 3.0},
            {"low_hz": 150.0, "boost_db": 4.5},
            {"low_hz": 100.0, "boost_db": 2.0},
        ]

        denoiser_types = ["dpdfnet4.onnx", "gtcrn_simple.onnx", "none"]

        for f_cfg in filter_configs:
            filtered = apply_formant_filter(chunk, low_hz=f_cfg["low_hz"], boost_db=f_cfg["boost_db"])

            for den_name in denoiser_types:
                if den_name == "none":
                    enhanced = filtered
                else:
                    cfg_den = dict(self.cfg, robust_denoiser=den_name)
                    enhanced = audio.denoise(filtered, cfg_den)

                # Try pure enhanced and 25% original blend
                for mix in (0.0, 0.25):
                    if mix > 0.0:
                        candidate_audio = norm(enhanced + mix * filtered)
                    else:
                        candidate_audio = norm(enhanced)

                    # Dynamic Blank Penalty Sweep (0.2 -> 0.5 -> 0.8 -> 1.1)
                    # Adaptive early stopping: if repetitive, back off
                    for bp in (0.2, 0.5, 0.8, 1.1):
                        rec = self.recognizers[f"greedy_{bp}"]
                        stream = rec.create_stream()
                        stream.accept_waveform(SR, candidate_audio)
                        rec.decode_stream(stream)
                        text = stream.result.text.strip()

                        score, reason = score_coherence(text, dur_sec)

                        # Check if this beats previous best
                        if score > best_candidate["score"]:
                            best_candidate = {
                                "score": score,
                                "text": text,
                                "reason": reason,
                                "params": {
                                    "low_hz": f_cfg["low_hz"],
                                    "boost_db": f_cfg["boost_db"],
                                    "denoiser": den_name,
                                    "mix": mix,
                                    "blank_penalty": bp
                                }
                            }

                        # If this setting produced repetitive tokens, don't test higher blank penalties
                        if "repetitive" in reason:
                            break

        return best_candidate


def scan_file(wav_path: str, start_sec=0.0, dur_sec=30.0, win_sec=5.0, step_sec=1.5):
    print(f"=== Auto-Tuning Formant Scanner: {wav_path} ===")
    print(f"Range: {start_sec:.1f}s - {start_sec + dur_sec:.1f}s | Window: {win_sec}s | Step: {step_sec}s\n")

    cfg = config.load()
    tuner = AdaptiveFormantTuner(cfg)

    full_audio = get_audio(wav_path, start_sec=start_sec, dur_sec=dur_sec)
    win_len = int(win_sec * SR)
    step_len = int(step_sec * SR)

    windows = [(i, i + win_len) for i in range(0, len(full_audio) - win_len + 1, step_len)]
    if not windows:
        windows = [(0, len(full_audio))]

    results = []
    for start_idx, end_idx in windows:
        t0 = start_sec + (start_idx / SR)
        t1 = start_sec + (end_idx / SR)
        chunk = full_audio[start_idx:end_idx]

        best = tuner.optimize_chunk(chunk, win_sec)
        if best["score"] > 0.5 and best["text"]:
            p = best["params"]
            param_summary = f"{p['denoiser'][:5]}|hp={int(p['low_hz'])}|b={p['boost_db']:.1f}|p={p['blank_penalty']}"
            print(f"[{t0:5.1f}s - {t1:5.1f}s] (score: {best['score']:5.1f} | {param_summary:<24}) -> {best['text']}")
            results.append((t0, t1, best["text"], best["score"]))

    # Clean deduplicated final synthesis
    print("\n=== Synthesized Coherent Output ===")
    deduped = []
    for s, e, text, sc in results:
        if not deduped:
            deduped.append((s, e, text))
            continue
        prev_s, prev_e, prev_text = deduped[-1]
        if text.lower() in prev_text.lower():
            continue
        if prev_text.lower() in text.lower():
            deduped[-1] = (prev_s, e, text)
            continue
        if s < prev_e:
            words_p = prev_text.split()
            words_c = text.split()
            merged = False
            for k in range(min(len(words_p), len(words_c)), 0, -1):
                if [w.lower() for w in words_p[-k:]] == [w.lower() for w in words_c[:k]]:
                    combined = " ".join(words_p + words_c[k:])
                    deduped[-1] = (prev_s, e, combined)
                    merged = True
                    break
            if not merged:
                deduped.append((s, e, text))
        else:
            deduped.append((s, e, text))

    for s, e, text in deduped:
        print(f"  [{s:5.1f}s - {e:5.1f}s]: {text}")


def main():
    parser = argparse.ArgumentParser(description="Auto-tweaking formant search")
    parser.add_argument("wav", nargs="?", default="/mnt/g/0-WAV/ishehereblake.wav")
    parser.add_argument("--start", type=float, default=6.0, help="start time in seconds")
    parser.add_argument("--dur", type=float, default=18.0, help="duration in seconds")
    parser.add_argument("--win", type=float, default=5.0, help="window length in seconds")
    parser.add_argument("--step", type=float, default=1.5, help="step size in seconds")
    args = parser.parse_args()

    scan_file(args.wav, start_sec=args.start, dur_sec=args.dur, win_sec=args.win, step_sec=args.step)


if __name__ == "__main__":
    main()
