"""Aggressiveness levels for noisy audio (`stt file -a N`, `--robust` == level 3).

0  off: stock VAD (threshold from config) + recognizer
1  loose VAD only (threshold 0.2, segments up to 10 s); no denoising, nearly free
2  + neural denoise, 50% original blended back, VAD 0.15
3  denoise, 30% blend, VAD 0.1, 5 s segments, min speech 0.1 s   (tuned on test_speech.mp3 + noise)
4  level 3 plus a second variant (60% blend, 20 s segments); keeps the more complete transcript
5  `--distant`: normalize-only (no denoiser) and level 3, keeps the more complete transcript. Measured best for
   speech at or below the room-noise level, where the denoiser only erases signal
"""
import sys
import threading
import time
from pathlib import Path
import numpy as np

from . import asr, audio

LEVELS = (0, 1, 2, 3, 4, 5)

_DIAG_TLS = threading.local()

# Complete common short words that must never be treated as truncated morphological prefixes
# of an adjacent word (prevents "we"+"went", "to"+"today's", "the"+"theater", "in"+"Indiana" merges).
COMMON_COMPLETE_SHORT_WORDS = {
    "a", "am", "an", "and", "are", "as", "at", "be", "but", "by", "can", "did", "do", "for",
    "get", "go", "got", "had", "has", "he", "her", "hi", "him", "his", "how", "i", "if", "in",
    "is", "it", "its", "let", "may", "me", "my", "no", "not", "now", "of", "off", "oh", "ok",
    "old", "on", "one", "or", "our", "out", "own", "put", "ran", "run", "saw", "say", "see",
    "she", "sit", "so", "ten", "the", "to", "too", "top", "try", "two", "up", "us", "use",
    "was", "way", "we", "who", "why", "win", "won", "yes", "yet", "you",
    # Common Spanish / Portuguese short complete words
    "al", "con", "da", "das", "de", "del", "dos", "el", "em", "en", "es", "eu", "la", "las",
    "le", "les", "lo", "los", "mas", "mi", "muy", "na", "nas", "ni", "nos", "os", "ou", "por",
    "que", "se", "si", "sin", "su", "sus", "te", "tu", "tus", "um", "uma", "un", "una", "uno",
    "va", "ve", "vi", "ya", "yo",
}

LOW_SNR_FILLER_HALLUCINATIONS = {
    "okay", "ok", "mm-hmm", "mmhmm", "uh-huh", "mhm", "hmm", "thank you", "thanks", "yeah", "bye",
}


def format_interval(start_sec: float, end_sec: float) -> str:
    """Format audio interval as [MM:SS.mmm -> MM:SS.mmm] (or HH:MM:SS.mmm if >= 1h)."""
    def _fmt(sec: float) -> str:
        total_ms = max(0, int(round(sec * 1000)))
        ms = total_ms % 1000
        total_s = total_ms // 1000
        s = total_s % 60
        total_m = total_s // 60
        if total_m >= 60:
            h = total_m // 60
            m = total_m % 60
            return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
        return f"{total_m:02d}:{s:02d}.{ms:03d}"

    return f"[{_fmt(start_sec)} -> {_fmt(end_sec)}]"


def _set_last_segment_diagnostics(diags: list[dict]) -> None:
    _DIAG_TLS.last_diagnostics = list(diags)


def get_last_segment_diagnostics(time_offset: float = 0.0) -> list[dict]:
    """Return acoustic burial & layer diagnostics for the most recent transcribe() call on this thread."""
    raw = getattr(_DIAG_TLS, "last_diagnostics", None) or []
    if time_offset == 0.0:
        return [dict(d) for d in raw]
    shifted = []
    for d in raw:
        item = dict(d)
        s = round(item["start"] + time_offset, 3)
        e = round(item["end"] + time_offset, 3)
        item["start"] = s
        item["end"] = e
        item["location"] = format_interval(s, e)
        shifted.append(item)
    return shifted


def _analyze_and_log_segment_burial(
    segments: list[tuple[float, float, str]],
    conditioned_audio: np.ndarray,
    recovery_tags: dict[tuple[float, float], str] | None = None,
    emit_logs: bool = True,
) -> list[dict]:
    """Compute local SNR vs noise floor and burial depth vs foreground speech, logging buried hits."""
    if not segments or len(conditioned_audio) == 0:
        _set_last_segment_diagnostics([])
        return []

    sr = audio.SR
    n_total = len(conditioned_audio)
    local_floor_prof = audio.compute_dynamic_pregate_profile(conditioned_audio, sr=sr, frame_sec=0.25, local_win_sec=8.0)
    global_floor = float(np.percentile(local_floor_prof, 25)) if len(local_floor_prof) > 0 else 1e-5

    seg_metrics = []
    for s, e, text in segments:
        i0 = max(0, min(n_total - 1, int(s * sr)))
        i1 = max(i0 + 1, min(n_total, int(e * sr)))
        seg_slice = conditioned_audio[i0:i1]

        # Active speech RMS (P80 of 50ms sub-frames so intra-utterance pauses don't depress level)
        fl = int(0.05 * sr)
        nf = len(seg_slice) // fl
        if nf >= 2:
            sub_rms = np.sqrt(np.mean(seg_slice[: nf * fl].reshape(nf, fl) ** 2, axis=1) + 1e-12)
            seg_rms = float(np.percentile(sub_rms, 80))
            sub_p20 = float(np.percentile(sub_rms, 20)) if nf >= 6 else seg_rms
        else:
            seg_rms = float(np.sqrt(np.mean(seg_slice ** 2) + 1e-12))
            sub_p20 = seg_rms

        if len(local_floor_prof) > 0:
            f0 = max(0, min(len(local_floor_prof) - 1, int(s / 0.25)))
            f1 = max(f0 + 1, min(len(local_floor_prof), int(e / 0.25) + 1))
            loc_floor = float(np.median(local_floor_prof[f0:f1]))
        else:
            loc_floor = global_floor
        loc_floor = max(loc_floor, 1e-6)

        snr_db = float(20.0 * np.log10(max(seg_rms, 1e-6) / loc_floor))
        intra_sub_snr = float(20.0 * np.log10(max(sub_p20, 1e-6) / loc_floor))
        tag = (recovery_tags or {}).get((round(s, 2), round(e, 2)), "primary")
        seg_metrics.append((s, e, text, seg_rms, loc_floor, snr_db, intra_sub_snr, tag))

    # Establish foreground reference RMS from clear foreground segments (SNR >= 6 dB) if present
    fg_candidates = [m[3] for m in seg_metrics if m[5] >= 6.0 and m[6] == "primary"]
    if not fg_candidates:
        fg_candidates = [m[3] for m in seg_metrics if m[5] >= 6.0]
    has_fg_ref = len(fg_candidates) > 0
    fg_ref_rms = float(np.percentile(fg_candidates, 85)) if has_fg_ref else max(m[3] for m in seg_metrics)

    diagnostics: list[dict] = []
    fg_count = 0
    buried_count = 0
    deepest_snr = 999.0
    deepest_vs_fg = 0.0

    for s, e, text, seg_rms, loc_floor, snr_db, intra_sub_snr, tag in seg_metrics:
        depth_vs_fg = float(20.0 * np.log10(max(seg_rms, 1e-6) / max(fg_ref_rms, 1e-6))) if has_fg_ref else 0.0
        depth_vs_fg = min(0.0, depth_vs_fg)

        if snr_db <= 3.0:
            layer = "subnoise-buried"
        elif snr_db <= 8.5 or (has_fg_ref and depth_vs_fg <= -9.0) or tag in ("gap-subnoise", "subnoise"):
            layer = "background-faint"
        else:
            layer = "foreground"

        loc_str = format_interval(s, e)
        diag = {
            "start": round(s, 3),
            "end": round(e, 3),
            "location": loc_str,
            "layer": layer,
            "snr_db": round(snr_db, 1),
            "burial_below_floor_db": round(-snr_db, 1),
            "depth_vs_foreground_db": round(depth_vs_fg, 1) if has_fg_ref else None,
            "recovery_source": tag,
        }
        if intra_sub_snr <= 3.0 and layer == "foreground":
            diag["intra_burial_snr_db"] = round(intra_sub_snr, 1)
        diagnostics.append(diag)

        if layer == "foreground":
            fg_count += 1
        else:
            buried_count += 1
            deepest_snr = min(deepest_snr, snr_db)
            deepest_vs_fg = min(deepest_vs_fg, depth_vs_fg)
            if emit_logs:
                fg_clause = f", {depth_vs_fg:+.1f} dB vs foreground" if has_fg_ref else ""
                snippet = text if len(text) <= 72 else text[:69] + "..."
                if snr_db < 0.0:
                    print(
                        f"[stt] Buried speech detected @ {loc_str}: {abs(snr_db):.1f} dB BELOW noise floor "
                        f"(local SNR {snr_db:+.1f} dB{fg_clause}, via {tag}) -> \"{snippet}\"",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[stt] Faint/background speech detected @ {loc_str}: {snr_db:+.1f} dB vs noise floor"
                        f"{fg_clause} (via {tag}) -> \"{snippet}\"",
                        file=sys.stderr,
                    )

    if emit_logs and fg_count > 0 and buried_count > 0:
        print(
            f"[stt] Acoustic layers recovered: {fg_count} foreground + {buried_count} buried/background "
            f"segment(s) (deepest at {deepest_snr:+.1f} dB vs noise floor, {deepest_vs_fg:+.1f} dB vs foreground).",
            file=sys.stderr,
        )

    _set_last_segment_diagnostics(diagnostics)
    return diagnostics


def _chunked_direct_decode(rec, samples: np.ndarray, max_chunk_sec: float = 22.0, overlap_sec: float = 2.0) -> list[tuple[float, float, str]]:
    """Decode audio without VAD in bounded windows to prevent O(T^2) conformer OOM on long clips."""
    if len(samples) == 0:
        return []
    dur = len(samples) / audio.SR
    max_len = int(max_chunk_sec * audio.SR)
    if len(samples) <= max_len:
        txts = asr.transcribe(rec, [samples])
        txt = txts[0].strip() if txts else ""
        return [(0.0, round(dur, 2), txt)] if txt else []

    step_len = max(int((max_chunk_sec - overlap_sec) * audio.SR), audio.SR * 5)
    spans = []
    chunks = []
    for s_idx in range(0, len(samples), step_len):
        e_idx = min(len(samples), s_idx + max_len)
        spans.append((round(s_idx / audio.SR, 2), round(e_idx / audio.SR, 2)))
        chunks.append(samples[s_idx:e_idx])
        if e_idx >= len(samples):
            break

    txts = asr.transcribe(rec, chunks)
    out = []
    for (s, e), t in zip(spans, txts):
        clean_t = (t or "").strip()
        if clean_t:
            out.append((s, e, clean_t))
    return out


def _variants(level: int, cfg: dict) -> list[dict]:
    """Each variant: denoise (bool), mix (original blended back), vad overrides."""
    loose = dict(vad_min_silence=cfg["robust_vad_min_silence"], vad_min_speech=cfg["robust_vad_min_speech"])
    if level == 5:
        quiet = dict(denoise=False, norm=True, vad=dict(loose, vad_threshold=0.1, vad_max_speech=5.0))
        return [quiet, _variants(3, cfg)[0]]
    if level == 1:
        return [dict(denoise=False, norm=True, vad=dict(loose, vad_threshold=0.2, vad_max_speech=10.0))]
    if level == 2:
        return [dict(denoise=True, mix=0.5, vad=dict(loose, vad_threshold=0.15, vad_max_speech=10.0))]
    main = dict(denoise=True, mix=cfg["robust_mix"],
                vad=dict(loose, vad_threshold=cfg["robust_vad_threshold"], vad_max_speech=cfg["robust_vad_max_speech"]))
    if level == 3:
        return [main]
    alt = dict(denoise=True, mix=0.6, vad=dict(main["vad"], vad_max_speech=20.0))
    return [main, alt]


def pregain(x: np.ndarray, target: float = 0.1, max_gain: float = 100.0) -> float:
    """Gain that brings the loud (speech) frames to a normal level; never attenuates."""
    n = len(x) // 512 * 512
    if n == 0:
        return 1.0
    e = np.sqrt((x[:n].reshape(-1, 512) ** 2).mean(axis=1))
    return float(np.clip(target / max(float(np.percentile(e, 95)), 1e-6), 1.0, max_gain))


def auto_tune_settings(dur_sec: float, samples: np.ndarray | None = None) -> dict:
    """Calculate optimal acoustic and decoding settings dynamically based on audio shape.

    If samples are provided, uses full physical acoustic profiling (quantiles, hum, SNR).
    Otherwise falls back to duration-based defaults.
    """
    if samples is not None and len(samples) > 0:
        prof = audio.profile_audio(samples)
        return {
            "win_sec": prof.win_sec,
            "step_sec": prof.step_sec,
            "blank_penalty": prof.blank_penalty,
            "boost_db": prof.formant_boost_db,
            "low_hz": prof.hpf_cutoff_hz,
            "notch_freqs": prof.notch_freqs,
            "pregate_threshold": prof.pregate_threshold,
            "snr_db": prof.snr_db,
            "reasoning": prof.reasoning,
        }

    if dur_sec <= 20.0:
        win_sec = 5.0
        step_sec = 1.25
        blank_penalty = 0.7
        boost_db = 3.5
        low_hz = 120.0
        overlap_pct = 75
    elif dur_sec <= 120.0:
        win_sec = 5.0
        step_sec = 1.75
        blank_penalty = 0.6
        boost_db = 3.0
        low_hz = 120.0
        overlap_pct = 65
    else: # Large file (> 2 min)
        win_sec = 6.0
        step_sec = 3.0
        blank_penalty = 0.5
        boost_db = 3.0
        low_hz = 130.0
        overlap_pct = 50

    reasoning = [
        f"Audio duration is {dur_sec:.1f}s. Assuming acoustic noise floor with faint vocal formants.",
        f"High-pass filter cutoff set to {low_hz:.0f} Hz to strip low-frequency rumble.",
        f"Speech formant equalization set to +{boost_db:.1f} dB (300-3200 Hz) to lift vowel/consonant harmonics.",
        f"Sliding window length set to {win_sec:.1f}s with {step_sec:.2f}s stride ({overlap_pct}% overlap).",
        f"Transducer blank penalty set to {blank_penalty:.2f} to encourage phoneme emission on weak cues.",
        "Multi-model denoising enabled (GTCRN complex spectral mapping + DPDFNet-4 + normalized HP)."
    ]

    return {
        "win_sec": win_sec,
        "step_sec": step_sec,
        "blank_penalty": blank_penalty,
        "boost_db": boost_db,
        "low_hz": low_hz,
        "notch_freqs": (),
        "pregate_threshold": 1e-4,
        "snr_db": 5.0,
        "overlap_pct": overlap_pct,
        "reasoning": reasoning
    }


def score_coherence(text: str, dur_sec: float) -> tuple[float, str]:
    """Score transcript based on lexical diversity, grammar plausibility, and non-repetition."""
    words = [w.strip(".,?!;:\"'()").lower() for w in text.split() if w.strip(".,?!;:\"'()")]
    if not words:
        return 0.0, "empty"

    wps_max = 8.0 if (dur_sec < 1.2 or len(words) <= 3) else 5.0
    wps = len(words) / max(dur_sec, 0.4)
    if wps > wps_max:
        return -5.0, f"too fast / hallucination ({wps:.1f} wps)"

    from collections import Counter
    import math
    counts = Counter(words)
    most_common_word, most_common_cnt = counts.most_common(1)[0]
    dom_ratio = most_common_cnt / len(words)

    # Calculate token Shannon entropy
    probs = [c / len(words) for c in counts.values()]
    entropy = max(0.0, -sum(p * math.log2(p) for p in probs))

    # Check for degenerate repetition loops (4+ words with extreme low entropy or >67% dominance)
    if len(words) >= 4 and (entropy < 1.0 or dom_ratio >= 0.67):
        return 0.05, f"repetitive token '{most_common_word}' (entropy={entropy:.2f})"

    # Allow natural short emphatic human repetition (e.g. "no, no, no") up to 3 words
    has_consecutive_3 = any(w1 == w2 == w3 for w1, w2, w3 in zip(words[:-2], words[1:-1], words[2:]))
    if has_consecutive_3 and len(words) > 3:
        return 0.05, f"repetitive token '{most_common_word}'"

    if len(words) >= 4:
        bigrams = list(zip(words[:-1], words[1:]))
        bi_counts = Counter(bigrams)
        most_bi, bi_cnt = bi_counts.most_common(1)[0]
        if (bi_cnt / len(bigrams)) >= 0.4:
            return 0.05, f"repetitive phrase '{' '.join(most_bi)}'"

    ttr = len(counts) / len(words)
    fitness = (len(counts) * 2.5) + (ttr * 6.0) - (dom_ratio * 4.0)
    return round(fitness, 2), "coherent"


def compute_dynamic_stream_weights(prof: audio.AcousticProfile, model_names: list[str]) -> dict[str, float]:
    """Calculate normalized acoustic-aware weights for candidate streams based on noise shape and SNR."""
    base_weights = {
        "hybrid": 0.24,
        "gtcrn": 0.22,
        "dp4": 0.18,
        "blend": 0.15,
        "formant_agc": 0.14,
        "spec_gate": 0.12,
        "hp": 0.10,
    }

    gamma_drone = min(1.0, prof.rumble_fraction * 1.5 + (0.3 if prof.hum_freq else 0.0))
    gamma_whisper = max(0.0, min(1.0, (8.0 - prof.snr_db) / 10.0)) * (1.0 - gamma_drone * 0.5)

    adj = {
        "gtcrn": 0.15 * gamma_drone - 0.10 * gamma_whisper,
        "hybrid": 0.08 * gamma_drone + 0.08 * gamma_whisper,
        "blend": 0.04 * gamma_drone + 0.10 * gamma_whisper,
        "formant_agc": -0.08 * gamma_drone + 0.18 * gamma_whisper,
        "dp4": 0.06 * gamma_whisper,
        "spec_gate": 0.02 * gamma_whisper,
        "hp": -0.05 * gamma_drone,
    }

    final_w = {}
    for name in model_names:
        bw = base_weights.get(name, 0.15)
        delta = adj.get(name, 0.0)
        final_w[name] = max(0.04, bw + delta)

    total = sum(final_w.values()) or 1.0
    return {k: round(v / total, 4) for k, v in final_w.items()}


def cn_rover_fuse(stream_candidates: list[dict], dur_sec: float) -> list[dict]:
    """Time-synchronous Confusion Network ROVER hypothesis fusion across candidate streams."""
    valid_streams = [s for s in stream_candidates if s.get("words")]
    if not valid_streams:
        return []
    if len(valid_streams) == 1:
        return valid_streams[0]["words"]

    tagged_words = []
    for s_idx, stream in enumerate(valid_streams):
        w_weight = stream.get("weight", 0.20)
        for w in stream["words"]:
            tagged_words.append({
                "word": w["word"],
                "start": w["global_start"],
                "end": w["global_end"],
                "conf": w.get("conf", 0.5),
                "stream_idx": s_idx,
                "stream_weight": w_weight,
            })

    if not tagged_words:
        return []

    tagged_words.sort(key=lambda x: x["start"])
    slots = []
    for tw in tagged_words:
        placed = False
        for slot in slots:
            slot_start = min(w["start"] for w in slot)
            slot_end = max(w["end"] for w in slot)
            overlap = max(0.0, min(slot_end, tw["end"]) - max(slot_start, tw["start"]))
            duration = max(slot_end, tw["end"]) - min(slot_start, tw["start"])
            iou = overlap / max(duration, 1e-6)
            center_dist = abs(((tw["start"] + tw["end"]) / 2.0) - ((slot_start + slot_end) / 2.0))
            if iou >= 0.20 or center_dist <= 0.28:
                if not any(w["stream_idx"] == tw["stream_idx"] for w in slot):
                    slot.append(tw)
                    placed = True
                    break
        if not placed:
            slots.append([tw])

    fused_words = []
    total_slot_weight = sum(stream.get("weight", 0.20) for stream in valid_streams) or 1.0
    for slot in slots:
        grouped = {}
        for w in slot:
            clean = w["word"].strip(".,?!;:\"'()").lower()
            if not clean:
                continue
            if clean not in grouped:
                grouped[clean] = {
                    "voters": set(),
                    "weights": 0.0,
                    "confs": [],
                    "individual_weights": [],
                    "words": [],
                    "starts": [],
                    "ends": [],
                }
            grouped[clean]["voters"].add(w["stream_idx"])
            grouped[clean]["weights"] += w["stream_weight"]
            grouped[clean]["confs"].append(w["conf"])
            grouped[clean]["individual_weights"].append(w["stream_weight"])
            grouped[clean]["words"].append(w["word"])
            grouped[clean]["starts"].append(w["start"])
            grouped[clean]["ends"].append(w["end"])

        if not grouped:
            continue

        best_clean = None
        best_score = -1.0
        for clean, data in grouped.items():
            voter_weight_sum = data["weights"]
            weight_frac = voter_weight_sum / total_slot_weight

            w_arr = np.array(data["individual_weights"], dtype=np.float32)
            c_arr = np.array(data["confs"], dtype=np.float32)
            weighted_conf = float(np.sum(w_arr * c_arr) / max(np.sum(w_arr), 1e-6))

            score = 0.45 * weight_frac + 0.55 * weighted_conf
            if len(data["voters"]) == 1:
                # Allow high-confidence single-specialist recoveries (e.g. dp4 / formant_agc / blend)
                # while still rejecting low-confidence single-stream noise hallucinations.
                if weighted_conf < 0.52:
                    continue
                if weight_frac < 0.14 and weighted_conf < 0.72:
                    continue
            if score > best_score:
                best_score = score
                best_clean = clean

        if best_clean is not None:
            data = grouped[best_clean]
            best_repr = max(data["words"], key=lambda x: (len(x), x[0].isupper()))
            fused_words.append({
                "word": best_repr,
                "global_start": round(float(np.median(data["starts"])), 3),
                "global_end": round(float(np.median(data["ends"])), 3),
                "conf": float(np.mean(data["confs"])),
            })

    fused_words.sort(key=lambda x: x["global_start"])
    return fused_words


def merge_words_morphological(emitted: list[dict], new_words: list[dict]):
    """Merge newly detected window words into global sequence without intra-window erasures."""
    # Snapshot prior_len so adjacent words emitted by the SAME window never delete one another.
    prior_len = len(emitted)

    for nw in new_words:
        nw_clean = nw["word"].strip(".,?!;:\"'()").lower()
        if not nw_clean:
            continue

        duplicate = False
        for idx in range(prior_len - 1, -1, -1):
            past = emitted[idx]
            # Stop scanning back once prior words are well before this word's timestamp
            if past["global_end"] < nw["global_start"] - 1.10 and past["global_start"] < nw["global_start"] - 1.10:
                break

            past_clean = past["word"].strip(".,?!;:\"'()").lower()
            t_diff = abs(past["global_start"] - nw["global_start"])
            overlap = max(0.0, min(past["global_end"], nw["global_end"]) - max(past["global_start"], nw["global_start"]))
            span = max(past["global_end"], nw["global_end"]) - min(past["global_start"], nw["global_start"])
            iou = overlap / max(span, 1e-6)

            if t_diff <= 0.50 or iou >= 0.25:
                if past_clean == nw_clean:
                    if len(nw["word"]) > len(past["word"]) and past["word"][-1] not in ".,?!;:":
                        past["word"] = nw["word"]
                    past["global_end"] = max(past["global_end"], nw["global_end"])
                    duplicate = True
                    break

                # Morphological boundary truncation repair (e.g. "pur" @ 0.8..1.1 + "purse." @ 0.82..1.25):
                # Require genuine interval overlap / tight start alignment, shorter fragment >= 3 chars,
                # and NOT a complete common short word ("the", "in", "and", "for", etc.).
                if (t_diff <= 0.25 or overlap > 0.02) and (
                    past_clean.startswith(nw_clean) or nw_clean.startswith(past_clean)
                ):
                    shorter = past_clean if len(past_clean) <= len(nw_clean) else nw_clean
                    if len(shorter) >= 3 and shorter not in COMMON_COMPLETE_SHORT_WORDS:
                        if len(nw_clean) > len(past_clean):
                            past["word"] = nw["word"]
                            past["global_end"] = max(past["global_end"], nw["global_end"])
                        duplicate = True
                        break

        if not duplicate:
            emitted.append(nw)


INCOMPLETE_CONTINUATIONS = {
    "it's", "i'm", "we're", "they're", "you're", "he's", "she's", "that's", "what's", "there's",
    "don't", "can't", "won't", "didn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "and", "but", "or", "if", "when", "while", "because", "so", "to", "of", "in", "for", "with",
    "at", "from", "into", "about", "like", "as", "than", "that", "which",
    "the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "our", "their",
    "i", "we", "you", "he", "she", "it", "they", "who", "whom", "whose", "what",
}

DISCOURSE_MARKERS = {
    "oh", "well", "okay", "yeah", "yes", "right", "alright", "uh", "um", "ah", "like", "no",
}


def assemble_cohesive_sentences(
    emitted_words: list[dict],
    audio_signal: np.ndarray | None = None,
    local_noise_floor: np.ndarray | None = None,
    sr: int = audio.SR,
) -> list[tuple[float, float, str]]:
    """Synthesize full cohesive conversational sentences from word timestamps using syntactic and acoustic continuity."""
    if not emitted_words:
        return []

    words = sorted(emitted_words, key=lambda w: w.get("global_start", 0.0))
    sentences = []
    curr_words = [words[0]]

    for next_word in words[1:]:
        prev_word_entry = curr_words[-1]
        prev_text = prev_word_entry["word"].strip()
        next_text = next_word["word"].strip()
        gap = next_word["global_start"] - prev_word_entry["global_end"]

        clean_prev = prev_text.rstrip(".,?!;:\"'").lower()
        clean_next = next_text.lstrip(".,?!;:\"'").lower()
        has_terminal_punct = prev_text.endswith((".", "!", "?"))

        max_bridge_gap = 0.65
        if clean_prev in INCOMPLETE_CONTINUATIONS:
            max_bridge_gap = 2.25
        elif clean_prev in DISCOURSE_MARKERS or clean_next in DISCOURSE_MARKERS:
            max_bridge_gap = 1.65
        elif not has_terminal_punct:
            max_bridge_gap = 1.10
        else:
            max_bridge_gap = 0.75

        is_acoustically_continuous = False
        if 0.0 < gap <= max_bridge_gap and audio_signal is not None:
            g_start_idx = int(prev_word_entry["global_end"] * sr)
            g_end_idx = int(next_word["global_start"] * sr)
            if g_end_idx > g_start_idx:
                gap_slice = audio_signal[g_start_idx:g_end_idx]
                gap_rms = float(np.sqrt(np.mean(gap_slice ** 2) + 1e-12))

                f_idx = min(len(local_noise_floor) - 1, int(prev_word_entry["global_end"] / 0.25)) if local_noise_floor is not None and len(local_noise_floor) > 0 else 0
                floor_rms = float(local_noise_floor[f_idx]) if local_noise_floor is not None and len(local_noise_floor) > 0 else 0.005
                if gap_rms >= floor_rms * 0.90:
                    is_acoustically_continuous = True

        should_bridge = False
        if gap <= 0.0:
            should_bridge = True
        elif gap <= max_bridge_gap:
            if clean_prev in INCOMPLETE_CONTINUATIONS:
                should_bridge = True
            elif clean_prev in DISCOURSE_MARKERS or clean_next in DISCOURSE_MARKERS:
                should_bridge = True
            elif not has_terminal_punct:
                should_bridge = True
            elif is_acoustically_continuous:
                should_bridge = True

        if should_bridge:
            curr_words.append(next_word)
        else:
            s_text = " ".join(w["word"] for w in curr_words).strip()
            if s_text:
                sentences.append((round(curr_words[0]["global_start"], 2), round(curr_words[-1]["global_end"], 2), s_text))
            curr_words = [next_word]

    if curr_words:
        s_text = " ".join(w["word"] for w in curr_words).strip()
        if s_text:
            sentences.append((round(curr_words[0]["global_start"], 2), round(curr_words[-1]["global_end"], 2), s_text))

    return sentences


def subnoise_transcribe(
    rec,
    samples: np.ndarray,
    cfg: dict,
    params: dict | None = None,
    progress_cb=None,
    model: str | None = None,
    threads: int | None = None,
    _internal_gap_scan: bool = False,
) -> list[tuple[float, float, str]]:
    """Aggressive multi-model denoising, adaptive filtering, and formant scanning for sub-noise speech."""
    if len(samples) == 0:
        if not _internal_gap_scan:
            _set_last_segment_diagnostics([])
        return []

    # 0. Reconstruct clipped microphone plateaus before linear / neural enhancement
    if float(np.max(np.abs(samples))) >= 0.985:
        samples, n_clipped = audio.soft_declip(samples)
        if n_clipped > 0 and not _internal_gap_scan:
            print(f"[stt] Subnoise: reconstructed {n_clipped} clipped saturation samples", file=sys.stderr)

    dur = len(samples) / audio.SR
    prof = audio.profile_audio(samples)
    tuned = params or auto_tune_settings(dur, samples=samples)

    low_hz = float(tuned.get("low_hz", prof.hpf_cutoff_hz))
    boost_db = float(tuned.get("boost_db", prof.formant_boost_db))
    win_len = max(int(1.5 * audio.SR), int(float(tuned.get("win_sec", prof.win_sec)) * audio.SR))
    step = max(int(0.5 * audio.SR), int(float(tuned.get("step_sec", prof.step_sec)) * audio.SR))
    bp = tuned.get("blank_penalty", prof.blank_penalty)
    pregate = float(tuned.get("pregate_threshold", prof.pregate_threshold))
    notch_freqs = tuple(tuned.get("notch_freqs", prof.notch_freqs))
    do_declick = bool(tuned.get("declick", True))
    declick_threshold = float(tuned.get("declick_threshold", 4.8))

    if not _internal_gap_scan:
        print(
            f"[stt] Subnoise mode: duration={dur:.1f}s, window={tuned.get('win_sec', prof.win_sec)}s, "
            f"step={tuned.get('step_sec', prof.step_sec)}s, HP={low_hz:.1f}Hz, boost={boost_db:.1f}dB, "
            f"notches={list(notch_freqs)}, blank_penalty={bp:.2f}, pregate={pregate:.6f}",
            file=sys.stderr,
        )

    # 1. Zero-phase Butterworth HPF + Harmonic Notches + Formant Conditioning
    hp_base = audio.butterworth_highpass(samples, cutoff_hz=low_hz, order=4)
    if notch_freqs:
        hp_base = audio.harmonic_notch_filter(hp_base, freqs=notch_freqs, bw_hz=6.0)

    if do_declick:
        hp_base, n_clicks = audio.suppress_impulsive_clicks(hp_base, threshold_sigma=declick_threshold)
        if n_clicks > 0 and not _internal_gap_scan:
            print(f"[stt] Subnoise: suppressed {n_clicks} mechanical impulse clicks", file=sys.stderr)

    if boost_db > 0.0:
        if prof.snr_db <= 6.0:
            hp = audio.whisper_formant_filter(hp_base, boost_db=boost_db)
        else:
            hp = audio.speech_formant_filter(hp_base, boost_db=boost_db)
    else:
        hp = hp_base

    # Dedicated formant-lifted copy so buried background whispers still emerge even when a loud
    # foreground speaker inflated global prof.snr_db >= 18 dB (which sets boost_db=0 on `hp`).
    whisper_hp = audio.whisper_formant_filter(hp_base, boost_db=max(boost_db, 3.5))
    auto_nz = audio.extract_auto_noise_profile(hp_base)
    hp_sub = auto_nz.subtract(hp) if (auto_nz is not None and prof.snr_db <= 8.0) else hp

    # 2. Multi-model enhancement candidates (ordered with `hybrid` first for Tier-1 fast probe)
    models: list[tuple[str, np.ndarray]] = []
    gtcrn_arr = None
    try:
        cfg_g = dict(cfg, robust_denoiser="gtcrn_simple.onnx")
        gtcrn_arr = audio.denoise(hp_sub, cfg_g, noise=auto_nz)
        hybrid_crossover = audio.spectral_crossover_blend(gtcrn_arr, hp)
        models.append(("hybrid", hybrid_crossover))
        models.append(("gtcrn", gtcrn_arr))
        # 80/20 denoised + formant-lifted raw blend restores subtle consonant cues erased by neural mask
        models.append(("blend", (0.80 * gtcrn_arr + 0.20 * whisper_hp).astype(np.float32)))
    except Exception as e:
        if not _internal_gap_scan:
            print(f"[stt] Warning: GTCRN denoiser unavailable: {e}", file=sys.stderr)

    try:
        cfg_d4 = dict(cfg, robust_denoiser="dpdfnet4.onnx")
        # Limit DPDFNet4 attenuation to 15 dB so sub-noise speech formants are preserved instead of gated to 0
        models.append(("dp4", audio.denoise(hp, cfg_d4, noise=auto_nz, attenuation_limit_db=15.0)))
    except Exception as e:
        if not _internal_gap_scan:
            print(f"[stt] Warning: DPDFNet4 denoiser unavailable: {e}", file=sys.stderr)

    is_long_audio = len(samples) > 180 * audio.SR
    if not is_long_audio:
        try:
            models.append(("spec_gate", audio.adaptive_spectral_gate(hp)))
            models.append(("formant_agc", audio.multiband_formant_agc(whisper_hp)))
        except Exception as e:
            if not _internal_gap_scan:
                print(f"[stt] Warning: Pure-NumPy DSP filters unavailable: {e}", file=sys.stderr)

    models.append(("hp", hp))

    all_cand_names = [m[0] for m in models] + (["spec_gate", "formant_agc"] if is_long_audio else [])
    stream_weights = compute_dynamic_stream_weights(prof, all_cand_names)
    if not _internal_gap_scan:
        print(f"[stt] Subnoise: using {len(all_cand_names)} enhancement streams: {all_cand_names}", file=sys.stderr)
        print(f"[stt] Subnoise: dynamic stream weights: {stream_weights}", file=sys.stderr)

    # Pre-compute continuous local noise floor envelopes per stream for accurate noise-bounded AGC
    local_noise_floor = audio.compute_dynamic_pregate_profile(hp)
    stream_floors: dict[str, float] = {}
    for s_name, s_arr in models:
        fl = int(0.05 * audio.SR)
        nf = len(s_arr) // fl
        if nf >= 2:
            s_rms = np.sqrt(np.mean(s_arr[: nf * fl].reshape(nf, fl) ** 2, axis=1) + 1e-12)
            nz_rms = s_rms[s_rms > 1e-7]
            stream_floors[s_name] = float(np.percentile(nz_rms, 20)) if len(nz_rms) >= 2 else float(np.percentile(s_rms, 20))
        else:
            stream_floors[s_name] = prof.noise_floor_rms

    # Quantized 3-tier blank penalty recognizer pool with modified_beam_search for subfloor speech recovery
    rec_t1 = asr.get_offline(cfg, model=model, threads=threads, blank_penalty=0.35, decoding_method="modified_beam_search", max_active_paths=16)
    rec_t2 = asr.get_offline(cfg, model=model, threads=threads, blank_penalty=float(bp or 0.65), decoding_method="modified_beam_search", max_active_paths=16)
    rec_t3 = asr.get_offline(cfg, model=model, threads=threads, blank_penalty=min(1.15, float(bp or 0.65) + 0.25), decoding_method="modified_beam_search", max_active_paths=16)

    # 3. Sliding window formant scan with context margins and dynamic pre-gating
    if len(samples) <= win_len:
        windows = [(0, len(samples))]
    else:
        windows = [(i, i + win_len) for i in range(0, len(samples) - win_len + 1, step)]
        if windows[-1][1] < len(samples):
            windows.append((len(samples) - win_len, len(samples)))

    margin_samples = int(0.75 * audio.SR)
    fade_samples = int(0.05 * audio.SR)
    raw_hits = []
    emitted_words = []
    total_wins = len(windows)
    skipped_pregate = 0
    fast_exits = 0

    last_progress_time = 0.0
    last_progress_pct = -5

    def extract_core_words(
        item: dict,
        left_margin_sec: float,
        core_dur_sec: float,
        ext_start_sec: float,
        has_left_neighbor: bool = False,
        has_right_neighbor: bool = False,
    ) -> tuple[list[dict], float, str]:
        tokens = item.get("tokens", [])
        timestamps = item.get("timestamps", [])
        durations = item.get("durations", [])
        tok_confs = item.get("token_confs", [])
        seq_conf = float(item.get("confidence", 0.5))

        if item.get("zero_skip_ratio", 0.0) > 0.60 and len(tokens) >= 3:
            return [], -10.0, "zero-skip hallucination burst"

        words_with_meta = []
        if tokens and len(timestamps) == len(tokens):
            durs_iter = durations if len(durations) == len(tokens) else [0.0] * len(tokens)
            confs_iter = tok_confs if len(tok_confs) == len(tokens) else [seq_conf] * len(tokens)
            for tok, ts, dur_tok, t_conf in zip(tokens, timestamps, durs_iter, confs_iter):
                is_word_start = tok.startswith((" ", "\u2581")) or not words_with_meta
                clean_tok = tok.lstrip(" \u2581")
                if not clean_tok:
                    continue
                eff_conf = float(t_conf) * (0.85 if dur_tok <= 0.0 else 1.0)
                if is_word_start:
                    words_with_meta.append({
                        "word": clean_tok,
                        "start": float(ts),
                        "end": float(ts + max(dur_tok, 0.04)),
                        "confs": [eff_conf],
                    })
                else:
                    words_with_meta[-1]["word"] += clean_tok
                    words_with_meta[-1]["end"] = float(max(words_with_meta[-1]["end"], ts + max(dur_tok, 0.02)))
                    words_with_meta[-1]["confs"].append(eff_conf)
        else:
            raw_text = item.get("text", "")
            for w in raw_text.split():
                words_with_meta.append({
                    "word": w,
                    "start": left_margin_sec,
                    "end": left_margin_sec + core_dur_sec,
                    "confs": [seq_conf],
                })

        # Trim outer 0.15s of interior core edges when an overlapping neighbor window covers them cleanly
        left_guard = 0.15 if (has_left_neighbor and core_dur_sec > 2.0) else -0.06
        right_guard = 0.15 if (has_right_neighbor and core_dur_sec > 2.0) else -0.06
        core_min = left_margin_sec + left_guard
        core_max = left_margin_sec + core_dur_sec - right_guard
        core_words = []
        for w in words_with_meta:
            w_center = 0.5 * (w["start"] + w["end"])
            if core_min <= w["start"] <= core_max or core_min <= w_center <= core_max:
                w_conf = float(np.mean(w["confs"])) if w.get("confs") else seq_conf
                core_words.append({
                    "word": w["word"],
                    "global_start": round(ext_start_sec + w["start"], 3),
                    "global_end": round(ext_start_sec + w["end"], 3),
                    "conf": round(w_conf, 4),
                })

        clean_text_lower = " ".join(w["word"] for w in core_words).strip(".,?!;:\"'").lower()
        if clean_text_lower in ("thank you", "thanks", "remember to like and subscribe", "like and subscribe", "bye"):
            if seq_conf < 0.65 or item.get("zero_skip_ratio", 0.0) > 0.25:
                return [], -10.0, "prior collapse hallucination"

        text = " ".join(w["word"] for w in core_words)
        sc, reason = score_coherence(text, core_dur_sec)
        score = sc * (0.6 + 0.4 * seq_conf)
        return core_words, score, reason

    for idx, (start_idx, end_idx) in enumerate(windows):
        t0 = start_idx / audio.SR
        t1 = end_idx / audio.SR
        has_left = idx > 0
        has_right = idx < total_wins - 1

        if progress_cb:
            pct = int((idx + 1) / total_wins * 100) if total_wins > 0 else 100
            now = time.time()
            if pct - last_progress_pct >= 5 or now - last_progress_time >= 1.0:
                progress_cb(idx + 1, total_wins, t1, dur)
                last_progress_pct = pct
                last_progress_time = now

        f_idx = min(len(local_noise_floor) - 1, int(t0 / 0.25)) if len(local_noise_floor) > 0 else 0
        local_floor_rms = float(local_noise_floor[f_idx]) if len(local_noise_floor) > 0 else prof.noise_floor_rms
        window_raw = hp[start_idx:end_idx]
        window_den = gtcrn_arr[start_idx:end_idx] if gtcrn_arr is not None else None

        if not audio.evaluate_window_pregate(window_raw, local_floor_rms, denoised_window=window_den):
            skipped_pregate += 1
            continue

        win_rms = float(np.sqrt(np.mean(window_raw ** 2) + 1e-12))
        local_snr_db = 20.0 * np.log10(max(win_rms, 1e-6) / max(local_floor_rms, 1e-6))
        if local_snr_db >= 10.0:
            active_rec = rec_t1
        elif local_snr_db >= 2.0:
            active_rec = rec_t2
        else:
            active_rec = rec_t3

        ext_start = max(0, start_idx - margin_samples)
        ext_end = min(len(samples), end_idx + margin_samples)
        left_margin_sec = (start_idx - ext_start) / audio.SR
        core_dur_sec = (end_idx - start_idx) / audio.SR
        ext_start_sec = ext_start / audio.SR

        def fade_and_agc(raw_slice: np.ndarray, floor_est: float) -> np.ndarray:
            c = raw_slice.copy()
            if ext_start > 0 and len(c) > fade_samples:
                c[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            if ext_end < len(samples) and len(c) > fade_samples:
                c[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            return audio.noise_bounded_agc(c, noise_floor_rms=floor_est)

        def prepare_stream_chunk(s_name: str, audio_arr: np.ndarray) -> np.ndarray:
            s_floor = min(local_floor_rms, stream_floors.get(s_name, local_floor_rms))
            return fade_and_agc(audio_arr[ext_start:ext_end], max(s_floor, 1e-5))

        # Tier 1: Single-Stream Primary Probe (`hybrid` crossover)
        probe_name, probe_arr = models[0]
        probe_chunk = prepare_stream_chunk(probe_name, probe_arr)
        probe_details = asr.transcribe_batch_detailed(active_rec, [probe_chunk])
        probe_item = probe_details[0] if probe_details else {}
        probe_words, probe_sc, _ = extract_core_words(
            probe_item, left_margin_sec, core_dur_sec, ext_start_sec, has_left, has_right
        )
        probe_txt = " ".join(w["word"] for w in probe_words).strip()

        if probe_txt and probe_item.get("confidence", 0.0) >= 0.65 and probe_item.get("zero_skip_ratio", 0.0) <= 0.35 and probe_sc > 1.0:
            raw_hits.append((round(t0, 2), round(t1, 2), probe_txt))
            merge_words_morphological(emitted_words, probe_words)
            fast_exits += 1
            if not _internal_gap_scan and (idx % max(1, total_wins // 10) == 0 or idx == total_wins - 1):
                print(
                    f"[stt] Subnoise window {idx+1}/{total_wins}: '{probe_txt[:60]}...' "
                    f"(score={probe_sc:.1f}, model={probe_name}+fast)",
                    file=sys.stderr,
                )
            continue

        # Tier 2: Escalation across multi-model streams (reuse probe_item when recognizer matches!)
        escalate_rec = rec_t3 if (local_snr_db < 3.0 or not probe_txt) else active_rec
        reuse_probe = escalate_rec is active_rec
        rem_models = models[1:] if reuse_probe else models

        cand_chunks = [prepare_stream_chunk(name, audio_arr) for name, audio_arr in rem_models]
        cand_names = [name for name, _ in rem_models]
        if is_long_audio:
            # Apply DSP enhancement BEFORE Tukey edge fades & noise-bounded AGC
            raw_hp_slice = hp[ext_start:ext_end]
            raw_wh_slice = whisper_hp[ext_start:ext_end]
            try:
                sg_slice = audio.adaptive_spectral_gate(raw_hp_slice)
                cand_chunks.append(fade_and_agc(sg_slice, max(local_floor_rms * 0.5, 1e-5)))
                cand_names.append("spec_gate")
                fa_slice = audio.multiband_formant_agc(raw_wh_slice)
                cand_chunks.append(fade_and_agc(fa_slice, max(local_floor_rms, 1e-5)))
                cand_names.append("formant_agc")
            except Exception:
                pass

        rem_details = asr.transcribe_batch_detailed(escalate_rec, cand_chunks) if cand_chunks else []
        if reuse_probe:
            all_names = [probe_name] + cand_names
            all_details = [probe_item] + rem_details
        else:
            all_names = cand_names
            all_details = rem_details

        rover_inputs = []
        for name, item in zip(all_names, all_details):
            words, sc, _ = extract_core_words(
                item, left_margin_sec, core_dur_sec, ext_start_sec, has_left, has_right
            )
            if words:
                w_weight = stream_weights.get(name, 0.20)
                rover_inputs.append({"words": words, "model": name, "weight": w_weight})

        fused_words = cn_rover_fuse(rover_inputs, core_dur_sec)
        fused_txt = " ".join(w["word"] for w in fused_words).strip()
        sc_fused, _ = score_coherence(fused_txt, core_dur_sec) if fused_txt else (0.0, "empty")

        if fused_txt and sc_fused > 0.5:
            raw_hits.append((round(t0, 2), round(t1, 2), fused_txt))
            merge_words_morphological(emitted_words, fused_words)
            if not _internal_gap_scan and (idx % max(1, total_wins // 10) == 0 or idx == total_wins - 1):
                print(
                    f"[stt] Subnoise window {idx+1}/{total_wins}: '{fused_txt[:60]}...' "
                    f"(score={sc_fused:.1f}, model=cn_rover)",
                    file=sys.stderr,
                )
        else:
            bursts = audio.detect_micro_bursts(window_raw)
            if not bursts:
                bursts = audio.detect_formant_modulation_spans(window_raw, min_smi=1.80, span_sec=2.2)
            for b_start, b_end in bursts:
                mb_start = start_idx + b_start
                mb_end = start_idx + b_end
                mb_ext_start = max(0, mb_start - margin_samples)
                mb_ext_end = min(len(samples), mb_end + margin_samples)
                mb_lmargin = (mb_start - mb_ext_start) / audio.SR
                mb_core_dur = (mb_end - mb_start) / audio.SR

                burst_name, burst_source = models[0]
                mb_c = burst_source[mb_ext_start:mb_ext_end].copy()
                if mb_ext_start > 0 and len(mb_c) > fade_samples:
                    mb_c[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
                if mb_ext_end < len(samples) and len(mb_c) > fade_samples:
                    mb_c[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
                mb_c = audio.noise_bounded_agc(mb_c, noise_floor_rms=max(stream_floors.get(burst_name, local_floor_rms), 1e-5))

                mb_res = asr.transcribe_batch_detailed(rec_t3, [mb_c])
                mb_item = mb_res[0] if mb_res else {}
                mb_words, mb_sc, _ = extract_core_words(mb_item, mb_lmargin, mb_core_dur, mb_ext_start / audio.SR)
                mb_txt = " ".join(w["word"] for w in mb_words).strip()
                if mb_txt and mb_sc > 0.5:
                    raw_hits.append((round(mb_start / audio.SR, 2), round(mb_end / audio.SR, 2), mb_txt))
                    merge_words_morphological(emitted_words, mb_words)
                    if not _internal_gap_scan:
                        print(f"[stt] Subnoise micro-burst at {mb_start/audio.SR:.1f}s: '{mb_txt}' (score={mb_sc:.1f})", file=sys.stderr)
                    break

    if not _internal_gap_scan and (skipped_pregate > 0 or fast_exits > 0):
        print(
            f"[stt] Subnoise cascade: skipped {skipped_pregate}/{total_wins} non-vocal windows, "
            f"{fast_exits} fast-exited ({((skipped_pregate + fast_exits)/total_wins)*100:.1f}% CPU cycles saved)",
            file=sys.stderr,
        )

    # 4. Assembly into cohesive conversational sentences with precise audio timestamps
    segments = assemble_cohesive_sentences(emitted_words, hp, local_noise_floor)
    if segments:
        if not _internal_gap_scan:
            print(f"[stt] Subnoise: synthesized {len(segments)} cohesive sentences from {len(emitted_words)} words", file=sys.stderr)
            tags = {(round(s, 2), round(e, 2)): "subnoise" for s, e, _ in segments}
            _analyze_and_log_segment_burial(segments, hp_base, recovery_tags=tags, emit_logs=True)
        return segments

    if raw_hits:
        if not _internal_gap_scan:
            print(f"[stt] Subnoise: falling back to window deduplication for {len(raw_hits)} hits...", file=sys.stderr)
        deduped = []
        for s, e, text in raw_hits:
            if not deduped:
                deduped.append((s, e, text))
                continue
            prev_s, prev_e, prev_text = deduped[-1]
            prev_words = [w.strip(".,?!;:\"'()") for w in prev_text.lower().split() if w.strip(".,?!;:\"'()")]
            curr_words = [w.strip(".,?!;:\"'()") for w in text.lower().split() if w.strip(".,?!;:\"'()")]
            if not curr_words:
                continue
            if not prev_words:
                deduped[-1] = (prev_s, e, text)
                continue
            is_subset = any(
                prev_words[i : i + len(curr_words)] == curr_words
                for i in range(max(0, len(prev_words) - len(curr_words) + 1))
            ) if len(curr_words) <= len(prev_words) else False
            if is_subset:
                continue
            is_superset = any(
                curr_words[i : i + len(prev_words)] == prev_words
                for i in range(max(0, len(curr_words) - len(prev_words) + 1))
            ) if len(curr_words) >= len(prev_words) else False
            if is_superset:
                deduped[-1] = (prev_s, e, text)
                continue
            if s < prev_e:
                merged = False
                for k in range(min(len(prev_words), len(curr_words)), 0, -1):
                    if prev_words[-k:] == curr_words[:k]:
                        combined = " ".join(prev_text.split() + text.split()[k:])
                        deduped[-1] = (prev_s, e, combined)
                        merged = True
                        break
                if not merged:
                    deduped.append((s, e, text))
            else:
                deduped.append((s, e, text))
        if not _internal_gap_scan:
            tags = {(round(s, 2), round(e, 2)): "subnoise" for s, e, _ in deduped}
            _analyze_and_log_segment_burial(deduped, hp_base, recovery_tags=tags, emit_logs=True)
        return deduped

    if not _internal_gap_scan:
        print("[stt] Subnoise: no coherent speech detected in any window", file=sys.stderr)
        print("[stt] Subnoise fallback: attempting chunked direct decode of HP-filtered audio...", file=sys.stderr)
    direct_segs = _chunked_direct_decode(rec_t2, hp)
    if direct_segs:
        if not _internal_gap_scan:
            print(f"[stt] Subnoise fallback success: '{direct_segs[0][2][:80]}...'", file=sys.stderr)
            tags = {(round(s, 2), round(e, 2)): "subnoise" for s, e, _ in direct_segs}
            _analyze_and_log_segment_burial(direct_segs, hp_base, recovery_tags=tags, emit_logs=True)
        return direct_segs

    if not _internal_gap_scan:
        _set_last_segment_diagnostics([])
    return []


def _is_trivial_filler_hallucination(segs: list[tuple[float, float, str]], snr_db: float, dur_sec: float) -> bool:
    """Return True when noisy audio (<8 dB SNR) produced only sparse 1-2 word filler hallucinations."""
    if not segs or snr_db >= 8.0 or dur_sec < 3.0:
        return False
    all_words = [w.strip(".,?!;:\"'()").lower() for *_, t in segs for w in t.split() if w.strip(".,?!;:\"'()")]
    if not all_words or len(all_words) > 2:
        return False
    joined = " ".join(all_words)
    return joined in LOW_SNR_FILLER_HALLUCINATIONS or all(w in LOW_SNR_FILLER_HALLUCINATIONS for w in all_words)


def _find_untranscribed_gaps(
    segments: list[tuple[float, float, str]],
    dur_sec: float,
    min_gap_sec: float = 0.85,
    guard_sec: float = 0.08,
) -> list[tuple[float, float]]:
    """Return untranscribed [gap_start, gap_end] intervals >= min_gap_sec outside existing segments."""
    if dur_sec < min_gap_sec:
        return []
    if not segments:
        return [(0.0, dur_sec)]

    occupied = sorted((max(0.0, s - guard_sec), min(dur_sec, e + guard_sec)) for s, e, _ in segments)
    merged_occ = [occupied[0]]
    for s, e in occupied[1:]:
        ps, pe = merged_occ[-1]
        if s <= pe:
            merged_occ[-1] = (ps, max(pe, e))
        else:
            merged_occ.append((s, e))

    gaps = []
    cursor = 0.0
    for s, e in merged_occ:
        if s - cursor >= min_gap_sec:
            gaps.append((cursor, s))
        cursor = max(cursor, e)
    if dur_sec - cursor >= min_gap_sec:
        gaps.append((cursor, dur_sec))
    return gaps


def transcribe(
    rec,
    samples: np.ndarray,
    cfg: dict,
    level: int,
    noise=None,
    mode: str = "lead",
    pre: bool = False,
    subnoise: bool = False,
    params: dict | None = None,
    progress_cb=None,
    model: str | None = None,
    threads: int | None = None,
    no_fallback: bool = False,
) -> list[tuple[float, float, str]]:
    """Return [(start, end, text)] for the whole signal, recovering both foreground and buried background speech."""
    if len(samples) == 0:
        _set_last_segment_diagnostics([])
        return []

    if subnoise:
        return subnoise_transcribe(rec, samples, cfg, params=params, progress_cb=progress_cb, model=model, threads=threads)

    # Reconstruct clipped plateaus if microphone input saturated near +/- 1.0
    if float(np.max(np.abs(samples))) >= 0.985:
        samples, _ = audio.soft_declip(samples)

    dur_sec = len(samples) / audio.SR

    # 0. Acoustic physical pre-profiling to eliminate infrasound rumble and electrical hum (Rule 4)
    prof = audio.profile_audio(samples)
    hp_conditioned = audio.butterworth_highpass(samples, cutoff_hz=prof.hpf_cutoff_hz)
    if prof.notch_freqs:
        hp_conditioned = audio.harmonic_notch_filter(hp_conditioned, freqs=prof.notch_freqs)

    working_samples = hp_conditioned if (prof.rumble_fraction >= 0.70 or prof.hum_freq is not None) else samples

    # Hoist neural denoiser output outside try_level so progressive escalation (levels 2->3->4->5)
    # and background gap scanning reuse the exact same denoised buffer instead of running ONNX 4x.
    den_cache: dict[str, np.ndarray | None] = {"den": None, "samples_g": None}

    def _get_denoised_working() -> tuple[np.ndarray, np.ndarray]:
        if den_cache["den"] is None:
            g = pregain(working_samples) if pre else 1.0
            samples_g = working_samples * g
            eff_noise = noise
            if eff_noise is None and prof.snr_db <= 12.0:
                eff_noise = audio.extract_auto_noise_profile(hp_conditioned * g)
            elif eff_noise is not None and g != 1.0:
                eff_noise = audio.NoiseProfile(eff_noise.clip * g)
            src = eff_noise.subtract(samples_g) if (eff_noise is not None and mode in ("sub", "both")) else samples_g
            den_cache["den"] = audio.denoise(src, cfg, eff_noise if mode in ("lead", "both") else None)
            den_cache["samples_g"] = samples_g
        return den_cache["den"], den_cache["samples_g"]

    def try_level(lvl: int, fallback_from: int | None = None) -> list[tuple[float, float, str]]:
        """Try transcription at a specific aggressiveness level, with chunked direct-decode fallback."""
        if fallback_from is not None:
            print(f"[stt] Fallback: level {fallback_from} found no speech, trying level {lvl}...", file=sys.stderr)
        elif lvl == 0:
            pass
        else:
            print(f"[stt] Using aggressiveness level {lvl}", file=sys.stderr)

        if lvl <= 0:
            segs = list(audio.segments(working_samples, cfg))
            found = [(s, e, t) for (s, e, _), t in zip(segs, asr.transcribe(rec, [g[2] for g in segs])) if t]
            if not found:
                print("[stt] Level 0 VAD found no segments; attempting direct decode of full audio...", file=sys.stderr)
                found = _chunked_direct_decode(rec, working_samples)
            return found

        results = []
        for v in _variants(lvl, cfg):
            x = audio.normalize(working_samples) if v.get("norm") else working_samples
            if v["denoise"]:
                den_arr, samples_g = _get_denoised_working()
                x = audio.normalize(den_arr + v["mix"] * samples_g)
            segs = list(audio.segments(x, dict(cfg, **v["vad"])))
            texts = asr.transcribe(rec, [g[2] for g in segs]) if segs else []
            found = [(s, e, t) for (s, e, _), t in zip(segs, texts) if t]
            if not found:
                print(f"[stt] Level {lvl} VAD found no segments; attempting direct decode of full audio...", file=sys.stderr)
                found = _chunked_direct_decode(rec, x)
            results.append(found)

        cap = 4.5 * len(samples) / audio.SR
        ok = [r for r in results if sum(len(t.split()) for *_, t in r) <= cap] or results
        return max(ok, key=lambda r: sum(len(t.split()) for *_, t in r))

    # 1. Foreground pass with progressive escalation
    best = try_level(level)
    if not no_fallback and _is_trivial_filler_hallucination(best, prof.snr_db, dur_sec):
        best = []

    if not no_fallback and not best and level < 5:
        prev_lvl = level
        for fallback_level in range(level + 1, 6):
            cand = try_level(fallback_level, fallback_from=prev_lvl)
            prev_lvl = fallback_level
            if cand and not _is_trivial_filler_hallucination(cand, prof.snr_db, dur_sec):
                best = cand
                break

    # 2. Full subnoise escalation if standard levels 0..5 found nothing (Rule 4)
    if not best and not no_fallback:
        print("[stt] All standard levels failed; attempting subnoise scan...", file=sys.stderr)
        return subnoise_transcribe(rec, samples, cfg, params=params, progress_cb=progress_cb, model=model, threads=threads)

    recovery_tags: dict[tuple[float, float], str] = {(round(s, 2), round(e, 2)): "primary" for s, e, _ in best}

    # 3. Foreground + Buried Background Gap Scan:
    # Even when loud foreground segments were transcribed, scan untranscribed gaps >= 1.6s using
    # multi-cue Speech Presence Probability (SPP) so faint / sub-noise background talkers are not lost.
    can_gap_scan = (
        best
        and not no_fallback
        and bool(cfg.get("subnoise_gap_scan", True))
        and dur_sec >= 2.2
        and hasattr(rec, "create_stream")
    )
    if can_gap_scan:
        gaps = _find_untranscribed_gaps(best, dur_sec, min_gap_sec=0.85, guard_sec=0.08)
        if gaps:
            # Obtain denoised signal once if available to sharpen buried-speech SPP emergence
            den_full = den_cache["den"]
            if den_full is None:
                try:
                    den_full = audio.denoise(hp_conditioned, dict(cfg, robust_denoiser="gtcrn_simple.onnx"))
                except Exception:
                    den_full = None

            recovered_gap_segs: list[tuple[float, float, str]] = []
            fg_word_spans = [
                (s, e, set(w.strip(".,?!;:\"'()").lower() for w in t.split() if w.strip(".,?!;:\"'()")))
                for s, e, t in best
            ]

            for g_start, g_end in gaps:
                gi0 = max(0, int(g_start * audio.SR))
                gi1 = min(len(hp_conditioned), int(g_end * audio.SR))
                gap_hp = hp_conditioned[gi0:gi1]
                gap_den = den_full[gi0:gi1] if (den_full is not None and len(den_full) >= gi1) else None

                cand_spans = audio.find_subnoise_speech_spans(
                    gap_hp, denoised=gap_den, spp_threshold=0.35, min_dur_sec=0.35, merge_gap_sec=1.0, pad_sec=0.35
                )
                if not cand_spans:
                    smi_spans = audio.detect_formant_modulation_spans(gap_hp, min_smi=1.80, span_sec=2.0)
                    for si0, si1 in smi_spans:
                        cand_spans.append({
                            "start_sec": round(si0 / audio.SR, 3),
                            "end_sec": round(si1 / audio.SR, 3),
                            "peak_spp": 0.40,
                            "local_snr_db": 0.0,
                        })

                # Merge overlapping candidate spans within this gap so we decode each region once
                merged_cands: list[tuple[float, float, float]] = []
                for cs in cand_spans:
                    cs_s = g_start + float(cs["start_sec"])
                    cs_e = g_start + float(cs["end_sec"])
                    cs_spp = float(cs.get("peak_spp", 0.4))
                    if merged_cands and cs_s <= merged_cands[-1][1] + 0.6:
                        ps, pe, pspp = merged_cands[-1]
                        merged_cands[-1] = (ps, max(pe, cs_e), max(pspp, cs_spp))
                    else:
                        merged_cands.append((cs_s, cs_e, cs_spp))

                for cs_s, cs_e, cs_spp in merged_cands:
                    # Pad by up to 0.55s inside gap bounds for clean acoustic context
                    slice_s = max(max(0.0, g_start - 0.35), cs_s - 0.55)
                    slice_e = min(min(dur_sec, g_end + 0.35), cs_e + 0.55)
                    if slice_e - slice_s < 0.60:
                        continue
                    si0 = int(slice_s * audio.SR)
                    si1 = int(slice_e * audio.SR)
                    gap_sub_segs = subnoise_transcribe(
                        rec,
                        samples[si0:si1],
                        cfg,
                        params=params,
                        model=model,
                        threads=threads,
                        _internal_gap_scan=True,
                    )
                    for rel_s, rel_e, g_txt in gap_sub_segs:
                        abs_s = round(slice_s + rel_s, 2)
                        abs_e = round(slice_s + rel_e, 2)
                        # Clamp into gap neighborhood and discard boundary echo duplicates of foreground words
                        if abs_e <= g_start + 0.05 or abs_s >= g_end - 0.05:
                            continue
                        g_words = [w.strip(".,?!;:\"'()").lower() for w in g_txt.split() if w.strip(".,?!;:\"'()")]
                        if not g_words:
                            continue
                        is_fg_echo = False
                        for f_s, f_e, f_set in fg_word_spans:
                            ov = max(0.0, min(abs_e, f_e) - max(abs_s, f_s))
                            if ov > 0.15 and f_set and (len(set(g_words) & f_set) / len(set(g_words))) >= 0.70:
                                is_fg_echo = True
                                break
                        if is_fg_echo:
                            continue
                        clamped_s = round(max(g_start - 0.08, abs_s), 2)
                        clamped_e = round(min(g_end + 0.08, max(clamped_s + 0.10, abs_e)), 2)
                        recovered_gap_segs.append((clamped_s, clamped_e, g_txt))
                        recovery_tags[(clamped_s, clamped_e)] = "gap-subnoise"

            if recovered_gap_segs:
                best = sorted(best + recovered_gap_segs, key=lambda item: (item[0], item[1]))

    _analyze_and_log_segment_burial(best, hp_conditioned, recovery_tags=recovery_tags, emit_logs=True)
    return best


# Multilingual transcription and translation
def transcribe_multilingual(
    rec,
    samples: np.ndarray,
    cfg: dict,
    source_lang: str = "auto",
    target_lang: str | None = None,
    translate: bool = False,
    level: int = 0,
    progress_cb=None,
    subnoise: bool = False,
    params: dict | None = None,
) -> list[tuple[float, float, str, str | None]]:
    """Transcribe audio with multilingual model, full robust/subnoise recovery, and optional translation."""
    from . import translation

    if len(samples) == 0:
        _set_last_segment_diagnostics([])
        return []

    try:
        multi_rec = asr.get_multilingual_offline(cfg)
        multi_model_name = cfg.get("multilingual_model")
    except Exception as e:
        print(f"[stt] Multilingual model not available: {e}", file=sys.stderr)
        multi_rec = rec
        multi_model_name = None

    # Run full robust / subnoise + background gap scan pipeline first so noisy & buried speech
    # is never dropped when --translate or --lang-source is used!
    triples = transcribe(
        multi_rec,
        samples,
        cfg,
        level=level,
        subnoise=subnoise,
        params=params,
        progress_cb=progress_cb if not translate else None,
        model=multi_model_name,
    )
    if not triples:
        return []

    detected_lang = source_lang
    if source_lang == "auto":
        combined_text = " ".join(t for _, _, t in triples).strip()
        detected_lang, conf = translation.detect_language_simple(combined_text)
        if conf < cfg.get("lang_detect_confidence_threshold", 0.55):
            audio_lang, audio_conf = asr.detect_language_from_audio(
                multi_rec,
                samples,
                min_duration=cfg.get("lang_detect_min_duration", 2.0),
            )
            if audio_conf > conf:
                detected_lang, conf = audio_lang, audio_conf
        print(f"[stt] Detected language: {detected_lang} (confidence: {conf:.2f})", file=sys.stderr)
        if conf < cfg.get("lang_detect_confidence_threshold", 0.55):
            print(f"[stt] Low confidence ({conf:.2f}), defaulting to English", file=sys.stderr)
            detected_lang = "en"

    valid_langs = set(translation.NLLB_CODES.keys())
    if detected_lang not in valid_langs:
        print(f"[stt] Unsupported language {detected_lang}, defaulting to English", file=sys.stderr)
        detected_lang = "en"

    if translate and target_lang and target_lang != detected_lang:
        print(f"[stt] Translating {detected_lang} -> {target_lang}...", file=sys.stderr)
        return translate_existing_segments(
            triples,
            detected_lang,
            target_lang,
            cfg,
            progress_cb=progress_cb,
        )

    return [(s, e, t, None) for s, e, t in triples]


def translate_existing_segments(
    segments: list[tuple[float, float, str]],
    source_lang: str,
    target_lang: str,
    cfg: dict,
    progress_cb=None,
) -> list[tuple[float, float, str, str]]:
    """Translate already-transcribed segments.
    
    Returns list of (start, end, source_text, translated_text).
    """
    from . import translation
    from pathlib import Path
    
    if source_lang == target_lang:
        return [(s, e, t, t) for s, e, t in segments]
    
    trans_model_dir = Path(cfg["models_dir"]) / cfg.get("translation_model", "nllb-200-distilled-600M")
    
    if not translation.is_translation_available(str(trans_model_dir)):
        print(f"[stt] Translation model not found at {trans_model_dir}", file=sys.stderr)
        return [(s, e, t, f"[Translation model not available]") for s, e, t in segments]
    
    translated = translation.translate_segments(
        segments,
        source_lang,
        target_lang,
        str(trans_model_dir),
        progress_cb=progress_cb,
    )
    
    return translated
