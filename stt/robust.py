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
import time
import numpy as np

from . import asr, audio

LEVELS = (0, 1, 2, 3, 4, 5)


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

    wps = len(words) / max(dur_sec, 0.5)
    if wps > 5.0:
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


def subnoise_transcribe(rec, samples: np.ndarray, cfg: dict, params: dict | None = None, progress_cb=None, model: str | None = None, threads: int | None = None) -> list[tuple[float, float, str]]:
    """Aggressive multi-model denoising, adaptive filtering, and formant scanning for sub-noise speech.

    1. Analyzes acoustic envelope, noise floor, hum, and SNR.
    2. Applies zero-phase Butterworth high-pass + harmonic notch + formant boost.
    3. Evaluates speech-band pre-gating to skip silent windows (60-80% CPU speedup).
    4. Batch-decodes candidate representations (GTCRN, DPDFNet-4, HP) in parallel.
    5. Dynamically escalates blank penalty on candidate speech windows.
    6. Deduplicates overlapping adjacent windows with punctuation-agnostic alignment.
    """
    if len(samples) == 0:
        return []

    dur = len(samples) / audio.SR
    prof = audio.profile_audio(samples)
    tuned = params or auto_tune_settings(dur, samples=samples)

    low_hz = float(tuned.get("low_hz", prof.hpf_cutoff_hz))
    boost_db = float(tuned.get("boost_db", prof.formant_boost_db))
    win_len = int(float(tuned.get("win_sec", prof.win_sec)) * audio.SR)
    step = int(float(tuned.get("step_sec", prof.step_sec)) * audio.SR)
    bp = tuned.get("blank_penalty", prof.blank_penalty)
    pregate = float(tuned.get("pregate_threshold", prof.pregate_threshold))
    notch_freqs = tuple(tuned.get("notch_freqs", prof.notch_freqs))

    if bp is not None:
        try:
            rec = asr.get_offline(cfg, model=model, threads=threads, blank_penalty=bp)
        except Exception:
            pass

    print(f"[stt] Subnoise mode: duration={dur:.1f}s, window={tuned.get('win_sec', prof.win_sec)}s, "
          f"step={tuned.get('step_sec', prof.step_sec)}s, HP={low_hz:.1f}Hz, boost={boost_db:.1f}dB, "
          f"notches={list(notch_freqs)}, blank_penalty={bp:.2f}, pregate={pregate:.6f}", file=sys.stderr)

    # 1. Zero-phase Butterworth HPF + Harmonic Notches + Smooth Formant & Consonant Shelf
    hp = audio.butterworth_highpass(samples, cutoff_hz=low_hz, order=4)
    if notch_freqs:
        hp = audio.harmonic_notch_filter(hp, freqs=notch_freqs, bw_hz=6.0)
    if boost_db > 0.0:
        hp = audio.speech_formant_filter(hp, boost_db=boost_db)

    # 2. Multi-model enhancement candidates
    models = []
    try:
        cfg_g = dict(cfg, robust_denoiser="gtcrn_simple.onnx")
        models.append(("gtcrn", audio.denoise(hp, cfg_g)))
    except Exception as e:
        print(f"[stt] Warning: GTCRN denoiser unavailable: {e}", file=sys.stderr)
    try:
        cfg_d4 = dict(cfg, robust_denoiser="dpdfnet4.onnx")
        models.append(("dp4", audio.denoise(hp, cfg_d4)))
    except Exception as e:
        print(f"[stt] Warning: DPDFNet4 denoiser unavailable: {e}", file=sys.stderr)
    models.append(("hp", hp))

    if len(models) == 1:
        print("[stt] Warning: No neural denoisers available; using high-pass only", file=sys.stderr)
    else:
        print(f"[stt] Subnoise: using {len(models)} enhancement models: {[m[0] for m in models]}", file=sys.stderr)

    # 3. Sliding window formant scan with context margins and speech-band pre-gating
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
    escalated_rec = None
    skipped_pregate = 0

    # Progress throttling
    last_progress_time = 0.0
    last_progress_pct = -5

    def extract_core_words(item: dict, left_margin_sec: float, core_dur_sec: float, ext_start_sec: float) -> tuple[list[dict], float, str]:
        tokens = item.get("tokens", [])
        timestamps = item.get("timestamps", [])
        durations = item.get("durations", [])

        # Burst hallucination check: zero-skip ratio > 0.60
        if item.get("zero_skip_ratio", 0.0) > 0.60 and len(tokens) >= 3:
            return [], -10.0, "zero-skip hallucination burst"

        words_with_meta = []
        if tokens and len(timestamps) == len(tokens):
            for tok, ts, dur in zip(tokens, timestamps, durations if len(durations) == len(tokens) else [0.0] * len(tokens)):
                is_word_start = tok.startswith(" ") or tok.startswith(" ") or not words_with_meta
                clean_tok = tok.lstrip(" ").lstrip(" ")
                if not clean_tok:
                    continue
                if is_word_start:
                    words_with_meta.append({"word": clean_tok, "start": float(ts), "end": float(ts + dur)})
                else:
                    words_with_meta[-1]["word"] += clean_tok
                    words_with_meta[-1]["end"] = float(ts + dur)
        else:
            raw_text = item.get("text", "")
            for w in raw_text.split():
                words_with_meta.append({"word": w, "start": left_margin_sec, "end": left_margin_sec + core_dur_sec})

        # Keep words whose centers or starts fall inside the target core window
        core_min = left_margin_sec - 0.06
        core_max = left_margin_sec + core_dur_sec + 0.06
        core_words = []
        for w in words_with_meta:
            if core_min <= w["start"] <= core_max:
                core_words.append({
                    "word": w["word"],
                    "global_start": round(ext_start_sec + w["start"], 3),
                    "global_end": round(ext_start_sec + w["end"], 3),
                })

        text = " ".join(w["word"] for w in core_words)
        sc, reason = score_coherence(text, core_dur_sec)
        conf = item.get("confidence", 0.5)
        score = sc * (0.6 + 0.4 * conf)
        return core_words, score, reason

    for idx, (start_idx, end_idx) in enumerate(windows):
        t0 = start_idx / audio.SR
        t1 = end_idx / audio.SR

        # Throttle progress callback: every 5% or 1 second
        if progress_cb:
            pct = int((idx + 1) / total_wins * 100) if total_wins > 0 else 100
            now = time.time()
            if pct - last_progress_pct >= 5 or now - last_progress_time >= 1.0:
                progress_cb(idx + 1, total_wins, t1, dur)
                last_progress_pct = pct
                last_progress_time = now

        # Adaptive speech-band energy pre-gate on core window
        window_raw = hp[start_idx:end_idx]
        sb_rms = audio.speech_band_energy(window_raw)
        if sb_rms < pregate:
            skipped_pregate += 1
            continue

        # Context-extended slice with 50ms outer Tukey edge fades
        ext_start = max(0, start_idx - margin_samples)
        ext_end = min(len(samples), end_idx + margin_samples)
        left_margin_sec = (start_idx - ext_start) / audio.SR
        core_dur_sec = (end_idx - start_idx) / audio.SR
        ext_start_sec = ext_start / audio.SR

        cand_chunks = []
        for _, audio_arr in models:
            c = audio_arr[ext_start:ext_end].copy()
            if ext_start > 0 and len(c) > fade_samples:
                c[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            if ext_end < len(samples) and len(c) > fade_samples:
                c[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            c = audio.noise_bounded_agc(c, noise_floor_rms=prof.noise_floor_rms)
            cand_chunks.append(c)

        cand_names = [name for name, _ in models]

        # Batch-decode all candidates in parallel across streams
        cand_details = asr.transcribe_batch_detailed(rec, cand_chunks)

        best_words = []
        best_score = -1.0
        best_text = ""
        best_model = ""

        for name, item in zip(cand_names, cand_details):
            words, sc, reason = extract_core_words(item, left_margin_sec, core_dur_sec, ext_start_sec)
            txt = " ".join(w["word"] for w in words).strip()
            if txt and sc > best_score:
                best_score = sc
                best_words = words
                best_text = txt
                best_model = name

        # Dynamic two-stage blank penalty escalation:
        if not best_text and sb_rms > pregate * 1.5:
            if escalated_rec is None:
                try:
                    esc_bp = min(1.15, float(bp or 0.6) + 0.35)
                    escalated_rec = asr.get_offline(cfg, model=model, threads=threads, blank_penalty=esc_bp)
                except Exception:
                    escalated_rec = rec
            if escalated_rec is not rec:
                esc_details = asr.transcribe_batch_detailed(escalated_rec, cand_chunks)
                for name, item in zip(cand_names, esc_details):
                    words, sc, reason = extract_core_words(item, left_margin_sec, core_dur_sec, ext_start_sec)
                    txt = " ".join(w["word"] for w in words).strip()
                    if txt and sc > best_score:
                        best_score = sc
                        best_words = words
                        best_text = txt
                        best_model = f"{name}+esc"

        if best_text and best_score > 0.5:
            raw_hits.append((round(t0, 2), round(t1, 2), best_text))
            # Merge winning words into global word stream with fuzzy duplicate check
            for cw in best_words:
                cw_clean = cw["word"].strip(".,?!;:\"'()").lower()
                duplicate = False
                for past_w in reversed(emitted_words[-8:]):
                    if abs(past_w["global_start"] - cw["global_start"]) <= 0.35:
                        past_clean = past_w["word"].strip(".,?!;:\"'()").lower()
                        if cw_clean == past_clean:
                            duplicate = True
                            break
                if not duplicate:
                    emitted_words.append(cw)

            if idx % max(1, total_wins // 10) == 0 or idx == total_wins - 1:
                print(f"[stt] Subnoise window {idx+1}/{total_wins}: '{best_text[:60]}...' "
                      f"(score={best_score:.1f}, model={best_model})", file=sys.stderr)

    if skipped_pregate > 0:
        print(f"[stt] Subnoise pre-gate: skipped {skipped_pregate}/{total_wins} silent/non-vocal windows "
              f"({(skipped_pregate/total_wins)*100:.1f}% CPU saved)", file=sys.stderr)

    # 4. Assembly into pause-delimited segments with precise audio timestamps
    segments = []
    if emitted_words:
        curr_seg = [emitted_words[0]]
        for w in emitted_words[1:]:
            gap = w["global_start"] - curr_seg[-1]["global_end"]
            prev_word = curr_seg[-1]["word"]
            is_punct = prev_word.endswith((".", "!", "?"))
            if gap > 0.65 or (is_punct and gap > 0.25):
                seg_text = " ".join(x["word"] for x in curr_seg).strip()
                if seg_text:
                    segments.append((round(curr_seg[0]["global_start"], 2), round(curr_seg[-1]["global_end"], 2), seg_text))
                curr_seg = [w]
            else:
                curr_seg.append(w)
        if curr_seg:
            seg_text = " ".join(x["word"] for x in curr_seg).strip()
            if seg_text:
                segments.append((round(curr_seg[0]["global_start"], 2), round(curr_seg[-1]["global_end"], 2), seg_text))

    if segments:
        print(f"[stt] Subnoise: assembled {len(segments)} pause-delimited segments from {len(emitted_words)} words", file=sys.stderr)
        return segments

    # Fallback to window-level deduplication if word timestamps were unavailable
    if raw_hits:
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
            is_subset = False
            for i in range(len(prev_words) - len(curr_words) + 1):
                if prev_words[i:i+len(curr_words)] == curr_words:
                    is_subset = True
                    break
            if is_subset:
                continue
            is_superset = False
            for i in range(len(curr_words) - len(prev_words) + 1):
                if curr_words[i:i+len(prev_words)] == prev_words:
                    is_superset = True
                    break
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
        return deduped

    print("[stt] Subnoise: no coherent speech detected in any window", file=sys.stderr)
    print("[stt] Subnoise fallback: attempting direct decode of full HP-filtered audio...", file=sys.stderr)
    txts = asr.transcribe(rec, [hp])
    txt = txts[0].strip() if txts else ""
    if txt:
        print(f"[stt] Subnoise fallback success: '{txt[:80]}...'", file=sys.stderr)
        return [(0.0, dur, txt)]
    return []


def transcribe(rec, samples: np.ndarray, cfg: dict, level: int, noise=None, mode: str = 'lead', pre: bool = False, subnoise: bool = False, params: dict | None = None, progress_cb=None, model: str | None = None, threads: int | None = None, no_fallback: bool = False) -> list[tuple[float, float, str]]:
    """Return [(start, end, text)] for the whole signal at the given aggressiveness.
    
    Implements progressive fallback: if requested level finds nothing, tries higher levels up to 5.
    For level 0 (off), also tries levels 1-5 if VAD finds no segments.
    """
    if subnoise:
        return subnoise_transcribe(rec, samples, cfg, params=params, progress_cb=progress_cb, model=model, threads=threads)

    dur_sec = len(samples) / audio.SR
    
    # Helper function to attempt transcription at a specific level
    def try_level(lvl: int, fallback_from: int | None = None) -> list[tuple[float, float, str]]:
        """Try transcription at a specific level, with full decode fallback."""
        if fallback_from is not None:
            print(f"[stt] Fallback: level {fallback_from} found no speech, trying level {lvl}...", file=sys.stderr)
        elif lvl == 0:
            pass  # no message for initial level 0
        else:
            print(f"[stt] Using aggressiveness level {lvl}", file=sys.stderr)

        if lvl <= 0:
            segs = list(audio.segments(samples, cfg))
            found = [(s, e, t) for (s, e, _), t in zip(segs, asr.transcribe(rec, [g[2] for g in segs])) if t]
            # If VAD found nothing, try decoding the whole clip directly (bypass VAD entirely)
            if not found:
                print(f"[stt] Level 0 VAD found no segments; attempting direct decode of full audio...", file=sys.stderr)
                t = asr.transcribe(rec, [samples])[0]
                if t:
                    found = [(0.0, dur_sec, t)]
            return found

        den = None
        results = []
        for v in _variants(lvl, cfg):
            x = audio.normalize(samples) if v.get("norm") else samples
            if v["denoise"]:
                if den is None:
                    g = pregain(samples) if pre else 1.0
                    samples_g = samples * g
                    nz = audio.NoiseProfile(noise.clip * g) if (noise is not None and g != 1.0) else noise
                    src = nz.subtract(samples_g) if (noise is not None and mode in ('sub', 'both')) else samples_g
                    den = audio.denoise(src, cfg, nz if mode in ('lead', 'both') else None)
                x = audio.normalize(den + v["mix"] * samples_g)
            segs = list(audio.segments(x, dict(cfg, **v["vad"])))
            texts = asr.transcribe(rec, [g[2] for g in segs]) if segs else []
            found = [(s, e, t) for (s, e, _), t in zip(segs, texts) if t]
            # Always try full decode fallback for noisy levels if VAD found nothing
            if not found:
                print(f"[stt] Level {lvl} VAD found no segments; attempting direct decode of full audio...", file=sys.stderr)
                t = asr.transcribe(rec, [x])[0]
                if t:
                    found = [(0.0, len(x) / audio.SR, t)]
            results.append(found)
        # Noise makes the recognizer stop early far more often than it hallucinate extra words,
        # so prefer the longer transcript, ignoring any that exceed a plausible speech rate.
        cap = 4.5 * len(samples) / audio.SR
        ok = [r for r in results if sum(len(t.split()) for *_, t in r) <= cap] or results
        return max(ok, key=lambda r: sum(len(t.split()) for *_, t in r))

    # Try the requested level first
    best = try_level(level)

    # Progressive fallback: if nothing found, try higher levels up to 5
    if not no_fallback and not best and level < 5:
        for fallback_level in range(level + 1, 6):
            best = try_level(fallback_level, fallback_from=level)
            if best:
                break

    # Auto-fallback: if standard levels (including level 5 fallback) find no speech, try subnoise scan
    if not best and not no_fallback:
        print(f"[stt] All standard levels failed; attempting subnoise scan...", file=sys.stderr)
        return subnoise_transcribe(rec, samples, cfg, params=params, progress_cb=progress_cb, model=model, threads=threads)

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
) -> list[tuple[float, float, str, str | None]]:
    """Transcribe audio with multilingual model, optional translation.
    
    Returns list of (start, end, source_text, translated_text_or_None).
    """
    from . import asr, translation
    
    dur_sec = len(samples) / audio.SR
    
    # Get multilingual recognizer
    try:
        multi_rec = asr.get_multilingual_offline(cfg)
    except Exception as e:
        print(f"[stt] Multilingual model not available: {e}", file=sys.stderr)
        # Fallback to standard transcribe
        triples = transcribe(rec, samples, cfg, level=level)
        return [(s, e, t, None) for s, e, t in triples]
    
    # Detect language if auto
    detected_lang = source_lang
    if source_lang == "auto":
        print(f"[stt] Auto-detecting language...", file=sys.stderr)
        detected_lang, conf = asr.detect_language_from_audio(
            multi_rec, samples, 
            min_duration=cfg.get("lang_detect_min_duration", 2.0)
        )
        print(f"[stt] Detected language: {detected_lang} (confidence: {conf:.2f})", file=sys.stderr)
        
        # Check confidence threshold
        if conf < cfg.get("lang_detect_confidence_threshold", 0.7):
            print(f"[stt] Low confidence ({conf:.2f}), defaulting to English", file=sys.stderr)
            detected_lang = "en"
    
    # Validate language
    valid_langs = {"en", "es", "pt"}
    if detected_lang not in valid_langs:
        print(f"[stt] Unsupported language {detected_lang}, defaulting to English", file=sys.stderr)
        detected_lang = "en"
    
    # Transcribe with VAD segmentation
    segs = list(audio.segments(samples, cfg))
    if not segs:
        # Try direct decode
        texts = asr.transcribe(multi_rec, [samples])
        text = texts[0] if texts else ""
        if text:
            result = [(0.0, dur_sec, text, None)]
        else:
            return []
    else:
        # Transcribe each segment
        chunk_samples = [g[2] for g in segs]
        texts = asr.transcribe(multi_rec, chunk_samples)
        result = [(s, e, t, None) for (s, e, _), t in zip(segs, texts) if t]
    
    # Translate if requested
    if translate and target_lang and target_lang != detected_lang:
        print(f"[stt] Translating {detected_lang} -> {target_lang}...", file=sys.stderr)
        trans_model_dir = Path(cfg["models_dir"]) / cfg.get("translation_model", "nllb-200-distilled-600M")
        
        if translation.is_translation_available(str(trans_model_dir)):
            translated = translation.translate_segments(
                [(s, e, t) for s, e, t, _ in result],
                detected_lang,
                target_lang,
                str(trans_model_dir),
                progress_cb=progress_cb,
            )
            # Merge: (start, end, source_text, translated_text)
            result = [(s, e, src, tgt) for (s, e, src, _), (_, _, _, tgt) in zip(result, translated)]
        else:
            print(f"[stt] Translation model not found at {trans_model_dir}", file=sys.stderr)
            # Add placeholder
            result = [(s, e, t, f"[Translation model not available]") for s, e, t, _ in result]
    
    return result


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
