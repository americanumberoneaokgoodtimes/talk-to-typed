"""Aggressiveness levels for noisy audio (`stt file -a N`, `--robust` == level 3).

0  off: stock VAD (threshold from config) + recognizer
1  loose VAD only (threshold 0.2, segments up to 10 s); no denoising, nearly free
2  + neural denoise, 50% original blended back, VAD 0.15
3  denoise, 30% blend, VAD 0.1, 5 s segments, min speech 0.1 s   (tuned on test_speech.mp3 + noise)
4  level 3 plus a second variant (60% blend, 20 s segments); keeps the more complete transcript
5  `--distant`: normalize-only (no denoiser) and level 3, keeps the more complete transcript. Measured best for
   speech at or below the room-noise level, where the denoiser only erases signal
"""
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


def auto_tune_settings(dur_sec: float) -> dict:
    """Deterministically calculate optimal acoustic and decoding settings based on audio duration.

    Assumes audio has a heavy noise floor with faint/distant vocal formants.
    """
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
        f"Audio duration is {dur_sec:.1f}s. Assuming heavy acoustic noise floor with speech near/below floor.",
        f"High-pass filter cutoff set to {low_hz:.0f} Hz to strip the >85% low-frequency rumble dominating room/mic audio.",
        f"Speech formant equalization set to +{boost_db:.1f} dB (300-3200 Hz) to lift vowel/consonant harmonics above noise.",
        f"Sliding window length set to {win_sec:.1f}s to preserve sentence context while avoiding long-window gain dilution.",
        f"Stride (step) set to {step_sec:.2f}s ({overlap_pct}% overlap) guaranteeing words crossing window borders are captured without cutoff.",
        f"Transducer blank penalty set to {blank_penalty:.1f} to encourage phoneme emission on weak acoustic cues while blocking repetitive loops.",
        "Multi-model denoising enabled (GTCRN complex spectral mapping + DPDFNet-4 gentle neural attenuation + normalized raw HP)."
    ]

    return {
        "win_sec": win_sec,
        "step_sec": step_sec,
        "blank_penalty": blank_penalty,
        "boost_db": boost_db,
        "low_hz": low_hz,
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
    counts = Counter(words)
    most_common_word, most_common_cnt = counts.most_common(1)[0]
    dom_ratio = most_common_cnt / len(words)

    if len(words) >= 3 and dom_ratio >= 0.5:
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


def subnoise_transcribe(rec, samples: np.ndarray, cfg: dict, params: dict | None = None, progress_cb=None) -> list[tuple[float, float, str]]:
    """Aggressive multi-model denoising and formant scanning for speech near/below the noise floor.

    1. Removes high-energy sub-120 Hz acoustic rumble (common in room mics/phone recordings).
    2. Runs multi-model enhancement passes (GTCRN spectral mapping, DPDFNet4 attenuation, normalized HP).
    3. Scans candidates with overlapping sliding windows to recover buried vocal formants.
    4. Merges and deduplicates candidate transcripts.
    """
    if len(samples) == 0:
        return []

    dur = len(samples) / audio.SR
    tuned = params or auto_tune_settings(dur)
    low_hz = tuned.get("low_hz", 120.0)
    boost_db = tuned.get("boost_db", 3.5)
    win_len = int(tuned.get("win_sec", 5.0) * audio.SR)
    step = int(tuned.get("step_sec", 2.0) * audio.SR)

    # 1. High-pass filter (>low_hz) + Formant boost (300-3200 Hz)
    fft = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(len(samples), 1 / audio.SR)
    fft[freqs < low_hz] = 0.0
    if boost_db > 0.0:
        speech_band = (freqs >= 300.0) & (freqs <= 3200.0)
        fft[speech_band] *= 10.0 ** (boost_db / 20.0)
    hp = np.fft.irfft(fft, n=len(samples)).astype(np.float32)

    # 2. Multi-model enhancement candidates
    models = []
    try:
        cfg_g = dict(cfg, robust_denoiser="gtcrn_simple.onnx")
        models.append(("gtcrn", audio.denoise(hp, cfg_g)))
    except Exception:
        pass
    try:
        cfg_d4 = dict(cfg, robust_denoiser="dpdfnet4.onnx")
        models.append(("dp4", audio.denoise(hp, cfg_d4)))
    except Exception:
        pass
    models.append(("hp", hp))

    # 3. Sliding window formant scan
    if len(samples) <= win_len:
        windows = [(0, len(samples))]
    else:
        windows = [(i, i + win_len) for i in range(0, len(samples) - win_len + 1, step)]
        if windows[-1][1] < len(samples):
            windows.append((len(samples) - win_len, len(samples)))

    norm = lambda c: c * (0.8 / max(float(np.abs(c).max()), 1e-6))
    raw_hits = []
    total_wins = len(windows)

    for idx, (start_idx, end_idx) in enumerate(windows):
        t0 = start_idx / audio.SR
        t1 = end_idx / audio.SR

        if progress_cb:
            progress_cb(idx + 1, total_wins, t1, dur)

        best_text = ""
        best_score = -1.0

        for name, den_audio in models:
            chunk = norm(den_audio[start_idx:end_idx])
            txts = asr.transcribe(rec, [chunk])
            txt = txts[0].strip() if txts else ""
            if txt:
                sc, reason = score_coherence(txt, (end_idx - start_idx) / audio.SR)
                if sc > best_score:
                    best_score = sc
                    best_text = txt

        if best_text and best_score > 0.5:
            raw_hits.append((round(t0, 2), round(t1, 2), best_text))

    if not raw_hits:
        return []

    # 4. Deduplicate overlapping adjacent windows
    deduped = []
    for s, e, text in raw_hits:
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
            words_prev = prev_text.split()
            words_curr = text.split()
            merged = False
            for k in range(min(len(words_prev), len(words_curr)), 0, -1):
                if [w.lower() for w in words_prev[-k:]] == [w.lower() for w in words_curr[:k]]:
                    combined = " ".join(words_prev + words_curr[k:])
                    deduped[-1] = (prev_s, e, combined)
                    merged = True
                    break
            if not merged:
                deduped.append((s, e, text))
        else:
            deduped.append((s, e, text))

    return deduped


def transcribe(rec, samples: np.ndarray, cfg: dict, level: int, noise=None, mode: str = 'lead', pre: bool = False, subnoise: bool = False, params: dict | None = None, progress_cb=None) -> list[tuple[float, float, str]]:
    """Return [(start, end, text)] for the whole signal at the given aggressiveness."""
    if subnoise:
        return subnoise_transcribe(rec, samples, cfg, params=params, progress_cb=progress_cb)

    if level <= 0:
        segs = list(audio.segments(samples, cfg))
        return [(s, e, t) for (s, e, _), t in zip(segs, asr.transcribe(rec, [g[2] for g in segs])) if t]
    den = None
    results = []
    for v in _variants(level, cfg):
        x = audio.normalize(samples) if v.get("norm") else samples
        if v["denoise"]:
            if den is None:
                g = pregain(samples) if pre else 1.0
                samples_g = samples * g
                nz = audio.NoiseProfile(noise.clip * g) if (noise is not None and g != 1.0) else noise
                src = nz.subtract(samples_g) if (noise is not None and mode in ('sub', 'both')) else samples_g
                # the net expects normal-level input; very quiet signals are treated as noise and erased
                den = audio.denoise(src, cfg, nz if mode in ('lead', 'both') else None)
            x = audio.normalize(den + v["mix"] * samples_g)
        segs = list(audio.segments(x, dict(cfg, **v["vad"])))
        texts = asr.transcribe(rec, [g[2] for g in segs]) if segs else []
        found = [(s, e, t) for (s, e, _), t in zip(segs, texts) if t]
        if not found and len(x) < 30 * audio.SR:
            # the inner VAD can reject a clip the caller already knows is speech; decode it whole
            t = asr.transcribe(rec, [x])[0]
            found = [(0.0, len(x) / audio.SR, t)] if t else []
        results.append(found)
    # Noise makes the recognizer stop early far more often than it hallucinate extra words,
    # so prefer the longer transcript, ignoring any that exceed a plausible speech rate.
    cap = 4.5 * len(samples) / audio.SR
    ok = [r for r in results if sum(len(t.split()) for *_, t in r) <= cap] or results
    best = max(ok, key=lambda r: sum(len(t.split()) for *_, t in r))

    # Auto-fallback: if standard levels find no speech and level >= 5, try subnoise scan
    if not best and level >= 5:
        return subnoise_transcribe(rec, samples, cfg, params=params, progress_cb=progress_cb)

    return best
