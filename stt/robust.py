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


def transcribe(rec, samples: np.ndarray, cfg: dict, level: int, noise=None, mode: str = 'lead', pre: bool = False) -> list[tuple[float, float, str]]:
    """Return [(start, end, text)] for the whole signal at the given aggressiveness."""
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
    return max(ok, key=lambda r: sum(len(t.split()) for *_, t in r))
