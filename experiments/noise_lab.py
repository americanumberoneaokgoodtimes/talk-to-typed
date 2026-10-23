"""Dev harness: mix noise into test_speech.mp3 at a given SNR and score transcription (WER)."""
import re
import sys
import numpy as np
import soundfile as sf
from stt import audio, asr, config

REF = "Don't put any random words in the book. Two or three purposeful sentences is good for a magazine for boys."
SPEECH = "/data/data/com.termux/files/home/stt/test_speech.mp3"
MODELS = config.load()["models_dir"]


def words(t):
    return re.sub(r"[^a-z' ]", "", t.lower().replace("-", " ")).split()


def wer(hyp, ref=REF):
    h, r = words(hyp), words(ref)
    d = np.zeros((len(r) + 1, len(h) + 1), int)
    d[:, 0] = range(len(r) + 1)
    d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + (r[i-1] != h[j-1]))
    return d[-1, -1] / len(r)


def noise(kind, n, rng):
    if kind == "white":
        x = rng.standard_normal(n)
    elif kind in ("pink", "brown"):
        f = np.fft.rfft(rng.standard_normal(n))
        k = np.maximum(np.arange(len(f)), 1)
        f /= np.sqrt(k) if kind == "pink" else k
        x = np.fft.irfft(f, n)
    elif kind == "babble":  # overlapping other-language speakers
        x = np.zeros(n)
        for w in ("de", "es", "fr", "en"):
            a, _ = sf.read(f"{MODELS}/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/test_wavs/{w}.wav", dtype="float32")
            a = audio.decode(f"{MODELS}/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/test_wavs/{w}.wav")
            x += np.tile(a, n // len(a) + 1)[rng.integers(0, len(a)):][:n] if len(a) >= n else np.resize(np.roll(a, rng.integers(0, len(a))), n)
    return (x / np.sqrt(np.mean(x**2))).astype(np.float32)


def mix(clean, kind, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    # SNR measured against speech-active RMS (top-energy frames), not the padded file RMS
    fr = clean[: len(clean) // 512 * 512].reshape(-1, 512)
    e = np.sqrt((fr**2).mean(1))
    sp = np.sqrt(np.mean(fr[e > 0.3 * e.max()] ** 2))
    n = noise(kind, len(clean), rng) * sp / (10 ** (snr_db / 20))
    return np.clip(clean + n, -1, 1).astype(np.float32)


def baseline(rec, cfg, x):
    segs = list(audio.segments(x, cfg))
    return " ".join(asr.transcribe(rec, [s[2] for s in segs]))


if __name__ == "__main__":
    cfg = config.load()
    rec = asr.offline(cfg)
    clean = audio.decode(SPEECH)
    print("clean:", baseline(rec, cfg, clean), "| WER", wer(baseline(rec, cfg, clean)))
    for kind in ("white", "pink", "brown", "babble"):
        for snr in (20, 10, 5, 0, -5, -10, -15):
            x = mix(clean, kind, snr)
            t = baseline(rec, cfg, x)
            print(f"{kind:7s} SNR {snr:4d} dB  WER {wer(t):.2f}  {t[:90]!r}", flush=True)
