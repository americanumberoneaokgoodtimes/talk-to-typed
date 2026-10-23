"""Simulate very quiet speech in the user's real room noise, from captures/whisper.wav."""
import numpy as np, soundfile as sf
from stt import audio, asr, config, robust
import sys; sys.path.insert(0, "."); from noise_lab import wer
cfg = config.load()
X, _ = sf.read("/data/data/com.termux/files/home/stt/captures/whisper.wav", dtype="float32")
REF = "Don't put any random words in the book."
UTT = [(2.0, 5.3), (6.3, 9.9), (11.8, 15.2), (17.2, 20.7)]
# noise-only regions (from the recording's silent gaps) tiled into a long room-noise bed
quiet = np.concatenate([X[int(a*16000):int(b*16000)] for a, b in ((0.0, 1.8), (5.6, 6.2), (10.2, 11.5), (15.4, 16.9), (21.0, 22.3))])
def noise_bed(n, seed):
    r = np.random.default_rng(seed); o = int(r.integers(0, len(quiet))); return np.resize(np.roll(quiet, o), n)
G = 15.0
den = audio.denoise(np.clip(X * G, -1, 1), cfg) / G  # ~clean whisper estimate (denoiser needs normal-level input)
def make(utt, scale, seed):
    a, b = UTT[utt]; sp = den[int(a*16000):int(b*16000)] * scale
    sp = sp / 1.0
    return (sp + noise_bed(len(sp), seed)).astype(np.float32)
def speech_rms(utt, scale):
    a, b = UTT[utt]; s = den[int(a*16000):int(b*16000)] * scale; return float(np.sqrt(np.mean(s**2)))
if __name__ == "__main__":
    print("noise rms %.4f" % np.sqrt(np.mean(quiet**2)))
    rec = asr.offline(cfg)
    for scale in (1.0, 0.5, 0.25):
        row = []
        for u in range(4):
            x = np.concatenate([np.zeros(4800, np.float32), make(u, scale, u), np.zeros(4800, np.float32)])
            row.append(f"{wer(' '.join(t for *_, t in robust.transcribe(rec, x, cfg, 3))):.2f}")
        print(f"speech x{scale}: speech rms~{np.mean([speech_rms(u, scale) for u in range(4)]):.4f}  -a3 WER per utt {row}", flush=True)
