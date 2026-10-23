import sys, subprocess
import numpy as np
sys.path.insert(0, ".")
from noise_lab import *
from stt import audio, asr, config
import copy

cfg = config.load(); rec = asr.offline(cfg); clean = audio.decode(SPEECH)
SR = 16000


def dec(x):
    s = rec.create_stream(); s.accept_waveform(SR, x); rec.decode_stream(s); return s.result.text.strip()


def stft_denoise(x, over=1.5, floor=0.05, win=512, hop=128, prof_pct=10):
    """Spectral subtraction with per-bin noise floor from low-energy percentile of frames."""
    w = np.hanning(win)
    pad = np.pad(x, (win, win))
    n = 1 + (len(pad) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    S = np.fft.rfft(pad[idx] * w, axis=1)
    mag = np.abs(S)
    noise = np.percentile(mag, prof_pct, axis=0) * 1.5 if False else np.percentile(mag, prof_pct, axis=0)
    # percentile of a Rayleigh-ish dist underestimates mean; rescale to approx mean magnitude
    noise = noise * 2.0
    clean_mag = np.maximum(mag - over * noise, floor * mag)
    S2 = S * (clean_mag / np.maximum(mag, 1e-9))
    out = np.zeros(len(pad)); wsum = np.zeros(len(pad))
    frames = np.fft.irfft(S2, axis=1) * w
    for i in range(n):
        out[i*hop:i*hop+win] += frames[i]; wsum[i*hop:i*hop+win] += w**2
    return (out / np.maximum(wsum, 1e-9))[win:-win][:len(x)].astype(np.float32)


def ffmpeg_filter(x, flt):
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", "16000", "-ac", "1", "-i", "-", "-af", flt, "-f", "f32le", "-"],
                       input=x.tobytes(), capture_output=True)
    return np.frombuffer(p.stdout, dtype=np.float32)


def norm(x):
    return x / max(np.abs(x).max(), 1e-6) * 0.7


def vad_path(x, thr, minsil=0.5, minsp=0.25):
    c = dict(cfg, vad_threshold=thr, vad_min_speech=minsp, vad_min_silence=minsil)
    segs = list(audio.segments(x, c))
    return " ".join(asr.transcribe(rec, [s[2] for s in segs])) if segs else ""


