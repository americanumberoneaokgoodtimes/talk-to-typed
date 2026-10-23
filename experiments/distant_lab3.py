import sys, subprocess
import numpy as np, soundfile as sf
sys.path.insert(0, "."); from noise_lab import wer
from stt import audio, asr, config, robust
cfg = config.load(); rec = asr.offline(cfg)
X, _ = sf.read("/data/data/com.termux/files/home/stt/captures/whisper.wav", dtype="float32")
def hp(x, f=120):
    return np.frombuffer(subprocess.run(["ffmpeg","-v","error","-f","f32le","-ar","16000","-ac","1","-i","-","-af",f"highpass=f={f}:poles=2,highpass=f={f}:poles=2","-f","f32le","-"],input=x.astype(np.float32).tobytes(),capture_output=True).stdout,dtype=np.float32)
H = hp(X); r = lambda a: float(np.sqrt(np.mean(a**2)))
UTT = [(2.0, 5.3), (6.3, 9.9), (11.8, 15.2), (17.2, 20.7)]
calib = X[: int(1.8 * 16000)]  # the recording's own quiet start = what calibration would capture
prof = audio.NoiseProfile(calib)
bedsrc = np.concatenate([X[int(a*16000):int(b*16000)] for a, b in ((5.6, 6.2), (10.2, 11.5), (15.4, 16.9), (21.0, 22.3))])
def bed(n, seed): return np.resize(np.roll(bedsrc, int(np.random.default_rng(seed).integers(0, len(bedsrc)))), n)
pad = np.zeros(4800, np.float32); REF = "Don't put any random words in the book."
def case(u, scale, seed):
    a, b = UTT[u]; h = H[int(a*16000):int(b*16000)]; return np.concatenate([pad, h * scale + bed(len(h), seed), pad]).astype(np.float32)
def dec(x, lvl=3, **kw): return " ".join(t for *_, t in robust.transcribe(rec, x, cfg, lvl, **kw))
if __name__ == "__main__":
    print(f"calibration noise rms {prof.rms:.4f}")
    for scale in (1.0, 0.5, 0.25, 0.125):
        res = {}
        for seed in (0, 1):
            for u in range(4):
                x = case(u, scale, seed * 10 + u)
                for name, kw in (("nopre", dict(pre=False)), ("pre", {}), ("pre+lead", dict(noise=prof, mode="lead")), ("pre+sub", dict(noise=prof, mode="sub")), ("pre+both", dict(noise=prof, mode="both"))):
                    res.setdefault(name, []).append(wer(dec(x, 3, **kw), REF))
        print(f"speech x{scale:<5} " + "  ".join(f"{k} {np.mean(v):.2f}" for k, v in res.items()), flush=True)
