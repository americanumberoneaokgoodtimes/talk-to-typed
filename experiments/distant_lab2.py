import sys, subprocess
import numpy as np, soundfile as sf
sys.path.insert(0, "."); from noise_lab import wer
from stt import audio, asr, config, robust
cfg = config.load(); rec = asr.offline(cfg)
X, _ = sf.read("/data/data/com.termux/files/home/stt/captures/whisper.wav", dtype="float32")
def hp(x, f=120, poles=2):
    return np.frombuffer(subprocess.run(["ffmpeg","-v","error","-f","f32le","-ar","16000","-ac","1","-i","-","-af",f"highpass=f={f}:poles={poles},highpass=f={f}:poles={poles}","-f","f32le","-"],
                         input=x.astype(np.float32).tobytes(), capture_output=True).stdout, dtype=np.float32)
H = hp(X)
r = lambda a: float(np.sqrt(np.mean(a**2)))
UTT = [(2.0, 5.3), (6.3, 9.9), (11.8, 15.2), (17.2, 20.7)]
gaps = [(0.0, 1.8), (5.6, 6.2), (10.2, 11.5), (15.4, 16.9), (21.0, 22.3)]
quiet = np.concatenate([H[int(a*16000):int(b*16000)] for a, b in gaps])
print(f"after highpass: noise rms {r(quiet):.5f} (was {r(np.concatenate([X[int(a*16000):int(b*16000)] for a,b in gaps])):.5f}); whisper utt rms {np.mean([r(H[int(a*16000):int(b*16000)]) for a,b in UTT]):.5f}")
def bed(n, seed): return np.resize(np.roll(quiet, int(np.random.default_rng(seed).integers(0, len(quiet)))), n)
def norm_rms(x, t=0.05):
    return np.clip(x * min(t / max(r(x), 1e-7), 200), -1, 1)
def case(u, scale, seed):
    a, b = UTT[u]; h = H[int(a*16000):int(b*16000)]; return (h * scale + bed(len(h), seed)).astype(np.float32)
pad = np.zeros(4800, np.float32)
def dec(x, lvl):
    return " ".join(t for *_, t in robust.transcribe(rec, np.concatenate([pad, x, pad]), cfg, lvl))
REF = "Don't put any random words in the book."
if __name__ == "__main__":
    for scale in (1.0, 0.5, 0.25, 0.125):
        rows = {"raw rumble L3": [], "hp+norm L0": [], "hp+norm L2": [], "hp+norm L3": []}
        for u in range(4):
            sp = case(u, scale, u)
            raw_rumble = (X[int(UTT[u][0]*16000):int(UTT[u][1]*16000)] - H[int(UTT[u][0]*16000):int(UTT[u][1]*16000)]) + sp  # rumble put back
            rows["raw rumble L3"].append(wer(dec(raw_rumble, 3), REF))
            n = norm_rms(sp)
            for k, l in (("hp+norm L0", 0), ("hp+norm L2", 2), ("hp+norm L3", 3)): rows[k].append(wer(dec(n, l), REF))
        print(f"speech x{scale:<5} (rms {np.mean([r(H[int(a*16000):int(b*16000)])*scale for a,b in UTT]):.5f}) " + "  ".join(f"{k} {np.mean(v):.2f}" for k, v in rows.items()), flush=True)
