import sys, itertools
import numpy as np
sys.path.insert(0, ".")
from noise_lab import *
from explore_lib import *
from explore2 import denoiser, run_dn
from explore3 import fit
d8 = denoiser("dpdfnet8")
grid = [("white", 0), ("white", -3), ("white", -5), ("pink", -3), ("pink", -5), ("babble", 5), ("babble", 2)]
cache = {}
for kind, snr in grid:
    for seed in (0, 1, 2):
        x = mix(clean, kind, snr, seed)
        cache[(kind, snr, seed)] = (x, fit(run_dn(d8, x)[0], len(x)))
print("enhanced cached", flush=True)

def run(alpha, thr, minsil, minsp, seg=True):
    per = {}
    for (kind, snr, seed), (x, y) in cache.items():
        s = norm(y + alpha * x)
        if seg:
            c = dict(cfg, vad_threshold=thr, vad_min_silence=minsil, vad_min_speech=minsp)
            segs = list(audio.segments(s, c))
            t = " ".join(asr.transcribe(rec, [g[2] for g in segs])) if segs else ""
        else:
            t = dec(s)
        per.setdefault((kind, snr), []).append(wer(t))
    return {k: np.mean(v) for k, v in per.items()}

print("whole-file reference:")
for a in (0.15, 0.3, 0.45):
    r = run(a, 0, 0, 0, seg=False); print(f" a={a} mean {np.mean(list(r.values())):.3f}", {f"{k[0][:2]}{k[1]}": round(v, 2) for k, v in r.items()}, flush=True)
print("with VAD:")
for a, thr, ms in itertools.product((0.3,), (0.05, 0.1, 0.2, 0.35), (0.5, 1.0)):
    r = run(a, thr, ms, 0.1); print(f" a={a} thr={thr} minsil={ms} mean {np.mean(list(r.values())):.3f}", {f"{k[0][:2]}{k[1]}": round(v, 2) for k, v in r.items()}, flush=True)
