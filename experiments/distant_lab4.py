import sys
import numpy as np
sys.path.insert(0, ".")
from distant_lab3 import *
def norm_pk(x, p=0.7): return x * (p / max(np.abs(x).max(), 1e-6))
def loose_dec(x):
    c = dict(cfg, vad_threshold=0.1, vad_min_speech=0.1, vad_max_speech=5.0)
    segs = list(audio.segments(x, c))
    t = " ".join(asr.transcribe(rec, [g[2] for g in segs])) if segs else ""
    return t or asr.transcribe(rec, [x])[0]
V = {
  "norm": lambda x: loose_dec(norm_pk(x)),
  "hp+norm": lambda x: loose_dec(norm_pk(hp(x))),
  "sub1.0+norm": lambda x: loose_dec(norm_pk(prof.subtract(x, over=1.0))),
  "sub2.0+norm": lambda x: loose_dec(norm_pk(prof.subtract(x, over=2.0))),
  "hp+sub2+norm": lambda x: loose_dec(norm_pk(prof.subtract(hp(x), over=2.0))),
  "L3 nopre": lambda x: dec(x, 3, pre=False),
}
for scale in (0.25, 0.125):
    res = {}
    for seed in (0, 1):
        for u in range(4):
            x = case(u, scale, seed * 10 + u)
            for k, f in V.items(): res.setdefault(k, []).append(wer(f(x), REF))
    print(f"speech x{scale:<5} " + "  ".join(f"{k} {np.mean(v):.2f}" for k, v in res.items()), flush=True)
