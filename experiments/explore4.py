import sys, time
import numpy as np
sys.path.insert(0, ".")
from noise_lab import *
from explore_lib import *
from explore2 import denoiser, run_dn
from explore3 import fit, windows

D = {k: denoiser(k) for k in ("gtcrn", "dpdfnet2", "dpdfnet4", "dpdfnet8")}

def candidates(x):
    out = {"raw": norm(x), "ss": norm(stft_denoise(x, over=2.0))}
    for k, d in D.items():
        y = fit(run_dn(d, x)[0], len(x))
        out[k] = norm(y)
        if k in ("dpdfnet4", "dpdfnet8"):
            out[k + "+.3"] = norm(y + 0.3 * x)
            out[k + "+.5"] = norm(y + 0.5 * x)
    return out

def hyps(cands, chunk=True):
    h = {}
    for k, s in cands.items():
        h[k] = dec(s)
        if chunk:
            h[k + "@4s"] = " ".join(dec(norm(c)) for c in windows(s, 4.0, 4.0))
    return h

def medoid(h):
    items = [(k, t) for k, t in h.items() if words(t)]
    if not items: return ""
    best = min(items, key=lambda kt: np.mean([wer(kt[1], o[1]) if len(words(o[1])) else 1 for o in items if o[0] != kt[0]] or [0]) - 0.001 * len(words(kt[1])))
    return best[1]

if __name__ == "__main__":
    grid = [("white", 0), ("white", -3), ("white", -5), ("pink", -3), ("pink", -5), ("babble", 5), ("babble", 2), ("babble", 0)]
    for kind, snr in grid:
        res = {"base": [], "best-single": [], "medoid": [], "oracle": []}
        singles = {}
        for seed in (0, 1, 2):
            x = mix(clean, kind, snr, seed)
            h = hyps(candidates(x))
            res["base"].append(wer(baseline(rec, cfg, x)))
            res["medoid"].append(wer(medoid(h)))
            w = {k: wer(t) for k, t in h.items()}
            res["oracle"].append(min(w.values()))
            for k, v in w.items(): singles.setdefault(k, []).append(v)
        bs = min(singles, key=lambda k: np.mean(singles[k]))
        res["best-single"] = singles[bs]
        print(f"{kind:7s}{snr:4d} dB  " + "  ".join(f"{k} {np.mean(v):.2f}" for k, v in res.items()) + f"   (best single: {bs})", flush=True)
