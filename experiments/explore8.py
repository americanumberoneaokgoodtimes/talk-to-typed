import sys
import numpy as np
sys.path.insert(0, ".")
from noise_lab import *
from explore7 import mp3rt

cfg = config.load(); rec = asr.offline(cfg); clean = audio.decode(SPEECH)
grid = [("white", 0), ("white", -3), ("white", -5), ("pink", -3), ("pink", -5), ("babble", 5), ("babble", 2)]
variants = [(a, mx) for a in (0.3, 0.6) for mx in (5.0, 20.0)]
def hyp(y, mx):
    c = dict(audio.robust_cfg(cfg), vad_max_speech=mx)
    segs = list(audio.segments(y, c))
    return " ".join(asr.transcribe(rec, [g[2] for g in segs])) if segs else ""
res = {"single(.3,20)": {}, "pick-longest": {}, "oracle": {}}
for kind, snr in grid:
    for seed in (0, 1, 2):
        x = mp3rt(mix(clean, kind, snr, seed)); H = []
        for a, mx in variants:
            y = audio.enhance(x, dict(cfg, robust_mix=a)); H.append(hyp(y, mx))
        dur = len(x) / 16000
        ok = [h for h in H if len(words(h)) <= 4.5 * dur] or H
        res["single(.3,20)"].setdefault((kind, snr), []).append(wer(H[1]))
        res["pick-longest"].setdefault((kind, snr), []).append(wer(max(ok, key=lambda h: len(words(h)))))
        res["oracle"].setdefault((kind, snr), []).append(min(wer(h) for h in H))
for name, per in res.items():
    r = {f"{k[0][:2]}{k[1]}": round(float(np.mean(v)), 2) for k, v in per.items()}
    print(f"{name:14s} mean {np.mean(list(r.values())):.3f}", r, flush=True)
