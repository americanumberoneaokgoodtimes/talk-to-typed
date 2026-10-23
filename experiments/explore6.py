import sys
import numpy as np
sys.path.insert(0, ".")
from noise_lab import *
from explore_lib import *
from explore2 import run_dn
from explore3 import fit
import sherpa_onnx as so
grid = [("white", 0), ("white", -3), ("white", -5), ("pink", -3), ("pink", -5), ("babble", 5), ("babble", 2)]
for att in (12.0, 25.0, 40.0, 100.0):
    c = so.OfflineSpeechDenoiserConfig(); c.model.dpdfnet.model = f"{MODELS}/dpdfnet8.onnx"; c.model.dpdfnet.attenuation_limit_db = att; c.model.num_threads = 2
    d = so.OfflineSpeechDenoiser(c); per = {}
    for kind, snr in grid:
        for seed in (0, 1, 2):
            x = mix(clean, kind, snr, seed); y = fit(run_dn(d, x)[0], len(x)); s = norm(y + 0.3 * x)
            segs = list(audio.segments(s, dict(cfg, vad_threshold=0.1, vad_min_speech=0.1)))
            per.setdefault((kind, snr), []).append(wer(" ".join(asr.transcribe(rec, [g[2] for g in segs])) if segs else ""))
    r = {f"{k[0][:2]}{k[1]}": round(float(np.mean(v)), 2) for k, v in per.items()}
    print(f"att={att} mean {np.mean(list(r.values())):.3f}", r, flush=True)
