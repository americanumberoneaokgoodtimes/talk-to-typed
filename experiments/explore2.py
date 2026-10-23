import sys, time
import numpy as np
sys.path.insert(0, ".")
from noise_lab import *
from explore_lib import *
import sherpa_onnx as so
M = MODELS

def denoiser(kind):
    c = so.OfflineSpeechDenoiserConfig()
    if kind == "gtcrn":
        c.model.gtcrn.model = f"{M}/gtcrn_simple.onnx"
    else:
        c.model.dpdfnet.model = f"{M}/{kind}.onnx"
        c.model.dpdfnet.attenuation_limit_db = 100.0
    c.model.num_threads = 2
    return so.OfflineSpeechDenoiser(c)

def run_dn(d, x):
    r = d.run(x, 16000)
    return np.asarray(r.samples, dtype=np.float32), r.sample_rate

if __name__=='__main__':
    dns = {k: denoiser(k) for k in ("gtcrn", "dpdfnet2", "dpdfnet4", "dpdfnet8")}
    cases = [("white", 0), ("white", -5), ("white", -10), ("pink", -5), ("pink", -10), ("babble", 5), ("babble", 0), ("babble", -5)]
    for kind, snr in cases:
        x = mix(clean, kind, snr)
        print(f"== {kind} {snr} dB   base: WER {wer(baseline(rec, cfg, x)):.2f}")
        for name, d in dns.items():
            t0 = time.time(); y, sr = run_dn(d, x); dt = time.time() - t0
            a = dec(norm(y)); b = vad_path(norm(y), 0.2)
            print(f"   {name:9s} novad WER {wer(a):.2f} | vad.2 WER {wer(b):.2f} | dn {dt:.1f}s  {a[:70]!r}", flush=True)
