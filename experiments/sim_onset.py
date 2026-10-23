import sys, types
import numpy as np
from stt import audio, config, mic
base = config.load()
x = audio.decode("/data/data/com.termux/files/home/stt/test_speech.mp3")
rng = np.random.default_rng(0)
def variant(name):
    y = x.copy()
    if name == "slow+quiet":      # soft 0.6s fade-in at 35% gain, plus room noise
        y *= 0.35; n = int(0.6*16000); y[:n] *= np.linspace(0, 1, n)**2
    if name == "very quiet": y *= 0.15
    return y + rng.standard_normal(len(y)).astype(np.float32) * 0.003
def run(y, cfg):
    y = np.concatenate([np.zeros(16000, np.float32), y, np.zeros(16000*3, np.float32)]); out = []
    class F:
        def start(self): pass
        def stop(self): pass
        def blocks(self):
            for i in range(0, len(y)-512, 512): yield y[i:i+512]
    mic.run_pulse(cfg, types.SimpleNamespace(model=None, threads=2, partial=False), lambda t,s,e: out.append(t), src=F())
    return " | ".join(out)
for name in ("normal", "slow+quiet", "very quiet"):
    y = variant(name)
    for label, c in (("old(pre .38, thr .5, min .25)", dict(base, mic_preroll=0.375, mic_vad_threshold=0.5, mic_vad_min_speech=0.25)),
                     ("new(pre 1.0, thr .35, min .1)", base)):
        print(f"{name:11s} {label:30s} -> {run(y, c)[:100]!r}", flush=True)
