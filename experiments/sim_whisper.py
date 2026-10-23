import sys, types
import numpy as np
from stt import audio, config, mic
base = config.load()
x = audio.decode("/data/data/com.termux/files/home/stt/test_speech.mp3")
rng = np.random.default_rng(1)
def run(y, cfg):
    y = np.concatenate([np.zeros(16000, np.float32), y, np.zeros(16000*3, np.float32)]); out = []
    class F:
        def start(self): pass
        def stop(self): pass
        def blocks(self):
            for i in range(0, len(y)-512, 512): yield y[i:i+512]
    mic.run_pulse(cfg, types.SimpleNamespace(model=None, threads=2, partial=False), lambda t,s,e: out.append(t), src=F())
    return " | ".join(out)
rms = float(np.sqrt(np.mean(x**2)))
for g in (1.0, 0.1, 0.03, 0.01, 0.004):
    y = x * g + rng.standard_normal(len(x)).astype(np.float32) * 0.0015  # quiet-room mic noise
    for agc in (False, True):
        print(f"speech RMS {rms*g:.4f} AGC={agc!s:5s} -> {run(y, dict(base, mic_agc=agc))[:95]!r}", flush=True)
