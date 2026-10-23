import sys, types, time
import numpy as np, soundfile as sf
from stt import audio, config, mic
cfg = config.load()
x, _ = sf.read("/data/data/com.termux/files/home/stt/captures/whisper.wav", dtype="float32")
out = []; t0 = time.time()
class F:
    def start(self): pass
    def stop(self): pass
    def blocks(self):
        for i in range(0, len(x) - 512, 512): yield x[i:i+512]
a = types.SimpleNamespace(model=None, threads=2, partial=False, aggressiveness=None, robust=False, whisper=True, save=None, debug=False)
mic.run_pulse_utterance(cfg, a, lambda t, s, e: out.append(f"[{time.time()-t0:4.1f}s] {t}"), src=F())
print("\n".join(out))
