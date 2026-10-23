"""Feed a recording through run_pulse as if it were the live mic (optionally paced in real time)."""
import sys, time, types
import numpy as np
from stt import audio, config, mic
cfg = config.load()
path, pace, threads = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
x = audio.decode(path)
x = np.concatenate([np.zeros(16000, np.float32), x, np.zeros(16000 * 3, np.float32)])

class Fake:
    def start(self): pass
    def stop(self): pass
    def blocks(self):
        for i in range(0, len(x) - 512, 512):
            if pace: time.sleep(512 / 16000 / pace)
            yield x[i:i+512]
t0 = time.time()
def emit(text, s, e): print(f"[{time.time()-t0:5.1f}s] {text}", flush=True)
mic.run_pulse(cfg, types.SimpleNamespace(model=None, threads=threads, partial=False), emit, src=Fake())
