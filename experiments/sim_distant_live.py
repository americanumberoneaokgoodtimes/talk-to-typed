import sys, types, time
sys.path.insert(0, ".")
from distant_lab3 import *
from stt import mic
import numpy as np
scale = float(sys.argv[1]); mode = sys.argv[2]
parts = [bed(int(2.5*16000), 99)]
for u in range(4): parts += [case(u, scale, u + 5), bed(int(1.2*16000), 50 + u)]
x = np.concatenate(parts).astype(np.float32)
out = []
class F:
    def start(self): pass
    def stop(self): pass
    def blocks(self):
        for i in range(0, len(x) - 512, 512): yield x[i:i+512]
a = types.SimpleNamespace(model=None, threads=2, partial=False, aggressiveness=None, robust=False, whisper=(mode=="whisper"), distant=(mode=="distant"), save=None, debug=False, no_calibrate=False)
mic.run_pulse_utterance(cfg, a, lambda t, s, e: out.append(t), src=F())
print(f"x{scale} {mode}: {len(out)} utterances, WER per:", [round(wer(t, REF), 2) for t in out])
