import sys, types, itertools
import numpy as np, soundfile as sf
from stt import audio, config, mic
cfg0 = config.load()
x, _ = sf.read(sys.argv[1], dtype="float32")
def run(cfg, mode):
    out = []
    class F:
        def start(self): pass
        def stop(self): pass
        def blocks(self):
            for i in range(0, len(x) - 512, 512): yield x[i:i+512]
    a = types.SimpleNamespace(model=None, threads=2, partial=False, aggressiveness=None, robust=False, whisper=mode=="whisper", distant=mode=="distant", save=None, debug=False, no_calibrate=False)
    mic.run_pulse_utterance(cfg, a, lambda t, s, e: out.append(t), src=F())
    return out
for name, kw in (("zero pad, preroll 1.0", dict(mic_noise_pad=False)), ("noise pad, preroll 1.0", dict(mic_noise_pad=True)),
                 ("noise pad, preroll 2.0", dict(mic_noise_pad=True, mic_preroll=2.0)), ("zero pad, preroll 2.0", dict(mic_noise_pad=False, mic_preroll=2.0))):
    o = run(dict(cfg0, **kw), sys.argv[2])
    good = sum(t.lower().startswith("don't put any random words in the book") for t in o)
    print(f"{name:26s} {len(o)} utts, full-sentence {good}/4:", [t[:28] for t in o], flush=True)
