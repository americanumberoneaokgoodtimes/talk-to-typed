import sys
import numpy as np
sys.path.insert(0, ".")
from noise_lab import *
from explore_lib import *
from explore2 import denoiser, run_dn

d4 = denoiser("dpdfnet4"); d8 = denoiser("dpdfnet8")

def fit(y, n):
    return np.pad(y, (0, max(0, n-len(y))))[:n]

def windows(x, win=3.0, hop=2.0):
    n, h = int(win*SR), int(hop*SR)
    return [x[i:i+n] for i in range(0, max(len(x)-n+h, 1), h) if len(x[i:i+n]) > SR*0.5]

def chunked(x, win, hop=None):
    hop = hop or win
    return " ".join(dec(norm(c)) for c in windows(x, win, hop))

cases = [("white", -5), ("pink", -5), ("babble", 0)]
for kind, snr in cases:
    x = mix(clean, kind, snr)
    y4, _ = run_dn(d4, x); y8, _ = run_dn(d8, x)
    ss = stft_denoise(x, over=2.0)
    print(f"== {kind} {snr}")
    for name, sig in (("raw", x), ("ss", ss), ("dn4", y4), ("dn8", y8),
                      ("dn4+.3raw", fit(y4,len(x)) + 0.3*x), ("dn4+.5raw", fit(y4,len(x)) + 0.5*x), ("ss+dn4", run_dn(d4, norm(ss))[0])):
        row = [f"whole {wer(dec(norm(sig))):.2f}"]
        for w in (2.0, 3.0, 4.0):
            row.append(f"chunk{w:.0f}s {wer(chunked(sig, w)):.2f}")
        row.append(f"ovl3/2 {wer(chunked(sig, 3.0, 1.5)):.2f}")
        print(f"   {name:10s}", " | ".join(row), flush=True)
