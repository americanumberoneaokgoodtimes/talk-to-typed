import sys
import numpy as np
sys.path.insert(0, ".")
from distant_lab3 import *
for scale in (1.0, 0.25, 0.125):
    res = {}
    for seed in (0, 1):
        for u in range(4):
            x = case(u, scale, seed * 10 + u)
            for k, l in (("L1", 1), ("L3", 3), ("L5", 5)): res.setdefault(k, []).append(wer(dec(x, l), REF))
    print(f"speech x{scale:<5} " + "  ".join(f"{k} {np.mean(v):.2f}" for k, v in res.items()), flush=True)
