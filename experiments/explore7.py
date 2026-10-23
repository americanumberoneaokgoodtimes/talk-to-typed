import sys, subprocess
import numpy as np
sys.path.insert(0, ".")
from noise_lab import *

def mp3rt(x):
    e = subprocess.run(["ffmpeg","-v","error","-f","f32le","-ar","16000","-ac","1","-i","-","-f","mp3","-"], input=x.tobytes(), capture_output=True).stdout
    d = subprocess.run(["ffmpeg","-v","error","-f","mp3","-i","-","-f","f32le","-ac","1","-ar","16000","-"], input=e, capture_output=True).stdout
    return np.frombuffer(d, dtype=np.float32)

if __name__ == "__main__":
    cfg = config.load(); rec = asr.offline(cfg); clean = audio.decode(SPEECH)
    grid = [("white", 0), ("white", -3), ("white", -5), ("pink", -3), ("pink", -5), ("babble", 5), ("babble", 2)]
    enh = {}
    for kind, snr in grid:
        for seed in (0, 1, 2):
            enh[(kind, snr, seed)] = audio.enhance(mp3rt(mix(clean, kind, snr, seed)), cfg)
    print("cached", flush=True)
    for mx in (1.5, 2.5, 3.5, 5.0, 20.0):
        for mix_a in (0.3,):
            per = {}
            for (kind, snr, seed), y in enh.items():
                c = dict(audio.robust_cfg(cfg), vad_max_speech=mx)
                segs = list(audio.segments(y, c))
                t = " ".join(asr.transcribe(rec, [g[2] for g in segs])) if segs else ""
                per.setdefault((kind, snr), []).append(wer(t))
            r = {f"{k[0][:2]}{k[1]}": round(float(np.mean(v)), 2) for k, v in per.items()}
            print(f"max_speech={mx} mean {np.mean(list(r.values())):.3f}", r, flush=True)
