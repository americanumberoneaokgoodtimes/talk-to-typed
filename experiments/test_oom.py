import numpy as np
from stt.asr import transcribe, offline
from stt.config import load


def run_oom_smoke_test():
    # Create a short test audio (5 seconds)
    samples = np.sin(2 * np.pi * 440 * np.arange(0, 5 * 16000) / 16000).astype(np.float32)
    cfg = load()
    cfg["offline_model"] = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    cfg["threads"] = 4

    rec = offline(cfg, threads=cfg["threads"])
    texts = transcribe(rec, [samples])

    print("Test successful - transcribe runs without OOM")
    if texts:
        print(f"Transcribed: {texts[0][:80]}...")
    else:
        print("Transcribed: empty")


if __name__ == "__main__":
    run_oom_smoke_test()

