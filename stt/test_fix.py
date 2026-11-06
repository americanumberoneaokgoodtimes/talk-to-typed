import numpy as np
import time
from stt.asr import transcribe
from stt import asr
from stt import config
from stt import audio

# Create a long test signal (10 seconds of sine wave)
def test_long_audio():
    # Generate a long audio file (10 seconds)
    duration = 10.0  # seconds
    sr = 16000
    samples = np.sin(2 * np.pi * 440 * np.arange(0, duration * sr) / sr)
    
    # Create a config dict
    cfg = config.load()
    cfg["offline_model"] = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"  # Use a model that exists
    cfg["threads"] = 4
    
    # Create a recognizer
    rec = asr.offline(cfg, threads=cfg["threads"])
    
    # Test with small chunks vs large chunks
    print("Testing with 1-second chunks...")
    chunk_size = sr  # 1 second chunks
    chunks = [samples[i:i+chunk_size] for i in range(0, len(samples), chunk_size)]
    
    start = time.time()
    texts = transcribe(rec, chunks)
    duration = time.time() - start
    print(f"Time taken: {duration:.2f}s")
    print(f"Result length: {len(texts)} transcript(s)")
    if texts:
        print(f"First transcript: {texts[0][:100]}...")

if __name__ == "__main__":
    test_long_audio()