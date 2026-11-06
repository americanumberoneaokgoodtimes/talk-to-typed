#!/usr/bin/env python3
"""Download test audio files for multilingual testing from Common Voice dataset."""
import os
import sys
import subprocess
import urllib.request
from pathlib import Path

# Test audio URLs (using Common Voice samples - these are sample URLs, replace with actual)
# For now, we'll use the existing test_wavs and create a script that can be extended

TEST_AUDIO_DIR = Path(__file__).parent.parent / "stt" / "test_audio"
TEST_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# We'll use the existing model test files and also download some samples
# Common Voice dataset: https://commonvoice.mozilla.org/en/datasets

def download_file(url, dest):
    """Download a file with progress."""
    try:
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        print(f"  OK: {dest}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

def main():
    print("=" * 60)
    print("Downloading multilingual test audio")
    print("=" * 60)
    
    # For now, copy existing test files to test_audio dir for easy access
    model_test_dir = Path.home() / "AppData" / "Local" / "stt" / "models" / "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8" / "test_wavs"
    
    if model_test_dir.exists():
        for lang in ["en", "de", "es", "fr"]:
            src = model_test_dir / f"{lang}.wav"
            if src.exists():
                dst = TEST_AUDIO_DIR / f"{lang}_ref.wav"
                import shutil
                shutil.copy2(src, dst)
                print(f"Copied {lang} reference: {dst}")
    
    # TODO: Add actual Common Voice downloads for Spanish and Portuguese
    # Common Voice download requires accepting terms, so we'll provide instructions
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("""
For Spanish and Portuguese test audio, download from Common Voice:

1. Go to https://commonvoice.mozilla.org/en/datasets
2. Accept terms and download the dataset for:
   - Spanish (es): cv-corpus-XX.0-YYYYMMDD-es.tar.gz
   - Portuguese (pt): cv-corpus-XX.0-YYYYMMDD-pt.tar.gz
3. Extract and find short clips in the 'clips' directory
4. Copy a few clips to this directory as:
   - {TEST_AUDIO_DIR}/es_sample.wav
   - {TEST_AUDIO_DIR}/pt_sample.wav

Alternatively, use these public domain samples:
- Spanish: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/test_wavs_es.zip
- Portuguese: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/test_wavs_pt.zip

For now, the existing test_wavs/es.wav and /fr.wav can be used for testing.
The Spanish model test file exists in the sherpa-onnx model directory if available.
""")
    
    print(f"\nTest audio directory: {TEST_AUDIO_DIR}")
    print("Files in directory:")
    for f in sorted(TEST_AUDIO_DIR.iterdir()):
        if f.is_file():
            print(f"  {f.name} ({f.stat().st_size} bytes)")

if __name__ == "__main__":
    main()