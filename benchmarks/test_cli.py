#!/usr/bin/env python3
"""Test the stt file CLI command directly."""
import subprocess
import sys
import os

def run_stt_file(args, description=""):
    """Run stt file command with given arguments."""
    cmd = [sys.executable, "-m", "stt.cli", "file"] + args
    print(f"\n{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Description: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        print(f"Exit code: {result.returncode}")
        return result
    except subprocess.TimeoutExpired:
        print("TIMEOUT - command took too long")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def main():
    print("Testing stt file CLI command")
    print("=" * 60)
    
    # Test with the provided English test wav
    en_wav = r'C:\Users\o\AppData\Local\stt\models\sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8\test_wavs\en.wav'
    
    # Test 1: Basic file transcription
    run_stt_file([en_wav], "Basic file transcription (level 0)")
    
    # Test 2: With aggressiveness level 3
    run_stt_file([en_wav, "-a", "3"], "With aggressiveness level 3")
    
    # Test 3: With --robust
    run_stt_file([en_wav, "--robust"], "With --robust (level 3)")
    
    # Test 4: With --subnoise
    run_stt_file([en_wav, "--subnoise"], "With --subnoise")
    
    # Test 5: With --jsonl
    run_stt_file([en_wav, "--jsonl"], "With --jsonl output")
    
    # Test 6: With --srt
    run_stt_file([en_wav, "--srt"], "With --srt output")
    
    # Test 7: With --vtt
    run_stt_file([en_wav, "--vtt"], "With --vtt output")
    
    # Test 8: With --out
    out_file = "test_output.txt"
    run_stt_file([en_wav, "--out", out_file], "With --out file")
    
    if os.path.exists(out_file):
        print(f"\nOutput file content:")
        with open(out_file, 'r') as f:
            print(f.read())
    
    # Test 9: With --verbose
    run_stt_file([en_wav, "-v"], "With --verbose")
    
    # Test 10: With --auto
    run_stt_file([en_wav, "--auto"], "With --auto")
    
    # Test 11: With --lang
    run_stt_file([en_wav, "--lang", "en"], "With --lang en")
    
    # Test 12: With --model list
    run_stt_file(["--model", "list"], "List models")

if __name__ == "__main__":
    main()