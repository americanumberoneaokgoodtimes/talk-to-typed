#!/usr/bin/env python3
"""Generate simple audio files for testing STT system."""

import numpy as np
import soundfile as sf
import os

def generate_tone(freq, duration, filename):
    """Generate a sine wave tone."""
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = np.sin(2 * np.pi * freq * t) * 0.3  # quiet volume
    sf.write(filename, tone, sample_rate, subtype='PCM_16')
    print(f"Generated tone {freq} Hz for {duration}s -> {filename}")

def generate_speech_like(filename):
    """Generate a simple speech-like signal with pauses and variations."""
    sample_rate = 16000
    t = np.linspace(0, 5, int(sample_rate * 5), endpoint=False)
    
    # Create some basic speech patterns - pulses with gaps
    speech_segments = []
    for i in range(4):
        # Each segment is 0.5s of 200Hz tone (vowel-like)
        segment_t = np.linspace(0, 0.5, int(sample_rate * 0.5), endpoint=False)
        segment = np.sin(2 * np.pi * 200 * segment_t) * 0.5
        speech_segments.append(segment)
        # Add 0.5s silence between segments
        silence = np.zeros(int(sample_rate * 0.5))
        speech_segments.append(silence)
    
    speech_signal = np.concatenate(speech_segments)
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.05, len(speech_signal))
    speech_signal = speech_signal + noise
    
    sf.write(filename, speech_signal, sample_rate, subtype='PCM_16')
    print(f"Generated speech-like audio -> {filename}")

def main():
    # Create test directory
    test_dir = "C:/Users/o/Desktop/stt/test_audio"
    os.makedirs(test_dir, exist_ok=True)
    
    # Generate different types of audio
    generate_tone(1000, 3, os.path.join(test_dir, "tone_1000Hz.wav"))
    generate_tone(500, 3, os.path.join(test_dir, "tone_500Hz.wav"))
    generate_speech_like(os.path.join(test_dir, "speech_like.wav"))
    
    # Create a simple test file that should be transcribed as "test"
    test_file = os.path.join(test_dir, "test.wav")
    # 1 second of 150Hz tone (0.5s) + 0.5s silence + 0.5s 150Hz tone (0.5s) + 1s silence
    sample_rate = 16000
    t = np.linspace(0, 2, int(sample_rate * 2), endpoint=False)
    part1 = np.sin(2 * np.pi * 150 * t[:int(sample_rate * 1)]) * 0.3
    part2 = np.sin(2 * np.pi * 150 * t[int(sample_rate * 0.5):int(sample_rate * 1.5)]) * 0.3
    test_signal = np.concatenate([part1, part2])
    sf.write(test_file, test_signal, sample_rate, subtype='PCM_16')
    print(f"Generated test file -> {test_file}")

if __name__ == "__main__":
    main()