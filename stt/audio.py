"""Audio helpers: ffmpeg decoding and Silero VAD segmentation."""
import subprocess
from pathlib import Path

import numpy as np
import sherpa_onnx

SR = 16000


def decode(path: str) -> np.ndarray:
    """Decode any ffmpeg-readable file to 16 kHz mono float32."""
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-vn", "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {p.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(p.stdout, dtype=np.float32)


def make_vad(cfg: dict, min_silence: float | None = None):
    c = sherpa_onnx.VadModelConfig()
    c.silero_vad.model = str(Path(cfg["models_dir"]) / cfg["vad_model"])
    c.silero_vad.threshold = cfg["vad_threshold"]
    c.silero_vad.min_silence_duration = min_silence if min_silence is not None else cfg["vad_min_silence"]
    c.silero_vad.min_speech_duration = cfg["vad_min_speech"]
    c.silero_vad.max_speech_duration = cfg["vad_max_speech"]
    c.silero_vad.window_size = 512
    c.sample_rate = SR
    c.num_threads = 1
    return sherpa_onnx.VoiceActivityDetector(c, buffer_size_in_seconds=60)


def segments(samples: np.ndarray, cfg: dict):
    """Yield (start_sec, end_sec, samples) speech segments."""
    vad = make_vad(cfg)
    w = 512

    def drain():
        while not vad.empty():
            s = vad.front
            yield s.start / SR, (s.start + len(s.samples)) / SR, np.asarray(s.samples, dtype=np.float32)
            vad.pop()

    for i in range(0, len(samples), w):
        chunk = samples[i : i + w]
        if len(chunk) < w:
            chunk = np.pad(chunk, (0, w - len(chunk)))
        vad.accept_waveform(chunk)
        yield from drain()
    vad.flush()
    yield from drain()


_denoisers: dict = {}


def denoise(samples: np.ndarray, cfg: dict, noise: "NoiseProfile | None" = None) -> np.ndarray:
    """Neural speech denoiser (DPDFNet), run in 30 s blocks to bound memory.

    With a calibrated `noise` profile, a clip of the real room noise is prepended as context and dropped afterwards.
    """
    key = (cfg["robust_denoiser"], cfg["threads"])
    if key not in _denoisers:
        c = sherpa_onnx.OfflineSpeechDenoiserConfig()
        path = str(Path(cfg["models_dir"]) / cfg["robust_denoiser"])
        if cfg["robust_denoiser"].startswith("gtcrn"):
            c.model.gtcrn.model = path
        else:
            c.model.dpdfnet.model = path
            c.model.dpdfnet.attenuation_limit_db = 100.0
        c.model.num_threads = min(cfg["threads"], 2)
        _denoisers[key] = sherpa_onnx.OfflineSpeechDenoiser(c)
    d = _denoisers[key]
    block = 30 * SR
    parts = []
    lead = noise.clip if noise is not None else np.zeros(0, np.float32)
    for i in range(0, len(samples), block):
        x = samples[i : i + block]
        y = np.asarray(d.run(np.concatenate([lead, x]), SR).samples, dtype=np.float32)[len(lead):]
        parts.append(np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)])
    return np.concatenate(parts) if parts else samples


def normalize(x: np.ndarray, peak: float = 0.7) -> np.ndarray:
    m = float(np.abs(x).max()) if len(x) else 0.0
    return x * (peak / m) if m > 1e-6 else x


class AGC:
    """Slow automatic gain control for the live mic, so quiet speech (whispering) reaches the VAD/recognizer.

    Tracks the noise floor (slow-rising minimum) and a speech envelope that only updates on blocks well above
    the floor. Gain targets `target` RMS, rises slowly, drops fast, and is capped so the *noise floor*
    is never amplified past ~0.02 RMS (otherwise silence would look like speech to the VAD).
    """

    def __init__(self, target: float = 0.06, max_gain: float = 40.0, noise_cap: float = 0.02, noise_floor: float | None = None):
        self.target, self.max_gain, self.noise_cap = target, max_gain, noise_cap
        self.nf = noise_floor  # a calibrated floor skips the slow learning phase
        self.env = 1e-3  # assume quiet until speech is heard; gain ramps up slowly from 1
        self.gain = 1.0 if noise_floor is None else float(min(max_gain, noise_cap / max(noise_floor, 1e-5), 10.0))

    def __call__(self, b: np.ndarray) -> np.ndarray:
        r = float(np.sqrt(np.mean(b * b))) + 1e-7
        self.nf = r if self.nf is None else (r if r < self.nf else self.nf + (r - self.nf) * 0.002)
        self.env = max(r, self.env) if r > 2.5 * self.nf + 1e-4 else self.env
        self.env *= 0.9978  # half-life ~ 10 s without loud speech
        cap = min(self.max_gain, self.noise_cap / max(self.nf, 1e-5))
        want = float(np.clip(self.target / max(self.env, 1e-5), 1.0, max(cap, 1.0)))
        self.gain = want if want < self.gain else self.gain + (want - self.gain) * 0.02
        return np.clip(b * self.gain, -1.0, 1.0)


class NoiseProfile:
    """Room-noise measurement from a few seconds of silence (`stt mic --whisper/--distant` calibration)."""

    WIN, HOP = 512, 128

    def __init__(self, clip: np.ndarray):
        self.clip = clip.astype(np.float32)
        self.rms = float(np.sqrt(np.mean(self.clip**2)))
        S = np.abs(self._stft(self.clip))
        self.mag = S.mean(axis=0)  # mean magnitude per bin

    def _stft(self, x):
        w = np.hanning(self.WIN)
        pad = np.pad(x, (self.WIN, self.WIN))
        n = 1 + (len(pad) - self.WIN) // self.HOP
        idx = np.arange(self.WIN)[None, :] + self.HOP * np.arange(n)[:, None]
        return np.fft.rfft(pad[idx] * w, axis=1)

    def subtract(self, x: np.ndarray, over: float = 1.5, floor: float = 0.1) -> np.ndarray:
        """Spectral subtraction using the measured noise magnitude spectrum."""
        S = self._stft(x)
        mag = np.abs(S)
        g = np.maximum(mag - over * self.mag, floor * mag) / np.maximum(mag, 1e-9)
        frames = np.fft.irfft(S * g, axis=1) * np.hanning(self.WIN)
        out = np.zeros(len(x) + 2 * self.WIN)
        wsum = np.zeros_like(out)
        w2 = np.hanning(self.WIN) ** 2
        for i in range(len(frames)):
            out[i * self.HOP : i * self.HOP + self.WIN] += frames[i]
            wsum[i * self.HOP : i * self.HOP + self.WIN] += w2
        return (out / np.maximum(wsum, 1e-9))[self.WIN : -self.WIN][: len(x)].astype(np.float32)
