"""Audio helpers: ffmpeg decoding and Silero VAD segmentation."""
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import sherpa_onnx

SR = 16000


def decode(path: str) -> np.ndarray | None:
    """Decode any ffmpeg-readable file to 16 kHz mono float32.
    
    Returns None if the file is empty/corrupt and cannot be decoded.
    """
    import struct
    ffmpeg_bin = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe") or "ffmpeg"
    try:
        p = subprocess.run(
            [ffmpeg_bin, "-v", "error", "-nostdin", "-i", path, "-vn", "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
            capture_output=True,
        )
    except FileNotFoundError:
        raise RuntimeError("ffmpeg binary not found on PATH. Please install ffmpeg.")
    if p.returncode != 0:
        stderr_txt = p.stderr.decode(errors='replace').strip()
        # Check for empty/corrupt file markers
        if "Invalid data" in stderr_txt or "0 bytes" in stderr_txt.lower() or p.stdout == b"":
            print(f"[stt] warning: skipping corrupt/empty file: {path}", file=sys.stderr)
            return None
        raise RuntimeError(f"ffmpeg failed on {path}: {stderr_txt}")
    data = np.frombuffer(p.stdout, dtype=np.float32)
    if len(data) == 0:
        print(f"[stt] warning: skipping empty file: {path}", file=sys.stderr)
        return None
    return data


def make_vad(cfg: dict, min_silence: float | None = None):
    c = sherpa_onnx.VadModelConfig()
    c.silero_vad.model = str(Path(cfg["models_dir"]) / cfg["vad_model"])
    c.silero_vad.threshold = cfg["vad_threshold"]
    c.silero_vad.min_silence_duration = min_silence if min_silence is not None else cfg["vad_min_silence"]
    c.silero_vad.min_speech_duration = cfg["vad_min_speech"]
    c.silero_vad.max_speech_duration = cfg["vad_max_speech"]
    c.silero_vad.window_size = cfg.get("vad_window_size", 512)
    c.sample_rate = SR
    c.num_threads = 1
    return sherpa_onnx.VoiceActivityDetector(c, buffer_size_in_seconds=60)


def segments(samples: np.ndarray, cfg: dict):
    """Yield (start_sec, end_sec, samples) speech segments."""
    vad = make_vad(cfg)
    w = cfg.get("vad_window_size", 512)

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
_denoisers_lock = threading.Lock()


def denoise(samples: np.ndarray, cfg: dict, noise: "NoiseProfile | None" = None) -> np.ndarray:
    """Neural speech denoiser (DPDFNet), run in 30 s blocks with overlap to bound memory and avoid boundary artifacts.

    With a calibrated `noise` profile, a clip of the real room noise is prepended as context and dropped afterwards.
    """
    key = (cfg["robust_denoiser"], cfg["threads"])
    with _denoisers_lock:
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
    # 1 s of lookahead context fed to the denoiser so the model can better denoise the
    # end of each block; the lookahead output is discarded (next block will process it properly).
    lookahead = SR
    parts = []
    lead = noise.clip if noise is not None else np.zeros(0, np.float32)

    for i in range(0, len(samples), block):
        end = min(i + block + lookahead, len(samples))
        x = samples[i:end]
        y = np.asarray(d.run(np.concatenate([lead, x]), SR).samples, dtype=np.float32)[len(lead):]
        # Keep only the block-length output; the lookahead improves boundary quality but is
        # reprocessed by the next block, so we don't include it here.
        if end < len(samples):
            y = y[:block]
        parts.append(y)
        lead = np.zeros(0, np.float32)  # noise lead used only for the first block

    out = np.concatenate(parts) if parts else samples
    # Denoiser frame-alignment can produce a slightly shorter output; pad to match input.
    if len(out) < len(samples):
        out = np.pad(out, (0, len(samples) - len(out)))
    return out[:len(samples)]


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
        self.gain = 1.0 if noise_floor is None else float(min(max_gain, noise_cap / max(noise_floor, 1e-5)))

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


def butterworth_highpass(samples: np.ndarray, cutoff_hz: float = 120.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth high-pass filter in frequency domain with no external dependencies."""
    if len(samples) == 0 or cutoff_hz <= 0:
        return samples
    n = len(samples)
    fft = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    freqs[0] = 1e-6
    h = 1.0 / np.sqrt(1.0 + (cutoff_hz / freqs) ** (2 * order))
    h[0] = 0.0
    return np.fft.irfft(fft * h, n=n).astype(np.float32)


def harmonic_notch_filter(samples: np.ndarray, freqs: tuple[float, ...] = (60.0, 120.0, 180.0), bw_hz: float = 6.0) -> np.ndarray:
    """Zero-phase parametric harmonic notch rejection for electrical hum and HVAC tones."""
    if len(samples) == 0 or not freqs:
        return samples
    n = len(samples)
    fft = np.fft.rfft(samples)
    f_axis = np.fft.rfftfreq(n, 1.0 / SR)
    h_notch = np.ones_like(f_axis, dtype=np.float32)
    for f0 in freqs:
        delta = np.abs(f_axis - f0)
        h_notch *= (delta**2) / (delta**2 + (bw_hz / 2.0)**2)
    return np.fft.irfft(fft * h_notch, n=n).astype(np.float32)


def speech_band_energy(samples: np.ndarray, low_hz: float = 300.0, high_hz: float = 3200.0) -> float:
    """Calculate RMS of energy isolated strictly within the human speech formant band (300-3200 Hz)."""
    if len(samples) == 0:
        return 0.0
    n = len(samples)
    fft = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    filtered_fft = np.zeros_like(fft)
    filtered_fft[mask] = fft[mask]
    filtered_samples = np.fft.irfft(filtered_fft, n=n)
    return float(np.sqrt(np.mean(filtered_samples ** 2) + 1e-12))


class AcousticProfile:
    """Physical acoustic analysis results and synthesized processing parameters for an audio signal."""

    def __init__(self, **kwargs):
        self.noise_floor_rms: float = kwargs.get("noise_floor_rms", 1e-4)
        self.peak_rms: float = kwargs.get("peak_rms", 0.05)
        self.snr_db: float = kwargs.get("snr_db", 10.0)
        self.hum_freq: float | None = kwargs.get("hum_freq", None)
        self.rumble_fraction: float = kwargs.get("rumble_fraction", 0.1)
        self.speech_fraction: float = kwargs.get("speech_fraction", 0.3)
        self.hpf_cutoff_hz: float = kwargs.get("hpf_cutoff_hz", 120.0)
        self.notch_freqs: tuple[float, ...] = kwargs.get("notch_freqs", ())
        self.formant_boost_db: float = kwargs.get("formant_boost_db", 3.0)
        self.blank_penalty: float = kwargs.get("blank_penalty", 0.6)
        self.pregate_threshold: float = kwargs.get("pregate_threshold", 1e-4)
        self.win_sec: float = kwargs.get("win_sec", 5.0)
        self.step_sec: float = kwargs.get("step_sec", 1.75)
        self.reasoning: list[str] = kwargs.get("reasoning", [])

    def to_dict(self) -> dict:
        return {
            "noise_floor_rms": round(self.noise_floor_rms, 6),
            "peak_rms": round(self.peak_rms, 4),
            "snr_db": round(self.snr_db, 1),
            "hum_freq": self.hum_freq,
            "rumble_fraction": round(self.rumble_fraction, 3),
            "speech_fraction": round(self.speech_fraction, 3),
            "hpf_cutoff_hz": round(self.hpf_cutoff_hz, 1),
            "notch_freqs": list(self.notch_freqs),
            "formant_boost_db": round(self.formant_boost_db, 1),
            "blank_penalty": round(self.blank_penalty, 2),
            "pregate_threshold": round(self.pregate_threshold, 6),
            "win_sec": round(self.win_sec, 2),
            "step_sec": round(self.step_sec, 2),
            "reasoning": self.reasoning,
        }


def profile_audio(samples: np.ndarray) -> AcousticProfile:
    """Analyze acoustic envelope, noise floor, hum harmonics, and SNR to synthesize dynamic DSP parameters."""
    dur = len(samples) / SR
    if len(samples) == 0:
        return AcousticProfile(
            noise_floor_rms=0.0, peak_rms=0.0, snr_db=0.0, hum_freq=None,
            rumble_fraction=0.0, speech_fraction=0.0, hpf_cutoff_hz=120.0,
            notch_freqs=(), formant_boost_db=0.0, blank_penalty=0.6,
            pregate_threshold=0.0, win_sec=5.0, step_sec=1.75, reasoning=["Empty audio sample"]
        )

    # 1. Quantile tracking of short-time frame energies (50ms frames = 800 samples)
    frame_len = int(0.05 * SR)
    n_frames = len(samples) // frame_len
    if n_frames > 0:
        frames = samples[:n_frames * frame_len].reshape(n_frames, frame_len)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
        noise_floor = float(np.percentile(frame_rms, 15))
        peak_rms = float(np.percentile(frame_rms, 95))
    else:
        noise_floor = float(np.sqrt(np.mean(samples ** 2) + 1e-12))
        peak_rms = noise_floor

    snr_db = float(20.0 * np.log10(max(peak_rms, 1e-6) / max(noise_floor, 1e-6)))

    # 2. Spectral distribution on representative sample (up to first 60 seconds)
    spec_dur = min(len(samples), 60 * SR)
    spec_samples = samples[:spec_dur]
    fft_mag2 = np.abs(np.fft.rfft(spec_samples)) ** 2
    freqs = np.fft.rfftfreq(spec_dur, 1.0 / SR)

    total_energy = float(np.sum(fft_mag2)) + 1e-12
    rumble_energy = float(np.sum(fft_mag2[freqs < 120.0]))
    speech_energy = float(np.sum(fft_mag2[(freqs >= 300.0) & (freqs <= 3200.0)]))

    rumble_frac = rumble_energy / total_energy
    speech_frac = speech_energy / total_energy

    # 3. Power-grid hum detection (50 Hz vs 60 Hz spectral peak prominence)
    hum_freq = None
    notch_freqs: tuple[float, ...] = ()
    reasoning = []

    def peak_prominence(center_f: float, half_bw: float = 2.0, bg_bw: float = 8.0) -> float:
        peak_mask = (freqs >= center_f - half_bw) & (freqs <= center_f + half_bw)
        bg_mask = (freqs >= center_f - bg_bw) & (freqs <= center_f + bg_bw) & ~peak_mask
        if not np.any(peak_mask) or not np.any(bg_mask):
            return 1.0
        p_val = float(np.max(fft_mag2[peak_mask]))
        bg_val = float(np.median(fft_mag2[bg_mask]) + 1e-12)
        return p_val / bg_val

    prom_60 = peak_prominence(60.0)
    prom_50 = peak_prominence(50.0)

    if prom_60 > 4.0 and prom_60 > prom_50 * 1.3:
        hum_freq = 60.0
        notch_freqs = (60.0, 120.0, 180.0)
        reasoning.append(f"Detected 60 Hz electrical grid hum (prominence={prom_60:.1f}x); engaged notches at 60/120/180 Hz.")
    elif prom_50 > 4.0 and prom_50 > prom_60 * 1.3:
        hum_freq = 50.0
        notch_freqs = (50.0, 100.0, 150.0)
        reasoning.append(f"Detected 50 Hz electrical grid hum (prominence={prom_50:.1f}x); engaged notches at 50/100/150 Hz.")
    else:
        reasoning.append("No prominent power-grid hum detected; power-line notch filters disabled to preserve vocal pitch.")

    # 4. Adaptive low-frequency rumble cutoff
    if rumble_frac >= 0.75:
        hpf_cutoff = 130.0
        reasoning.append(f"Heavy low-frequency rumble dominant ({rumble_frac*100:.1f}% energy <120Hz); raised Butterworth HPF to 130 Hz.")
    elif rumble_frac >= 0.40:
        hpf_cutoff = 115.0
        reasoning.append(f"Moderate low-frequency rumble ({rumble_frac*100:.1f}% energy <120Hz); set Butterworth HPF to 115 Hz.")
    elif rumble_frac >= 0.20:
        hpf_cutoff = 95.0
        reasoning.append(f"Mild low-frequency energy ({rumble_frac*100:.1f}%); lowered HPF to 95 Hz to preserve deep male pitch fundamentals.")
    else:
        hpf_cutoff = 80.0
        reasoning.append(f"Low rumble floor ({rumble_frac*100:.1f}%); minimal 80 Hz HPF applied to preserve full vocal harmonics.")

    # 5. Formant boost calculation
    if snr_db >= 18.0 and speech_frac >= 0.35:
        formant_boost = 0.0
        reasoning.append(f"Clean speech detected (SNR={snr_db:.1f}dB, speech={speech_frac*100:.1f}%); formant boost bypassed to avoid distortion.")
    elif snr_db <= 0.0 or speech_frac <= 0.10:
        formant_boost = float(min(5.0, 2.5 + max(0.0, -snr_db) * 0.25))
        reasoning.append(f"Speech near or below noise floor (SNR={snr_db:.1f}dB); formant band boost set to +{formant_boost:.1f} dB.")
    else:
        formant_boost = float(np.clip(3.5 - (snr_db - 5.0) * 0.2, 1.0, 4.0))
        reasoning.append(f"Moderate noise floor (SNR={snr_db:.1f}dB); formant band boosted by +{formant_boost:.1f} dB.")

    # 6. Adaptive transducer blank penalty
    blank_penalty = float(np.clip(0.55 + (8.0 - snr_db) * 0.03, 0.3, 1.05))
    reasoning.append(f"Transducer blank penalty dynamically tuned to {blank_penalty:.2f} based on estimated SNR ({snr_db:.1f} dB).")

    # 7. Dynamic speech-band pre-gate threshold
    # Measure speech band RMS in quiet frames
    if n_frames > 0:
        quiet_indices = np.argsort(frame_rms)[:max(1, n_frames // 5)]
        quiet_samples = frames[quiet_indices].flatten()
        speech_floor = speech_band_energy(quiet_samples)
    else:
        speech_floor = noise_floor * 0.3
    pregate_threshold = float(max(1e-5, speech_floor * 0.25))
    reasoning.append(f"Dynamic speech-band pre-gate threshold synthesized at {pregate_threshold:.6f} (silence gate at 0.25x quiet-band floor).")

    # 8. Adaptive windowing
    if dur <= 25.0:
        win_sec, step_sec = 5.0, 1.25
    elif dur <= 120.0:
        win_sec, step_sec = 5.0, 1.75
    else:
        win_sec, step_sec = 6.0, 2.5

    return AcousticProfile(
        noise_floor_rms=noise_floor,
        peak_rms=peak_rms,
        snr_db=snr_db,
        hum_freq=hum_freq,
        rumble_fraction=rumble_frac,
        speech_fraction=speech_frac,
        hpf_cutoff_hz=hpf_cutoff,
        notch_freqs=notch_freqs,
        formant_boost_db=formant_boost,
        blank_penalty=blank_penalty,
        pregate_threshold=pregate_threshold,
        win_sec=win_sec,
        step_sec=step_sec,
        reasoning=reasoning,
    )
