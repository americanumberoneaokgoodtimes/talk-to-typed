"""Audio helpers: ffmpeg decoding and Silero VAD segmentation."""
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import sherpa_onnx

SR = 16000


def parse_time_spec(spec: str) -> float:
    """Parse a single time specification into seconds as float.

    Supports:
      - Unit-tagged strings: '00h00m07s635z', '7s635z', '1m30s', '500ms', '10s'
        where h=hours, m=minutes, s=seconds, z/ms=milliseconds.
      - Clock formats: '01:23:45.678', '00:07.635', '1:30'
      - Raw numbers: '7.635', '10'
    """
    s = spec.strip().lower()
    if not s:
        raise ValueError("Empty time specification")
    if s.startswith("-"):
        raise ValueError(f"Time value cannot be negative: {spec!r}")

    # 1. Unit-tagged format (e.g. '00h00m07s635z', '7s635z', '1m30s', '10s')
    unit_matches = re.findall(r"(\d+(?:\.\d+)?)\s*([a-z]+)", s)
    if unit_matches:
        leftover = re.sub(r"(\d+(?:\.\d+)?)\s*([a-z]+)", "", s).strip()
        if not leftover:
            total_sec = 0.0
            unit_map = {
                "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
                "m": 60.0, "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
                "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
                "z": 0.001, "ms": 0.001, "msec": 0.001, "millis": 0.001,
            }
            for val_str, u in unit_matches:
                if u in unit_map:
                    total_sec += float(val_str) * unit_map[u]
                else:
                    raise ValueError(f"Unknown time unit '{u}' in '{spec}'")
            return total_sec

    # 2. Clock format: HH:MM:SS.mmm or MM:SS.mmm
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 3:
            h = float(parts[0])
            m = float(parts[1])
            sec = float(parts[2])
            return h * 3600.0 + m * 60.0 + sec
        elif len(parts) == 2:
            m = float(parts[0])
            sec = float(parts[1])
            return m * 60.0 + sec
        else:
            raise ValueError(f"Invalid clock time format '{spec}'")

    # 3. Pure numeric (seconds)
    try:
        return float(s)
    except ValueError:
        raise ValueError(
            f"Could not parse time value '{spec}'. Examples: '00h00m07s635z', '07s635z', '1:30', '10s'"
        )


def parse_time_range(range_str: str) -> tuple[float, float | None]:
    """Parse a time range string into (start_sec, end_sec).

    Supports formats like:
      - '00h00m07s635z-00h00m10s000z'
      - '07s635z-10s'
      - '7s-10s'
      - '00:07.635-00:10.000'
      - '7.635-10.0'
      - '-10s' (from beginning up to 10s)
      - '7s-' (from 7s to end of file)
      - '7s..10s', '7s to 10s'
    """
    s = range_str.strip()
    if not s:
        raise ValueError("Empty time range")

    # Normalize delimiters
    delimiters = [" to ", " TO ", "To ", " - ", "..", ","]
    for d in delimiters:
        if d in s:
            s = s.replace(d, "-")
            break

    # Handle open ranges or dash-separated ranges
    if s.startswith("-"):
        start_sec = 0.0
        end_sec = parse_time_spec(s[1:])
    elif s.endswith("-"):
        start_sec = parse_time_spec(s[:-1])
        end_sec = None
    elif "-" in s:
        parts = s.split("-", 1)
        start_sec = parse_time_spec(parts[0])
        end_sec = parse_time_spec(parts[1])
    else:
        # Single time given e.g. "7s" -> from 7s to end
        start_sec = parse_time_spec(s)
        end_sec = None

    if end_sec is not None and end_sec <= start_sec:
        raise ValueError(
            f"Invalid time range: end time ({end_sec:.3f}s) must be greater than start time ({start_sec:.3f}s)"
        )

    return start_sec, end_sec


def decode(
    path: str,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> np.ndarray | None:
    """Decode any ffmpeg-readable file to 16 kHz mono float32.

    If start_sec or end_sec are provided, only that time range is extracted using ffmpeg.
    Returns None if the file is empty/corrupt and cannot be decoded.
    """
    ffmpeg_bin = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe") or "ffmpeg"
    cmd = [ffmpeg_bin, "-v", "error", "-nostdin"]
    if start_sec is not None and start_sec > 0:
        cmd.extend(["-ss", f"{start_sec:.3f}"])
    cmd.extend(["-i", path, "-vn"])
    if end_sec is not None:
        dur = end_sec - (start_sec or 0.0)
        if dur > 0:
            cmd.extend(["-t", f"{dur:.3f}"])
    cmd.extend(["-f", "f32le", "-ac", "1", "-ar", str(SR), "-"])

    try:
        p = subprocess.run(cmd, capture_output=True)
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
    if data.ndim > 1:
        data = np.mean(data, axis=-1 if data.shape[-1] < data.shape[0] else 0).astype(np.float32)
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
    """Yield (start_sec, end_sec, samples) speech segments.

    If standard Silero VAD detects no speech (e.g. in heavy noise or faint distant speech),
    automatically falls back to harmonic pitch salience VAD to salvage voiced intervals.
    """
    vad = make_vad(cfg)
    w = cfg.get("vad_window_size", 512)

    def drain():
        while not vad.empty():
            s = vad.front
            yield s.start / SR, (s.start + len(s.samples)) / SR, np.asarray(s.samples, dtype=np.float32)
            vad.pop()

    count = 0
    for i in range(0, len(samples), w):
        chunk = samples[i : i + w]
        if len(chunk) < w:
            chunk = np.pad(chunk, (0, w - len(chunk)))
        vad.accept_waveform(chunk)
        for seg in drain():
            count += 1
            yield seg
    vad.flush()
    for seg in drain():
        count += 1
        yield seg

    # Harmonicity Fallback: If Silero VAD detected 0 speech segments in noisy/faint audio,
    # evaluate physical glottal periodicity (70-350 Hz) to recover buried utterances.
    if count == 0 and len(samples) >= int(0.25 * SR):
        pitch_active = harmonic_pitch_vad(samples, sr=SR)
        hop_sec = 0.010
        min_speech_frames = int(0.12 / hop_sec)
        max_silence_gap = int(0.35 / hop_sec)

        in_speech = False
        start_frame = 0
        silence_gap = 0
        intervals = []

        for f_idx, is_act in enumerate(pitch_active):
            if is_act:
                if not in_speech:
                    in_speech = True
                    start_frame = f_idx
                silence_gap = 0
            else:
                if in_speech:
                    silence_gap += 1
                    if silence_gap > max_silence_gap:
                        dur_frames = (f_idx - silence_gap) - start_frame
                        if dur_frames >= min_speech_frames:
                            intervals.append((start_frame, f_idx - silence_gap))
                        in_speech = False
                        silence_gap = 0

        if in_speech:
            dur_frames = len(pitch_active) - start_frame
            if dur_frames >= min_speech_frames:
                intervals.append((start_frame, len(pitch_active)))

        for sf, ef in intervals:
            s_sec = max(0.0, sf * hop_sec - 0.10)
            e_sec = min(len(samples) / SR, ef * hop_sec + 0.15)
            s_idx = int(s_sec * SR)
            e_idx = int(e_sec * SR)
            chunk = samples[s_idx:e_idx]
            if len(chunk) > 0:
                yield s_sec, e_sec, chunk


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
    """Zero-phase Butterworth high-pass filter with symmetric reflection padding to eliminate circular wrap-around."""
    if len(samples) == 0 or cutoff_hz <= 0:
        return samples
    n_orig = len(samples)
    pad_len = min(n_orig, SR // 2)
    x_mean = float(np.mean(samples))
    x_centered = samples - x_mean
    x_padded = np.pad(x_centered, pad_len, mode='reflect')

    n = len(x_padded)
    fft = np.fft.rfft(x_padded)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    freqs[0] = 1e-6
    h = 1.0 / np.sqrt(1.0 + (cutoff_hz / freqs) ** (2 * order))
    h[0] = 0.0
    filtered = np.fft.irfft(fft * h, n=n)
    return (filtered[pad_len : pad_len + n_orig] + x_mean).astype(np.float32)


def harmonic_notch_filter(samples: np.ndarray, freqs: tuple[float, ...] = (60.0, 120.0, 180.0), bw_hz: float = 6.0) -> np.ndarray:
    """Zero-phase parametric harmonic notch rejection with symmetric reflection padding to eliminate ringing."""
    if len(samples) == 0 or not freqs:
        return samples
    n_orig = len(samples)
    pad_len = min(n_orig, SR // 2)
    x_mean = float(np.mean(samples))
    x_centered = samples - x_mean
    x_padded = np.pad(x_centered, pad_len, mode='reflect')

    n = len(x_padded)
    fft = np.fft.rfft(x_padded)
    f_axis = np.fft.rfftfreq(n, 1.0 / SR)
    h_notch = np.ones_like(f_axis, dtype=np.float32)
    for f0 in freqs:
        delta = np.abs(f_axis - f0)
        h_notch *= (delta**2) / (delta**2 + (bw_hz / 2.0)**2)
    filtered = np.fft.irfft(fft * h_notch, n=n)
    return (filtered[pad_len : pad_len + n_orig] + x_mean).astype(np.float32)


def speech_formant_filter(samples: np.ndarray, boost_db: float = 3.0) -> np.ndarray:
    """Frequency-domain formant enhancement with smooth raised-cosine transitions
    and high-frequency consonant shelf to preserve unvoiced fricatives and plosives without sinc ringing."""
    if boost_db <= 0.0 or len(samples) == 0:
        return samples

    n_orig = len(samples)
    pad_len = min(n_orig, SR // 2)
    x_mean = float(np.mean(samples))
    x_demeaned = samples - x_mean
    x_padded = np.pad(x_demeaned, pad_len, mode="reflect")
    n = len(x_padded)

    fft = np.fft.rfft(x_padded)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)

    # Smooth transition bands:
    # 0 dB below 150 Hz
    # Raised cosine ramp from 150 to 400 Hz up to boost_db
    # Flat core formant band 400 to 3200 Hz at boost_db
    # Raised cosine ramp down from 3200 to 4500 Hz to consonant shelf
    # Flat consonant shelf 4500 to 6500 Hz at min(boost_db, 2.5 dB)
    # Raised cosine roll-off 6500 to 7500 Hz down to 0 dB
    # 0 dB above 7500 Hz
    shelf_db = min(boost_db, 2.5)
    gain_db = np.zeros_like(freqs)

    r1 = (freqs >= 150.0) & (freqs < 400.0)
    w1 = (freqs[r1] - 150.0) / (400.0 - 150.0)
    gain_db[r1] = boost_db * 0.5 * (1.0 - np.cos(np.pi * w1))

    r2 = (freqs >= 400.0) & (freqs < 3200.0)
    gain_db[r2] = boost_db

    r3 = (freqs >= 3200.0) & (freqs < 4500.0)
    w3 = (freqs[r3] - 3200.0) / (4500.0 - 3200.0)
    gain_db[r3] = shelf_db + (boost_db - shelf_db) * 0.5 * (1.0 + np.cos(np.pi * w3))

    r4 = (freqs >= 4500.0) & (freqs < 6500.0)
    gain_db[r4] = shelf_db

    r5 = (freqs >= 6500.0) & (freqs < 7500.0)
    w5 = (freqs[r5] - 6500.0) / (7500.0 - 6500.0)
    gain_db[r5] = shelf_db * 0.5 * (1.0 + np.cos(np.pi * w5))

    linear_gain = 10.0 ** (gain_db / 20.0)
    filtered = np.fft.irfft(fft * linear_gain, n=n)
    return (filtered[pad_len : pad_len + n_orig] + x_mean).astype(np.float32)


def speech_band_energy(samples: np.ndarray, low_hz: float = 300.0, high_hz: float = 3200.0) -> float:
    """Calculate RMS of energy strictly within speech formant band via frequency-domain Parseval integration."""
    if len(samples) == 0:
        return 0.0
    n = len(samples)
    fft = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return 0.0
    mag2 = np.abs(fft[mask]) ** 2
    is_dc_or_nyq = (freqs[mask] == 0.0) | (freqs[mask] == SR / 2.0)
    energy = 2.0 * np.sum(mag2[~is_dc_or_nyq]) + np.sum(mag2[is_dc_or_nyq])
    return float(np.sqrt(max(0.0, energy / (float(n) * float(n))) + 1e-12))


def noise_bounded_agc(
    chunk: np.ndarray,
    noise_floor_rms: float,
    target_speech_rms: float = 0.18,
    max_noise_floor_rms: float = 0.025,
    max_gain: float = 25.0,
    min_gain: float = 1.0,
) -> np.ndarray:
    """Acoustic AGC that boosts faint speech while preventing ambient noise from saturating the ASR encoder."""
    if len(chunk) == 0:
        return chunk
    p90 = float(np.percentile(np.abs(chunk), 90))
    if p90 < 1e-6:
        return chunk
    noise_cap = max_noise_floor_rms / max(noise_floor_rms, 1e-5)
    desired_gain = (target_speech_rms * 1.6) / p90
    effective_gain = float(np.clip(desired_gain, min_gain, min(noise_cap, max_gain)))
    scaled = chunk * effective_gain
    peak_val = float(np.max(np.abs(scaled)))
    if peak_val > 0.85:
        mask = np.abs(scaled) > 0.70
        scaled[mask] = np.sign(scaled[mask]) * (0.70 + 0.15 * np.tanh((np.abs(scaled[mask]) - 0.70) / 0.15))
    return scaled.astype(np.float32)


def spectral_crossover_blend(
    denoised: np.ndarray,
    clean_hp: np.ndarray,
    sr: int = SR,
    f_split_low: float = 3200.0,
    f_split_high: float = 4200.0,
    hp_high_weight: float = 0.70,
) -> np.ndarray:
    """Blend neural denoiser's clean low-band (<3.2 kHz) with HP filter's crisp, un-smeared high-band consonants (>4.2 kHz).

    Eliminates neural musical noise and restores fricative articulation (/s/, /sh/, /f/, /t/, /k/)
    using a zero-phase frequency-domain raised-cosine crossover with dynamic hiss modulation.
    """
    if len(denoised) == 0 or len(clean_hp) == 0:
        return denoised

    m = min(len(denoised), len(clean_hp))
    den = denoised[:m]
    hp = clean_hp[:m]

    pad_len = min(m, sr // 2)
    den_pad = np.pad(den, pad_len, mode="reflect")
    hp_pad = np.pad(hp, pad_len, mode="reflect")
    n_fft = len(den_pad)

    fft_den = np.fft.rfft(den_pad)
    fft_hp = np.fft.rfft(hp_pad)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    mask_high = np.zeros_like(freqs, dtype=np.float32)
    ramp_idx = (freqs >= f_split_low) & (freqs < f_split_high)
    w = (freqs[ramp_idx] - f_split_low) / (f_split_high - f_split_low)
    mask_high[ramp_idx] = 0.5 * (1.0 - np.cos(np.pi * w))
    mask_high[freqs >= f_split_high] = 1.0

    # Stationary hiss rejection in high band: check modulation depth in 4-7 kHz
    hiss_mask = (freqs >= 4000.0) & (freqs <= 7000.0)
    effective_hp_weight = hp_high_weight
    if np.any(hiss_mask):
        hp_mag = np.abs(fft_hp[hiss_mask])
        med = float(np.median(hp_mag) + 1e-9)
        p90 = float(np.percentile(hp_mag, 90))
        mod_ratio = p90 / (2.0 * med)
        effective_hp_weight = float(hp_high_weight * np.clip(mod_ratio, 0.35, 1.0))

    gain_hp = mask_high * effective_hp_weight
    gain_den = 1.0 - gain_hp

    fft_fused = (fft_den * gain_den) + (fft_hp * gain_hp)
    fused_padded = np.fft.irfft(fft_fused, n=n_fft)
    fused = fused_padded[pad_len : pad_len + m]
    return fused.astype(np.float32)


def compute_dynamic_pregate_profile(
    hp_samples: np.ndarray,
    sr: int = SR,
    frame_sec: float = 0.25,
    local_win_sec: float = 6.0,
) -> np.ndarray:
    """Compute continuous time-varying noise floor envelope across the audio (one threshold per 0.25s frame)."""
    frame_len = int(frame_sec * sr)
    n_frames = len(hp_samples) // frame_len
    if n_frames == 0:
        return np.array([1e-4], dtype=np.float32)

    frames = hp_samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)

    win_frames = max(4, int(local_win_sec / frame_sec))
    half_win = win_frames // 2
    local_floor = np.empty(n_frames, dtype=np.float32)

    for i in range(n_frames):
        w_start = max(0, i - half_win)
        w_end = min(n_frames, i + half_win + 1)
        local_floor[i] = float(np.percentile(frame_rms[w_start:w_end], 15))

    return local_floor


def evaluate_window_pregate(
    window_samples: np.ndarray,
    local_floor_rms: float,
    sr: int = SR,
) -> bool:
    """Evaluate whether a window should be transcribed using local SNR and Harmonic Formant Contrast (HFC).

    Allows faint whispers during quiet intervals and detects formant peaks even below broadband noise floor.
    """
    if len(window_samples) == 0:
        return False

    win_rms = float(np.sqrt(np.mean(window_samples ** 2) + 1e-12))
    if win_rms >= local_floor_rms * 1.20:
        return True

    fft_mag = np.abs(np.fft.rfft(window_samples))
    freqs = np.fft.rfftfreq(len(window_samples), 1.0 / sr)
    sb_mask = (freqs >= 300.0) & (freqs <= 3400.0)
    if not np.any(sb_mask):
        return False

    sb_mag = fft_mag[sb_mask]
    mean_val = float(np.mean(sb_mag) + 1e-9)
    peak_val = float(np.max(sb_mag))
    hfc = peak_val / mean_val
    return hfc >= 5.0


def detect_micro_bursts(
    window_samples: np.ndarray,
    sr: int = SR,
    sub_sec: float = 0.5,
    par_threshold: float = 2.0,
) -> list[tuple[int, int]]:
    """Detect high-kurtosis sub-intervals within a non-emitting macro window to spawn targeted 2.5s micro-windows.

    Includes anti-impulse filtering (>=150ms duration) to avoid triggering on mic pops.
    """
    sub_len = int(sub_sec * sr)
    n_subs = len(window_samples) // sub_len
    if n_subs < 4:
        return []

    subs = window_samples[: n_subs * sub_len].reshape(n_subs, sub_len)
    sub_rms = np.sqrt(np.mean(subs ** 2, axis=1) + 1e-12)

    base_floor = float(np.median(sub_rms))
    max_idx = int(np.argmax(sub_rms))
    peak_val = sub_rms[max_idx]
    par = peak_val / max(base_floor, 1e-6)

    # Anti-impulse check: ensure adjacent sub-frame also has elevated energy (>= 1.35x baseline)
    has_duration = False
    if max_idx > 0 and sub_rms[max_idx - 1] >= base_floor * 1.35:
        has_duration = True
    elif max_idx < n_subs - 1 and sub_rms[max_idx + 1] >= base_floor * 1.35:
        has_duration = True

    micro_spans = []
    if par >= par_threshold and has_duration:
        center_sample = int((max_idx + 0.5) * sub_len)
        half_micro = int(1.25 * sr)
        micro_start = max(0, center_sample - half_micro)
        micro_end = min(len(window_samples), micro_start + int(2.5 * sr))
        if micro_end - micro_start >= int(2.0 * sr):
            micro_spans.append((micro_start, micro_end))

    return micro_spans


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

    prom_60 = 0.50 * peak_prominence(60.0) + 0.35 * peak_prominence(120.0) + 0.15 * peak_prominence(180.0)
    prom_50 = 0.50 * peak_prominence(50.0) + 0.35 * peak_prominence(100.0) + 0.15 * peak_prominence(150.0)

    if prom_60 > 3.0 and prom_60 > prom_50 * 1.25:
        hum_freq = 60.0
        notch_freqs = (60.0, 120.0, 180.0)
        reasoning.append(f"Detected 60 Hz electrical grid hum (harmonic comb prominence={prom_60:.1f}x); engaged notches at 60/120/180 Hz.")
    elif prom_50 > 3.0 and prom_50 > prom_60 * 1.25:
        hum_freq = 50.0
        notch_freqs = (50.0, 100.0, 150.0)
        reasoning.append(f"Detected 50 Hz electrical grid hum (harmonic comb prominence={prom_50:.1f}x); engaged notches at 50/100/150 Hz.")
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


def adaptive_spectral_gate(
    samples: np.ndarray,
    sr: int = SR,
    win_len: int = 512,
    hop_len: int = 128,
    g_floor_db: float = -20.0,
    alpha_dd: float = 0.96,
) -> np.ndarray:
    """Pure-NumPy Decision-Directed Wiener Filter with Leaky Minimum Statistics Noise Tracking.

    Tracks the stationary noise floor per STFT bin and applies a smooth, musical-noise-free
    gain mask using the Ephraim-Malah decision-directed a priori SNR estimator.
    """
    if len(samples) < win_len:
        return samples.copy().astype(np.float32)

    n_samples = len(samples)
    g_floor = 10.0 ** (g_floor_db / 20.0)

    # 1. Windowing and STFT
    window = np.hanning(win_len).astype(np.float32)
    n_frames = 1 + (n_samples - win_len) // hop_len

    pad_samples = (n_frames - 1) * hop_len + win_len
    if pad_samples < n_samples:
        n_frames += 1
        pad_len = (n_frames - 1) * hop_len + win_len - n_samples
        samples_padded = np.pad(samples, (0, pad_len), mode="reflect")
    else:
        samples_padded = samples[:pad_samples]

    frames_idx = np.arange(win_len)[None, :] + np.arange(n_frames)[:, None] * hop_len
    frames = samples_padded[frames_idx] * window
    stft = np.fft.rfft(frames, axis=1)  # Shape: (n_frames, n_bins)
    mag2 = np.abs(stft) ** 2
    n_bins = stft.shape[1]

    # 2. Minimum Statistics Noise Floor Tracking
    noise_psd = np.mean(mag2[:min(5, n_frames)], axis=0)
    smooth_power = noise_psd.copy()

    prior_s_mag = np.abs(stft[0])
    out_stft = np.zeros_like(stft)

    beta_rise = 0.003
    beta_fall = 0.15
    alpha_p = 0.80

    for m in range(n_frames):
        cur_mag2 = mag2[m]
        smooth_power = alpha_p * smooth_power + (1.0 - alpha_p) * cur_mag2

        # Asymmetric update of noise floor
        mask_rise = smooth_power > noise_psd
        noise_psd[mask_rise] += beta_rise * (smooth_power[mask_rise] - noise_psd[mask_rise])
        noise_psd[~mask_rise] += beta_fall * (smooth_power[~mask_rise] - noise_psd[~mask_rise])
        noise_psd = np.maximum(noise_psd, 1e-12)

        # 3. Decision-Directed A Priori SNR
        post_snr = cur_mag2 / noise_psd
        prior_snr = alpha_dd * ((prior_s_mag ** 2) / noise_psd) + (1.0 - alpha_dd) * np.maximum(post_snr - 1.0, 0.0)

        # 4. Wiener Gain calculation
        gain = prior_snr / (1.0 + prior_snr)

        # Inter-bin smoothing to eliminate isolated musical noise tones
        gain_smooth = np.convolve(gain, [0.20, 0.60, 0.20], mode="same")
        gain_bounded = np.maximum(gain_smooth, g_floor)

        s_mag = gain_bounded * np.abs(stft[m])
        prior_s_mag = s_mag
        out_stft[m] = s_mag * np.exp(1j * np.angle(stft[m]))

    # 5. Overlap-Add Time Domain Synthesis
    synth_frames = np.fft.irfft(out_stft, axis=1) * window
    out_samples = np.zeros(len(samples_padded) + win_len, dtype=np.float32)
    win_sum = np.zeros_like(out_samples)
    w2 = window ** 2

    for m in range(n_frames):
        idx = m * hop_len
        out_samples[idx : idx + win_len] += synth_frames[m]
        win_sum[idx : idx + win_len] += w2

    reconstructed = out_samples / np.maximum(win_sum, 1e-9)
    return reconstructed[:n_samples].astype(np.float32)


def harmonic_pitch_vad(
    samples: np.ndarray,
    sr: int = SR,
    frame_sec: float = 0.025,
    hop_sec: float = 0.010,
    f0_min: float = 70.0,
    f0_max: float = 350.0,
    nccf_threshold: float = 0.35,
    hcr_threshold: float = 2.5,
) -> np.ndarray:
    """Detect faint voiced speech buried in severe noise using NCCF autocorrelation and HPS.

    Returns a boolean array of speech activity per frame (hop_sec resolution).
    """
    frame_len = int(frame_sec * sr)
    hop_len = int(hop_sec * sr)
    n_frames = 1 + (len(samples) - frame_len) // hop_len
    if n_frames <= 0:
        return np.array([False], dtype=bool)

    lag_min = int(sr / f0_max)
    lag_max = int(sr / f0_min)

    speech_active = np.zeros(n_frames, dtype=bool)

    # Pre-emphasis to flatten glottal spectral tilt and accentuate higher harmonics
    hp_in = samples[1:] - 0.95 * samples[:-1]
    hp_in = np.pad(hp_in, (1, 0))

    window = np.hanning(frame_len).astype(np.float32)

    for i in range(n_frames):
        s_idx = i * hop_len
        chunk = hp_in[s_idx : s_idx + frame_len]

        c_rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-12))
        if c_rms < 1e-5:
            continue

        ext_len = frame_len + lag_max
        if s_idx + ext_len > len(hp_in):
            continue
        ext_chunk = hp_in[s_idx : s_idx + ext_len]

        # Normalized cross-correlation
        n_fft = 1024
        fft_c = np.fft.rfft(chunk * window, n=n_fft)
        fft_ext = np.fft.rfft(ext_chunk[:n_fft], n=n_fft)
        corr = np.fft.irfft(fft_ext * np.conj(fft_c), n=n_fft)

        chunk_e = float(np.sum((chunk * window) ** 2))
        lags_corr = corr[lag_min : lag_max + 1]
        nccf_max = float(np.max(lags_corr)) / (chunk_e + 1e-7)

        # Harmonic Product Spectrum
        mag = np.abs(np.fft.rfft(chunk * window, n=512))
        n_pts = len(mag) // 4
        if n_pts > 0:
            hps = mag[:n_pts] * mag[::2][:n_pts] * mag[::3][:n_pts] * mag[::4][:n_pts]
            hcr = float(np.max(hps)) / float(np.median(hps) + 1e-12)
        else:
            hcr = 1.0

        if nccf_max >= nccf_threshold or hcr >= hcr_threshold:
            speech_active[i] = True

    return speech_active


def multiband_formant_agc(
    samples: np.ndarray,
    sr: int = SR,
    target_formant_rms: float = 0.12,
    max_noise_boost_db: float = 6.0,
) -> np.ndarray:
    """Zero-phase 3-band dynamic range equalizer that lifts faint speech formants
    without amplifying sub-rumble or expanding background pause hiss.
    """
    if len(samples) == 0:
        return samples.copy().astype(np.float32)

    n = len(samples)
    pad_len = min(n, sr // 2)
    s_padded = np.pad(samples - np.mean(samples), pad_len, mode="reflect")
    n_fft = len(s_padded)

    fft = np.fft.rfft(s_padded)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # 1. Band Mask Construction (Smooth Raised-Cosine Crossovers)
    # Band 1: 0 - 140 Hz (Rumble)
    # Band 2: 140 - 3400 Hz (Speech Formants)
    # Band 3: 3400 - 7500 Hz (Consonant Fricatives)
    mask_b1 = np.zeros_like(freqs)
    mask_b2 = np.zeros_like(freqs)
    mask_b3 = np.zeros_like(freqs)

    r_low = (freqs >= 100.0) & (freqs < 180.0)
    w_low = (freqs[r_low] - 100.0) / 80.0
    mask_b1[freqs < 100.0] = 1.0
    mask_b1[r_low] = 0.5 * (1.0 + np.cos(np.pi * w_low))

    r_high = (freqs >= 3000.0) & (freqs < 3800.0)
    w_high = (freqs[r_high] - 3000.0) / 800.0
    mask_b2[r_low] = 0.5 * (1.0 - np.cos(np.pi * w_low))
    mask_b2[(freqs >= 180.0) & (freqs < 3000.0)] = 1.0
    mask_b2[r_high] = 0.5 * (1.0 + np.cos(np.pi * w_high))

    mask_b3[r_high] = 0.5 * (1.0 - np.cos(np.pi * w_high))
    mask_b3[(freqs >= 3800.0) & (freqs < 7500.0)] = 1.0

    # 2. Extract Band Waveforms in Time Domain
    sig_b1 = np.fft.irfft(fft * mask_b1, n=n_fft)[pad_len : pad_len + n]
    sig_b2 = np.fft.irfft(fft * mask_b2, n=n_fft)[pad_len : pad_len + n]
    sig_b3 = np.fft.irfft(fft * mask_b3, n=n_fft)[pad_len : pad_len + n]

    # 3. Dynamic Gain Calculation per Band
    # Band 1 (Rumble): strictly attenuate or unity (never gain)
    g1 = 0.25

    # Band 2 (Core Formants): Speech-aware bounded gain
    p90_b2 = float(np.percentile(np.abs(sig_b2), 90))
    floor_b2 = float(np.percentile(np.abs(sig_b2), 15))

    desired_g2 = (target_formant_rms * 1.5) / max(p90_b2, 1e-5)
    noise_cap_g2 = 10.0 ** (max_noise_boost_db / 20.0) * (floor_b2 / max(floor_b2, 1e-5))
    g2 = float(np.clip(desired_g2, 1.0, min(18.0, max(1.0, noise_cap_g2 * 4.0))))

    # Band 3 (Consonants): Gentle consonant articulation lift
    p90_b3 = float(np.percentile(np.abs(sig_b3), 90))
    desired_g3 = (target_formant_rms * 0.8) / max(p90_b3, 1e-5)
    g3 = float(np.clip(desired_g3, 1.0, min(8.0, g2 * 0.75)))

    # 4. Soft Saturation Synthesis
    out_linear = (sig_b1 * g1) + (sig_b2 * g2) + (sig_b3 * g3)

    peak = float(np.max(np.abs(out_linear)))
    if peak > 0.75:
        out_linear = np.tanh(out_linear / 0.75) * 0.75

    return out_linear.astype(np.float32)
