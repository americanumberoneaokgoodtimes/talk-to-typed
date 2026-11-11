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
    # Automatically suppress mechanical keyboard clicks and sharp impulses before VAD
    do_declick = cfg.get("declick", True)
    thresh = float(cfg.get("declick_threshold", 4.8))
    if do_declick:
        vad_samples, _ = suppress_impulsive_clicks(samples, threshold_sigma=thresh)
    else:
        vad_samples = samples
    vad = make_vad(cfg)
    w = cfg.get("vad_window_size", 512)

    def drain():
        while not vad.empty():
            s = vad.front
            yield s.start / SR, (s.start + len(s.samples)) / SR, np.asarray(s.samples, dtype=np.float32)
            vad.pop()

    emitted_spans: list[tuple[float, float]] = []
    for i in range(0, len(vad_samples), w):
        chunk = vad_samples[i : i + w]
        if len(chunk) < w:
            chunk = np.pad(chunk, (0, w - len(chunk)))
        vad.accept_waveform(chunk)
        for seg in drain():
            emitted_spans.append((seg[0], seg[1]))
            yield seg
    vad.flush()
    for seg in drain():
        emitted_spans.append((seg[0], seg[1]))
        yield seg

    def _pitch_intervals_in_span(span_samples: np.ndarray, offset_sec: float) -> list[tuple[float, float, np.ndarray]]:
        if len(span_samples) < int(0.25 * SR):
            return []
        pitch_active = harmonic_pitch_vad(span_samples, sr=SR)
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

        recovered = []
        for sf, ef in intervals:
            s_local = max(0.0, sf * hop_sec - 0.10)
            e_local = min(len(span_samples) / SR, ef * hop_sec + 0.15)
            s_idx = int(s_local * SR)
            e_idx = int(e_local * SR)
            sub_chunk = span_samples[s_idx:e_idx]
            if len(sub_chunk) > 0:
                recovered.append((offset_sec + s_local, offset_sec + e_local, sub_chunk))
        return recovered

    # Harmonicity Fallback: recover voiced speech when Silero VAD misses either the whole clip
    # or >= 3.0s unsegmented gaps (e.g., when a single loud transient/cough fired Silero elsewhere).
    total_dur = len(vad_samples) / SR
    if not emitted_spans and len(vad_samples) >= int(0.25 * SR):
        for seg in _pitch_intervals_in_span(vad_samples, 0.0):
            yield seg
    elif emitted_spans and float(cfg.get("vad_threshold", 0.5)) <= 0.35 and total_dur >= 4.0:
        cursor = 0.0
        gaps: list[tuple[float, float]] = []
        for gs, ge in sorted(emitted_spans):
            if gs - cursor >= 3.0:
                gaps.append((cursor, gs))
            cursor = max(cursor, ge)
        if total_dur - cursor >= 3.0:
            gaps.append((cursor, total_dur))
        for gs, ge in gaps:
            g_slice = vad_samples[int(gs * SR) : int(ge * SR)]
            for seg in _pitch_intervals_in_span(g_slice, gs):
                yield seg


_denoisers: dict = {}
_denoisers_lock = threading.Lock()


def denoise(
    samples: np.ndarray,
    cfg: dict,
    noise: "NoiseProfile | None" = None,
    attenuation_limit_db: float | None = None,
) -> np.ndarray:
    """Neural speech denoiser (GTCRN / DPDFNet), run in 30 s blocks with causal past pre-roll and crossfade.

    Uses 1.0 s of past audio pre-roll (or calibrated room `noise` on block 0) to warm up causal RNN
    hidden states and 50 ms raised-cosine crossfades across block boundaries to eliminate splice clicks.
    """
    if len(samples) == 0:
        return samples.copy().astype(np.float32)

    atten_db = float(attenuation_limit_db if attenuation_limit_db is not None else cfg.get("dpdfnet_attenuation_limit_db", 100.0))
    key = (str(cfg.get("models_dir", "")), cfg["robust_denoiser"], cfg["threads"], round(atten_db, 1))
    with _denoisers_lock:
        if key not in _denoisers:
            c = sherpa_onnx.OfflineSpeechDenoiserConfig()
            path = str(Path(cfg["models_dir"]) / cfg["robust_denoiser"])
            if cfg["robust_denoiser"].startswith("gtcrn"):
                c.model.gtcrn.model = path
            else:
                c.model.dpdfnet.model = path
                c.model.dpdfnet.attenuation_limit_db = atten_db
            c.model.num_threads = min(cfg["threads"], 2)
            _denoisers[key] = sherpa_onnx.OfflineSpeechDenoiser(c)
        d = _denoisers[key]

    block = 30 * SR
    preroll = SR
    xfade = int(0.05 * SR)
    n_total = len(samples)

    if n_total <= block:
        lead = noise.clip if noise is not None else np.zeros(0, np.float32)
        y = np.asarray(d.run(np.concatenate([lead, samples]), SR).samples, dtype=np.float32)[len(lead):]
        if len(y) < n_total:
            y = np.pad(y, (0, n_total - len(y)))
        return y[:n_total]

    out = np.zeros(n_total, dtype=np.float32)
    prev_end = 0
    for i in range(0, n_total, block):
        end = min(i + block, n_total)
        if i == 0:
            lead = noise.clip if noise is not None else np.zeros(0, np.float32)
            raw_in = np.concatenate([lead, samples[0:end]])
            drop_prefix = len(lead)
            y_full = np.asarray(d.run(raw_in, SR).samples, dtype=np.float32)
            y = y_full[drop_prefix:]
            target_len = end
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            out[0:end] = y[:target_len]
            prev_end = end
        else:
            pre_start = max(0, i - preroll)
            raw_in = samples[pre_start:end]
            y_full = np.asarray(d.run(raw_in, SR).samples, dtype=np.float32)
            need_len = end - pre_start
            if len(y_full) < need_len:
                y_full = np.pad(y_full, (0, need_len - len(y_full)))
            y_full = y_full[:need_len]

            # Crossfade over the last `xfade` samples prior to `i`
            xf_len = min(xfade, i - pre_start, prev_end - pre_start)
            if xf_len > 0:
                xf_start = i - xf_len
                w_in = (0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, xf_len, dtype=np.float32))))
                pre_offset = xf_start - pre_start
                out[xf_start:i] = out[xf_start:i] * (1.0 - w_in) + y_full[pre_offset : pre_offset + xf_len] * w_in

            body_offset = i - pre_start
            out[i:end] = y_full[body_offset : body_offset + (end - i)]
            prev_end = end

    return out


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
    # Do NOT restore DC mean: HPF is designed to remove DC/low-frequency offset.
    # Restoring x_mean would re-introduce the DC bias that biases the denoiser's input.
    return filtered[pad_len : pad_len + n_orig].astype(np.float32)


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
    # Notch filter: preserve DC mean (we only removed hum harmonics, not the DC offset)
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


def soft_declip(samples: np.ndarray, clip_ratio: float = 0.985, max_plateau: int = 48) -> tuple[np.ndarray, int]:
    """Pure-NumPy cubic Hermite declipper that reconstructs hard-saturated waveform peaks before neural denoising.

    Eliminates odd-harmonic square-wave splatter across 300-3500 Hz caused by overdriven cheap microphones.
    """
    if len(samples) < 32:
        return samples.copy().astype(np.float32), 0

    peak = float(np.max(np.abs(samples)))
    if peak < 0.94:
        return samples.copy().astype(np.float32), 0

    thresh = clip_ratio * peak
    clipped = np.abs(samples) >= thresh
    if not np.any(clipped):
        return samples.copy().astype(np.float32), 0

    out = samples.astype(np.float32).copy()
    n = len(out)
    idx = np.where(clipped)[0]
    runs: list[tuple[int, int]] = []
    s_cur = idx[0]
    e_cur = idx[0]
    for k in idx[1:]:
        if k <= e_cur + 1:
            e_cur = k
        else:
            if 2 <= (e_cur - s_cur + 1) <= max_plateau:
                runs.append((s_cur, e_cur))
            s_cur = e_cur = k
    if 2 <= (e_cur - s_cur + 1) <= max_plateau:
        runs.append((s_cur, e_cur))

    if not runs:
        return out, 0

    repaired = 0
    for s, e in runs:
        if s < 2 or e >= n - 2:
            continue
        w = e - s + 1
        y0 = float(out[s - 1])
        y1 = float(out[e + 1])
        m0 = float(out[s - 1] - out[s - 2]) * w
        m1 = float(out[e + 2] - out[e + 1]) * w
        t = np.linspace(0.0, 1.0, w + 2, dtype=np.float32)[1:-1]
        h00 = (2.0 * t**3 - 3.0 * t**2 + 1.0)
        h10 = (t**3 - 2.0 * t**2 + t)
        h01 = (-2.0 * t**3 + 3.0 * t**2)
        h11 = (t**3 - t**2)
        interp = h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1
        out[s : e + 1] = interp
        repaired += 1

    if repaired > 0:
        new_peak = float(np.max(np.abs(out)))
        if new_peak > 0.95:
            out *= (0.92 / new_peak)
    return out.astype(np.float32), repaired


def extract_auto_noise_profile(samples: np.ndarray, sr: int = SR, clip_sec: float = 1.25) -> "NoiseProfile | None":
    """Automatically extract a calibrated NoiseProfile from the quietest sustained non-zero background span."""
    win = int(clip_sec * sr)
    if len(samples) < max(win * 2, int(2.0 * sr)):
        return None
    hop = max(sr // 4, win // 4)
    best_rms = float("inf")
    best_slice = None
    for start in range(0, len(samples) - win + 1, hop):
        sl = samples[start : start + win]
        # Skip codec-muted digital silence spans so we profile true acoustic background floor
        rms = float(np.sqrt(np.mean(sl ** 2)))
        nz_frac = float(np.mean(np.abs(sl) > 1e-6))
        if rms > 2e-5 and nz_frac >= 0.70 and rms < best_rms:
            best_rms = rms
            best_slice = sl
    if best_slice is None:
        return None
    try:
        return NoiseProfile(best_slice.copy())
    except Exception:
        return None


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
    upper_gain = max(min_gain, min(noise_cap, max_gain))
    effective_gain = float(np.clip(desired_gain, min_gain, upper_gain))
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

    Uses true time-domain 16 ms frame modulation + spectral flatness gating in 4-7 kHz so stationary broadband
    hiss is rejected while transient fricative bursts (/s/, /sh/, /f/, /t/, /k/) and structured harmonics pass cleanly.
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

    # Extract high-band time-domain residual to measure true temporal consonant modulation vs stationary hiss
    hp_high_td = np.fft.irfft(fft_hp * mask_high, n=n_fft)[pad_len : pad_len + m].astype(np.float32)
    den_low_td = np.fft.irfft(fft_den * (1.0 - mask_high * hp_high_weight), n=n_fft)[pad_len : pad_len + m].astype(np.float32)
    den_high_td = np.fft.irfft(fft_den * (mask_high * hp_high_weight), n=n_fft)[pad_len : pad_len + m].astype(np.float32)

    hiss_mask = (freqs >= 4000.0) & (freqs <= 7000.0)
    flatness = 1.0
    if np.any(hiss_mask):
        hb_pow = np.abs(fft_hp[hiss_mask]) ** 2 + 1e-12
        geom = float(np.exp(np.mean(np.log(hb_pow))))
        arith = float(np.mean(hb_pow))
        flatness = geom / max(arith, 1e-12)

    f_len = max(64, int(0.016 * sr))
    n_fr = m // f_len
    if n_fr >= 4:
        fr = hp_high_td[: n_fr * f_len].reshape(n_fr, f_len)
        fr_rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
        floor_hi = float(np.percentile(fr_rms, 20))
        p90_hi = float(np.percentile(fr_rms, 90))
        temp_mod = p90_hi / max(floor_hi, 1e-7)

        if flatness < 0.35:
            # Structured harmonic / narrowband high-frequency resonance
            env_gate = np.ones(m, dtype=np.float32)
        elif temp_mod < 1.45:
            # Stationary broadband hiss with minimal consonant burst activity
            env_gate = np.full(m, 0.18, dtype=np.float32)
        else:
            # Time-varying consonant gate: pass bursts above 1.35x floor, attenuate stationary hiss valleys
            fr_snr = np.clip((fr_rms / max(floor_hi, 1e-7) - 1.15) / 1.5, 0.18, 1.0).astype(np.float32)
            fr_snr = np.convolve(fr_snr, np.array([0.25, 0.50, 0.25], dtype=np.float32), mode="same")
            env_gate = np.repeat(fr_snr, f_len)
            if len(env_gate) < m:
                env_gate = np.pad(env_gate, (0, m - len(env_gate)), mode="edge")
    else:
        env_gate = np.full(m, 0.85 if flatness < 0.45 else 0.35, dtype=np.float32)

    fused = den_low_td + den_high_td * (1.0 - env_gate) + (hp_high_td * hp_high_weight) * env_gate
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
    denoised_window: np.ndarray | None = None,
) -> bool:
    """Evaluate whether a window should be transcribed using short-frame emergence, syllabic modulation, and HFC.

    Detects brief (<1s) whispers inside 5s windows and sub-noise talking via 2-8 Hz syllabic envelope modulation.
    """
    if len(window_samples) == 0:
        return False

    win_rms = float(np.sqrt(np.mean(window_samples ** 2) + 1e-12))
    if win_rms >= local_floor_rms * 1.16:
        return True

    # Check short-time 200ms peak emergence so a 0.8s utterance in a 5s window isn't diluted by 4.2s of noise
    sub_len = int(0.20 * sr)
    n_sub = len(window_samples) // sub_len
    if n_sub >= 3:
        sub_rms = np.sqrt(np.mean(window_samples[: n_sub * sub_len].reshape(n_sub, sub_len) ** 2, axis=1) + 1e-12)
        if float(np.percentile(sub_rms, 90)) >= max(local_floor_rms, float(np.percentile(sub_rms, 20))) * 1.28:
            return True

    if denoised_window is not None and len(denoised_window) >= sub_len * 2:
        nd = len(denoised_window) // sub_len
        d_rms = np.sqrt(np.mean(denoised_window[: nd * sub_len].reshape(nd, sub_len) ** 2, axis=1) + 1e-12)
        d_floor = float(np.percentile(d_rms, 20))
        d_peak = float(np.max(d_rms))
        if d_peak >= max(2.5e-4, d_floor * 1.65):
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
    if hfc >= 4.4:
        return True

    # Syllabic 2-8 Hz envelope modulation check for speech buried below broadband stationary noise
    hop = int(0.025 * sr)
    n_env = len(window_samples) // hop
    if n_env >= 16:
        env = np.sqrt(np.mean(window_samples[: n_env * hop].reshape(n_env, hop) ** 2, axis=1) + 1e-12)
        log_env = np.log(env + 1e-7) - float(np.mean(np.log(env + 1e-7)))
        env_fft = np.abs(np.fft.rfft(log_env)) ** 2
        env_freqs = np.fft.rfftfreq(n_env, 0.025)
        syl_mask = (env_freqs >= 2.0) & (env_freqs <= 8.0)
        hi_mask = (env_freqs > 10.0)
        if np.any(syl_mask) and np.any(hi_mask):
            syl_ratio = float(np.mean(env_fft[syl_mask])) / float(np.mean(env_fft[hi_mask]) + 1e-9)
            if syl_ratio >= 3.2:
                return True

    return False


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
        self.raw_noise_floor_rms: float = kwargs.get("raw_noise_floor_rms", self.noise_floor_rms)
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
            "raw_noise_floor_rms": round(self.raw_noise_floor_rms, 6),
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
            noise_floor_rms=0.0, raw_noise_floor_rms=0.0, peak_rms=0.0, snr_db=0.0, hum_freq=None,
            rumble_fraction=0.0, speech_fraction=0.0, hpf_cutoff_hz=120.0,
            notch_freqs=(), formant_boost_db=0.0, blank_penalty=0.6,
            pregate_threshold=0.0, win_sec=5.0, step_sec=1.75, reasoning=["Empty audio sample"]
        )

    # 1. Raw frame energies (50ms frames = 800 samples) for quiet-frame selection
    frame_len = int(0.05 * SR)
    n_frames = len(samples) // frame_len
    if n_frames > 0:
        raw_frames = samples[:n_frames * frame_len].reshape(n_frames, frame_len)
        raw_frame_rms = np.sqrt(np.mean(raw_frames ** 2, axis=1) + 1e-12)
        raw_noise_floor = float(np.percentile(raw_frame_rms, 15))
    else:
        raw_noise_floor = float(np.sqrt(np.mean(samples ** 2) + 1e-12))

    # 2. Spectral distribution on representative sample.
    # For files > 60s, use stratified windows (25%, 50%, 75%) and average spectra so that
    # intro music / ambient noise at the start doesn't miscalibrate filters for the whole file.
    WIN_LEN = 20 * SR  # 20-second analysis window
    if len(samples) > 60 * SR:
        total = len(samples)
        anchors = [int(total * f) for f in (0.25, 0.50, 0.75)]
        windows = [samples[max(0, a - WIN_LEN // 2): a + WIN_LEN // 2] for a in anchors]
        windows = [w for w in windows if len(w) >= SR]  # skip sub-second stubs
        if windows:
            max_w = max(len(w) for w in windows)
            stack = np.vstack([np.pad(w, (0, max_w - len(w))) for w in windows]).astype(np.float32)
            fft_mag2 = np.mean(np.abs(np.fft.rfft(stack, axis=1)) ** 2, axis=0)
            spec_dur = max_w
        else:
            spec_dur = min(len(samples), 60 * SR)
            fft_mag2 = np.abs(np.fft.rfft(samples[:spec_dur])) ** 2
    else:
        spec_dur = len(samples)
        fft_mag2 = np.abs(np.fft.rfft(samples)) ** 2
    freqs = np.fft.rfftfreq(spec_dur, 1.0 / SR)

    total_energy = float(np.sum(fft_mag2)) + 1e-12
    rumble_energy = float(np.sum(fft_mag2[freqs < 120.0]))
    speech_energy = float(np.sum(fft_mag2[(freqs >= 300.0) & (freqs <= 3200.0)]))

    rumble_frac = rumble_energy / total_energy
    speech_frac = speech_energy / total_energy

    # 3. Power-grid hum detection (50 Hz vs 60 Hz spectral peak prominence, Rule 4 > 4.0x)
    # Measure hum prominence on quiet / non-vocal frames when available so vocal pitch harmonics
    # (e.g. 100/120/150/180 Hz) on active frames never falsely trigger power-line notches.
    if n_frames >= 10:
        q_thresh = float(np.percentile(raw_frame_rms, 35))
        q_idx = np.where(raw_frame_rms <= max(q_thresh, 1e-7))[0]
        q_samples = raw_frames[q_idx].flatten() if len(q_idx) > 0 else spec_samples
        if len(q_samples) >= 4 * frame_len:
            q_dur = min(len(q_samples), 60 * SR)
            q_fft_mag2 = np.abs(np.fft.rfft(q_samples[:q_dur])) ** 2
            q_freqs = np.fft.rfftfreq(q_dur, 1.0 / SR)
        else:
            q_fft_mag2, q_freqs = fft_mag2, freqs
    else:
        q_fft_mag2, q_freqs = fft_mag2, freqs

    hum_freq = None
    notch_freqs: tuple[float, ...] = ()
    reasoning = []

    def peak_prominence(center_f: float, half_bw: float = 2.0, bg_bw: float = 8.0) -> float:
        peak_mask = (q_freqs >= center_f - half_bw) & (q_freqs <= center_f + half_bw)
        bg_mask = (q_freqs >= center_f - bg_bw) & (q_freqs <= center_f + bg_bw) & ~peak_mask
        if not np.any(peak_mask) or not np.any(bg_mask):
            return 1.0
        p_val = float(np.max(q_fft_mag2[peak_mask]))
        bg_val = float(np.median(q_fft_mag2[bg_mask]) + 1e-12)
        return p_val / bg_val

    fund_60 = peak_prominence(60.0)
    fund_50 = peak_prominence(50.0)
    prom_60 = 0.50 * fund_60 + 0.35 * peak_prominence(120.0) + 0.15 * peak_prominence(180.0)
    prom_50 = 0.50 * fund_50 + 0.35 * peak_prominence(100.0) + 0.15 * peak_prominence(150.0)

    if fund_60 >= 2.5 and prom_60 > 4.0 and prom_60 > prom_50 * 1.25:
        hum_freq = 60.0
        notches = [60.0, 120.0, 180.0]
        for h in (240.0, 300.0, 360.0, 420.0, 480.0, 540.0, 600.0):
            if peak_prominence(h) >= 4.0:
                notches.append(h)
        notch_freqs = tuple(notches)
        reasoning.append(
            f"Detected 60 Hz electrical grid hum (harmonic comb prominence={prom_60:.1f}x); "
            f"engaged notches at {'/'.join(str(int(f)) for f in notch_freqs)} Hz."
        )
    elif fund_50 >= 2.5 and prom_50 > 4.0 and prom_50 > prom_60 * 1.25:
        hum_freq = 50.0
        notches = [50.0, 100.0, 150.0]
        for h in (200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0):
            if peak_prominence(h) >= 4.0:
                notches.append(h)
        notch_freqs = tuple(notches)
        reasoning.append(
            f"Detected 50 Hz electrical grid hum (harmonic comb prominence={prom_50:.1f}x); "
            f"engaged notches at {'/'.join(str(int(f)) for f in notch_freqs)} Hz."
        )
    else:
        reasoning.append("No prominent power-grid hum detected; power-line notch filters disabled to preserve vocal pitch.")

    # 4. Adaptive low-frequency rumble cutoff
    if rumble_frac >= 0.70:
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

    # 5. Conditioned in-band RMS & SNR estimation (post-HPF / post-notch)
    # Measuring noise_floor_rms after removing out-of-band <120Hz rumble and hum prevents
    # downstream noise_bounded_agc and pre-gating from misclassifying stripped rumble as in-band hiss.
    cond_samples = butterworth_highpass(samples, cutoff_hz=hpf_cutoff, order=4)
    if notch_freqs:
        cond_samples = harmonic_notch_filter(cond_samples, freqs=notch_freqs, bw_hz=6.0)

    if n_frames > 0:
        frames = cond_samples[:n_frames * frame_len].reshape(n_frames, frame_len)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
        active_rms = frame_rms[frame_rms > 1e-6]
        if len(active_rms) >= 4:
            noise_floor = float(np.percentile(active_rms, 15))
            p95_rms = float(np.percentile(active_rms, 95))
            # Capture sparse short utterances (<5% duty cycle) via top sustained frames (~0.4s)
            top_k = min(len(active_rms), max(2, min(8, len(active_rms) // 8)))
            sustained_peak = float(np.mean(np.sort(active_rms)[-top_k:]))
            peak_rms = max(p95_rms, sustained_peak)
        else:
            noise_floor = float(np.percentile(frame_rms, 15))
            peak_rms = float(np.percentile(frame_rms, 95))
    else:
        noise_floor = float(np.sqrt(np.mean(cond_samples ** 2) + 1e-12))
        peak_rms = noise_floor

    snr_db = float(20.0 * np.log10(max(peak_rms, 1e-6) / max(noise_floor, 1e-6)))

    # Update speech fraction on conditioned spectrum if heavy rumble was stripped
    if rumble_frac >= 0.40:
        cond_spec = cond_samples[:spec_dur]
        cond_mag2 = np.abs(np.fft.rfft(cond_spec)) ** 2
        speech_frac = float(np.sum(cond_mag2[(freqs >= 300.0) & (freqs <= 3200.0)])) / (float(np.sum(cond_mag2)) + 1e-12)

    # 6. Formant boost calculation
    if snr_db >= 18.0 and speech_frac >= 0.35:
        formant_boost = 0.0
        reasoning.append(f"Clean speech detected (SNR={snr_db:.1f}dB, speech={speech_frac*100:.1f}%); formant boost bypassed to avoid distortion.")
    elif snr_db <= 0.0 or speech_frac <= 0.10:
        formant_boost = float(min(5.0, 2.5 + max(0.0, -snr_db) * 0.25))
        reasoning.append(f"Speech near or below noise floor (SNR={snr_db:.1f}dB); formant band boost set to +{formant_boost:.1f} dB.")
    else:
        formant_boost = float(np.clip(3.5 - (snr_db - 5.0) * 0.2, 1.0, 4.0))
        reasoning.append(f"Moderate noise floor (SNR={snr_db:.1f}dB); formant band boosted by +{formant_boost:.1f} dB.")

    # 7. Adaptive transducer blank penalty (Rule 4: 0.30 - 1.05)
    blank_penalty = float(np.clip(0.55 + (8.0 - snr_db) * 0.03, 0.3, 1.05))
    reasoning.append(f"Transducer blank penalty dynamically tuned to {blank_penalty:.2f} based on estimated SNR ({snr_db:.1f} dB).")

    # 8. Dynamic speech-band pre-gate threshold
    if n_frames > 0:
        quiet_indices = np.argsort(frame_rms)[:max(1, n_frames // 5)]
        quiet_samples = frames[quiet_indices].flatten()
        speech_floor = speech_band_energy(quiet_samples)
    else:
        speech_floor = noise_floor * 0.3
    pregate_threshold = float(max(1e-5, speech_floor * 0.25))
    reasoning.append(f"Dynamic speech-band pre-gate threshold synthesized at {pregate_threshold:.6f} (silence gate at 0.25x quiet-band floor).")

    # 9. Adaptive windowing
    if dur <= 25.0:
        win_sec, step_sec = 5.0, 1.25
    elif dur <= 120.0:
        win_sec, step_sec = 5.0, 1.75
    else:
        win_sec, step_sec = 6.0, 2.5

    return AcousticProfile(
        noise_floor_rms=noise_floor,
        raw_noise_floor_rms=raw_noise_floor,
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


def _exp1_approx(v: np.ndarray) -> np.ndarray:
    """Pure-NumPy Abramowitz & Stegun (5.1.53 / 5.1.56) approximation of E_1(v)."""
    v_safe = np.clip(v, 1e-7, 40.0)
    out = np.empty_like(v_safe, dtype=np.float32)
    low = v_safe <= 1.0
    if np.any(low):
        vl = v_safe[low]
        out[low] = (
            -np.log(vl)
            - 0.57721566
            + vl * (0.99999193 + vl * (-0.24991055 + vl * (0.05519968 + vl * (-0.00976004 + vl * 0.00107857))))
        )
    high = ~low
    if np.any(high):
        vh = v_safe[high]
        num = vh * vh + 2.334733 * vh + 0.250621
        den = vh * vh + 3.330657 * vh + 1.681534
        out[high] = (np.exp(-vh) / vh) * (num / den)
    return np.maximum(out, 0.0)


def adaptive_spectral_gate(
    samples: np.ndarray,
    sr: int = SR,
    win_len: int = 512,
    hop_len: int = 128,
    g_floor_db: float = -20.0,
    alpha_dd: float = 0.96,
) -> np.ndarray:
    """Pure-NumPy Ephraim-Malah Log-MMSE Spectral Estimator with SPP-Modulated MCRA Noise Tracking.

    Initializes the stationary noise PSD from the 20th-percentile non-faded frames (preventing both
    Tukey-fade zero poisoning and t=0 speech poisoning) and applies Log-MMSE gain to preserve
    sub-noise speech formants without musical noise.
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
    mag = np.abs(stft)
    mag2 = mag ** 2

    # 2. Unbiased Minimum-Frame Noise PSD Initialization on non-zero / non-faded frames
    # Selecting the quietest valid frames by total frame power and taking the per-bin mean
    # avoids the 4.5x (-6.5 dB) downward bias of per-bin P20 on exponential STFT power.
    frame_power = np.mean(mag2, axis=1)
    max_fp = float(np.max(frame_power)) if len(frame_power) > 0 else 0.0
    valid_idx = np.where(frame_power > max(1e-12, max_fp * 1e-4))[0]
    if len(valid_idx) > 0:
        valid_powers = frame_power[valid_idx]
        n_quiet = max(1, int(np.ceil(0.30 * len(valid_idx))))
        quiet_order = np.argsort(valid_powers)[:n_quiet]
        noise_psd = np.maximum(np.mean(mag2[valid_idx[quiet_order]], axis=0), 1e-12).astype(np.float32)
    else:
        noise_psd = np.maximum(np.mean(mag2, axis=0), 1e-12).astype(np.float32)

    # Also keep smoothed prior PSD initialized properly
    smooth_power = noise_psd.copy()
    prior_s_mag2 = (noise_psd * 0.25).copy()
    out_stft = np.zeros_like(stft)

    beta_rise = 0.006
    beta_fall = 0.04  # Reduced from 0.14: aggressive fall caused noise-estimate drop during pauses,
                      # leading to word-onset under-enhancement when speech resumed.
    alpha_p = 0.80

    for m in range(n_frames):
        cur_mag2 = mag2[m]
        smooth_power = alpha_p * smooth_power + (1.0 - alpha_p) * cur_mag2

        # SPP-modulated MCRA update: freeze upward drift on likely speech bins (>2.5x noise mean)
        sr_ratio = smooth_power / np.maximum(noise_psd, 1e-12)
        p_speech = np.clip((sr_ratio - 2.2) / 3.5, 0.0, 1.0)
        alpha_d = (1.0 - p_speech) * beta_rise

        mask_rise = smooth_power > noise_psd
        noise_psd[mask_rise] += alpha_d[mask_rise] * (smooth_power[mask_rise] - noise_psd[mask_rise])
        noise_psd[~mask_rise] += beta_fall * (smooth_power[~mask_rise] - noise_psd[~mask_rise])
        noise_psd = np.maximum(noise_psd, 1e-12)

        # 3. Decision-Directed A Priori SNR
        post_snr = np.minimum(cur_mag2 / noise_psd, 1000.0)
        prior_snr = alpha_dd * (prior_s_mag2 / noise_psd) + (1.0 - alpha_dd) * np.maximum(post_snr - 1.0, 0.0)
        prior_snr = np.maximum(prior_snr, 1e-6)

        # 4. Ephraim-Malah Log-MMSE Gain
        xi_term = prior_snr / (1.0 + prior_snr)
        v = xi_term * post_snr
        gain = xi_term * np.exp(0.5 * np.minimum(_exp1_approx(v), 12.0))
        gain = np.minimum(gain, 1.0)

        # Inter-bin smoothing to eliminate isolated musical noise tones
        gain_smooth = np.convolve(gain, [0.20, 0.60, 0.20], mode="same")
        gain_bounded = np.clip(gain_smooth, g_floor, 1.0)

        s_mag = gain_bounded * mag[m]
        prior_s_mag2 = s_mag ** 2
        out_stft[m] = stft[m] * gain_bounded

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
    nccf_threshold: float = 0.40,
    hcr_threshold: float = 3.0,
) -> np.ndarray:
    """Detect faint voiced speech buried in severe noise using exact Cauchy-Schwarz NCCF and high-res HPS.

    Conditions signal to the vocal pitch + low-formant band (70 - 1200 Hz) without destructive
    high-pass pre-emphasis, removes 50/60 Hz hum, and computes exact unwindowed normalized
    cross-correlation (bound = 1.0) plus 2048-point zero-padded Harmonic Product Spectrum.
    """
    if len(samples) == 0:
        return np.array([False], dtype=bool)

    # 1. Conditioning: remove sub-70Hz rumble, electrical hum, and >1200Hz broadband hiss
    filtered = butterworth_highpass(samples, cutoff_hz=max(65.0, f0_min - 5.0))
    prof = profile_audio(samples)
    if prof.notch_freqs:
        filtered = harmonic_notch_filter(filtered, freqs=prof.notch_freqs)

    if len(filtered) >= 64:
        n_orig = len(filtered)
        pad_len = min(n_orig, sr // 4)
        fp = np.pad(filtered - float(np.mean(filtered)), pad_len, mode="reflect")
        f_fft = np.fft.rfft(fp)
        f_hz = np.fft.rfftfreq(len(fp), 1.0 / sr)
        lp_mask = np.ones_like(f_hz, dtype=np.float32)
        trans = (f_hz > 1100.0) & (f_hz < 1500.0)
        lp_mask[trans] = 0.5 * (1.0 + np.cos(np.pi * (f_hz[trans] - 1100.0) / 400.0))
        lp_mask[f_hz >= 1500.0] = 0.0
        pitch_band = np.fft.irfft(f_fft * lp_mask, n=len(fp))[pad_len : pad_len + n_orig].astype(np.float32)
    else:
        pitch_band = filtered.astype(np.float32)

    frame_len = int(frame_sec * sr)
    hop_len = int(hop_sec * sr)
    n_frames = 1 + (len(pitch_band) - frame_len) // hop_len
    if n_frames <= 0:
        return np.array([False], dtype=bool)

    lag_min = max(1, int(sr / f0_max))
    lag_max = int(sr / f0_min)

    speech_active = np.zeros(n_frames, dtype=bool)
    window = np.hanning(frame_len).astype(np.float32)
    n_fft_corr = 1
    while n_fft_corr < frame_len + lag_max + 1:
        n_fft_corr *= 2

    for i in range(n_frames):
        s_idx = i * hop_len
        chunk_raw = filtered[s_idx : s_idx + frame_len]
        chunk = pitch_band[s_idx : s_idx + frame_len]
        if len(chunk) < frame_len:
            continue

        c_rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-12))
        if c_rms < 1e-5:
            continue

        ext_len = frame_len + lag_max
        if s_idx + ext_len > len(pitch_band):
            ext_chunk = np.pad(pitch_band[s_idx:], (0, ext_len - (len(pitch_band) - s_idx)))
        else:
            ext_chunk = pitch_band[s_idx : s_idx + ext_len]

        # Symmetric unwindowed zero-mean Cauchy-Schwarz NCCF (attains 1.0 on periodic signal)
        c0 = chunk - float(np.mean(chunk))
        ext0 = ext_chunk - float(np.mean(ext_chunk[:frame_len]))
        chunk_e = float(np.sum(c0 ** 2))
        if chunk_e < 1e-10:
            continue

        fft_c = np.fft.rfft(c0, n=n_fft_corr)
        fft_ext = np.fft.rfft(ext0, n=n_fft_corr)
        corr = np.fft.irfft(fft_ext * np.conj(fft_c), n=n_fft_corr)

        cum_sq = np.concatenate(([0.0], np.cumsum(ext0 ** 2)))
        del_idx = np.arange(lag_min, lag_max + 1)
        del_e = cum_sq[del_idx + frame_len] - cum_sq[del_idx]
        denom = np.sqrt(chunk_e * np.maximum(del_e, 1e-12))

        lags_corr = corr[lag_min : lag_max + 1]
        nccf_max = float(np.max(lags_corr / denom))

        # High-resolution 2048-point zero-padded Harmonic Product Spectrum (7.8 Hz/bin)
        mag = np.abs(np.fft.rfft((chunk_raw - float(np.mean(chunk_raw))) * window, n=2048))
        max_h = 3 if len(mag) >= 2048 else 2
        n_pts = len(mag) // max_h
        min_bin = max(1, int(f0_min * 2048 / sr))
        max_bin = min(n_pts, int(f0_max * 2048 / sr) + 1)
        if max_bin > min_bin + 2:
            hps = mag[:n_pts].copy()
            for h in range(2, max_h + 1):
                hps *= mag[::h][:n_pts]
            hps_roi = hps[min_bin:max_bin]
            hcr = float(np.max(hps_roi)) / float(np.median(hps_roi) + 1e-12)
        else:
            hcr = 1.0

        if (nccf_max >= nccf_threshold and hcr >= hcr_threshold) or nccf_max >= (nccf_threshold + 0.22):
            speech_active[i] = True

    # Median filter smoothing (3 frames ~ 30ms) to suppress isolated transient spikes
    if len(speech_active) >= 3:
        smoothed = speech_active.copy()
        for i in range(1, len(speech_active) - 1):
            smoothed[i] = (
                (speech_active[i - 1] and speech_active[i])
                or (speech_active[i] and speech_active[i + 1])
                or (speech_active[i - 1] and speech_active[i + 1])
            )
        speech_active = smoothed

    return speech_active


def multiband_formant_agc(
    samples: np.ndarray,
    sr: int = SR,
    target_formant_rms: float = 0.12,
    max_noise_boost_db: float = 6.0,
) -> np.ndarray:
    """Zero-phase 3-band dynamic range equalizer with envelope-gated formant expansion.

    Lifts faint speech formants (140 - 3400 Hz) and consonant fricatives (3400 - 7500 Hz)
    while strictly bounding stationary background hiss amplification.
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

    # 3. Speech-Aware Envelope & Spectral Structure Gating
    g1 = 0.25
    max_bg_linear = 10.0 ** (max_noise_boost_db / 20.0)

    # Measure spectral flatness in core formant band (180 - 3000 Hz)
    b2_bins = (freqs >= 180.0) & (freqs <= 3000.0)
    if np.any(b2_bins):
        b2_pow = (np.abs(fft[b2_bins]) ** 2) + 1e-12
        flatness_b2 = float(np.exp(np.mean(np.log(b2_pow))) / np.mean(b2_pow))
    else:
        flatness_b2 = 0.5

    frame_len = max(64, int(0.020 * sr))
    n_fr = n // frame_len
    if n_fr >= 2:
        fr_b2 = np.sqrt(np.mean(sig_b2[: n_fr * frame_len].reshape(n_fr, frame_len) ** 2, axis=1) + 1e-12)
        floor_b2 = float(np.percentile(fr_b2, 15))
        peak_b2 = float(np.percentile(fr_b2, 92))
        mod_ratio = peak_b2 / max(floor_b2, 1e-6)

        # Stationary broadband hiss (high flatness + near-unit envelope modulation): cap near unity
        is_stationary_hiss = (flatness_b2 > 0.35) and (mod_ratio < 1.35)
        bg_cap = 1.20 if is_stationary_hiss else min(max_bg_linear, max(1.0, 0.030 / max(floor_b2, 1e-5)))

        desired_peak_g2 = target_formant_rms / max(peak_b2, 1e-5)
        if is_stationary_hiss:
            g2_vec = np.full(n, float(np.clip(desired_peak_g2, 1.0, bg_cap)), dtype=np.float32)
        elif flatness_b2 < 0.22 and mod_ratio < 1.35:
            # Sustained tonal formant (e.g. steady vowel): lift smoothly up to tonal cap
            g2_scalar = float(np.clip(desired_peak_g2, 1.0, min(8.0, max_bg_linear * 2.5)))
            g2_vec = np.full(n, g2_scalar, dtype=np.float32)
        else:
            # Modulated speech in noise: keep pause floor <= bg_cap, expand active syllables upward
            speech_cap = min(12.0, max(bg_cap, bg_cap * 2.75))
            peak_gain = float(np.clip(desired_peak_g2, 1.0, speech_cap))
            floor_gain = float(np.clip(desired_peak_g2, 1.0, bg_cap))
            snr_fr = np.clip((fr_b2 / max(floor_b2, 1e-6) - 1.15) / 1.35, 0.0, 1.0)
            fr_gain = floor_gain + snr_fr * max(0.0, peak_gain - floor_gain)
            # Smooth frame gain across 3 frames (~60ms) and interpolate to sample rate
            if len(fr_gain) >= 3:
                fr_gain = np.convolve(fr_gain, [0.25, 0.50, 0.25], mode="same")
            centers = (np.arange(n_fr, dtype=np.float32) + 0.5) * frame_len
            g2_vec = np.interp(np.arange(n, dtype=np.float32), centers, fr_gain).astype(np.float32)
    else:
        rms_b2 = float(np.sqrt(np.mean(sig_b2 ** 2) + 1e-12))
        g2_vec = np.full(n, float(np.clip(target_formant_rms / max(rms_b2, 1e-5), 1.0, max_bg_linear)), dtype=np.float32)

    p90_b3 = float(np.percentile(np.abs(sig_b3), 90))
    desired_g3 = (target_formant_rms * 0.8) / max(p90_b3, 1e-5)
    g3_vec = np.clip(np.minimum(desired_g3, g2_vec * 0.75), 1.0, 6.0).astype(np.float32)

    # 4. Soft Saturation Synthesis
    out_linear = (sig_b1 * g1) + (sig_b2 * g2_vec) + (sig_b3 * g3_vec)

    peak = float(np.max(np.abs(out_linear)))
    if peak > 0.75:
        out_linear = np.tanh(out_linear / 0.75) * 0.75

    return out_linear.astype(np.float32)


def suppress_impulsive_clicks(
    samples: np.ndarray,
    sr: int = SR,
    threshold_sigma: float = 4.8,
    max_click_samples: int = 24,
) -> tuple[np.ndarray, int]:
    """Pure-NumPy Adaptive Impulse & Keyboard Click Removal.

    Detects ultra-fast mechanical switch transients (<1.5ms) using 2nd-order difference
    curvature and Median Absolute Deviation (MAD), replacing click samples with cubic Hermite
    spline inpainting from surrounding clean acoustic context without attenuating speech consonants.
    """
    if len(samples) < 128:
        return samples.copy().astype(np.float32), 0

    out = samples.copy()
    n = len(out)

    d2 = np.zeros(n, dtype=np.float32)
    d2[1:-1] = np.abs(out[2:] - 2.0 * out[1:-1] + out[:-2])

    # Compute active global curvature scale so codec-gated digital-zero blocks (MAD=0)
    # cannot trigger false outlier erasures on consonant release bursts at speech onset.
    nonzero_d2 = d2[d2 > 1e-7]
    if len(nonzero_d2) >= 16:
        g_med = float(np.median(nonzero_d2))
        active_global_scale = 1.4826 * float(np.median(np.abs(nonzero_d2 - g_med)))
    else:
        active_global_scale = 1e-4
    scale_floor = max(0.08 * active_global_scale, 1e-5)

    block_len = 256
    n_blocks = n // block_len
    if n_blocks > 0:
        blocks = d2[: n_blocks * block_len].reshape(n_blocks, block_len)
        med_b = np.median(blocks, axis=1)
        mad_b = np.median(np.abs(blocks - med_b[:, None]), axis=1)

        local_scale = np.maximum(np.repeat(1.4826 * mad_b, block_len), scale_floor)
        local_med = np.repeat(med_b, block_len)

        rem = n - len(local_scale)
        if rem > 0:
            local_scale = np.pad(local_scale, (0, rem), mode="edge")
            local_med = np.pad(local_med, (0, rem), mode="edge")

        z_scores = np.abs(d2 - local_med) / local_scale
    else:
        med = float(np.median(d2))
        mad = float(np.median(np.abs(d2 - med)))
        z_scores = np.abs(d2 - med) / max(1.4826 * mad, scale_floor)

    outliers = z_scores > threshold_sigma
    if not np.any(outliers):
        return out.astype(np.float32), 0

    outlier_idx = np.where(outliers)[0]
    clusters = []
    curr = [outlier_idx[0]]
    for idx in outlier_idx[1:]:
        if idx - curr[-1] <= 4:
            curr.append(idx)
        else:
            clusters.append((curr[0], curr[-1]))
            curr = [idx]
    if curr:
        clusters.append((curr[0], curr[-1]))

    clicks_removed = 0
    for s_idx, e_idx in clusters:
        s = max(0, s_idx - 2)
        e = min(n - 1, e_idx + 2)
        width = e - s + 1
        if width <= max_click_samples:
            clicks_removed += 1
            left_val = out[s - 1] if s > 0 else 0.0
            right_val = out[e + 1] if e < n - 1 else 0.0

            t = np.linspace(0.0, 1.0, width, dtype=np.float32)
            h00 = (1.0 + 2.0 * t) * (1.0 - t) ** 2
            h01 = (3.0 - 2.0 * t) * (t ** 2)

            d_left = (out[s - 1] - out[s - 3]) * 0.5 if s > 2 else 0.0
            d_right = (out[min(n - 1, e + 3)] - out[e + 1]) * 0.5 if e < n - 3 else 0.0

            h10 = t * ((1.0 - t) ** 2)
            h11 = (t ** 2) * (t - 1.0)

            out[s : e + 1] = h00 * left_val + h01 * right_val + h10 * d_left * width + h11 * d_right * width

    return out.astype(np.float32), clicks_removed


def whisper_formant_filter(
    samples: np.ndarray,
    sr: int = SR,
    boost_db: float = 4.0,
    force_whisper: bool = False,
) -> np.ndarray:
    """Pure-NumPy Whisper & Distant Speech Formant Equalizer.

    Detects unvoiced spectral resonance in the 1.5 - 4.0 kHz band and applies dynamic tilt
    equalization that cuts sub-350 Hz rumble while boosting unvoiced formant ridges.
    """
    if len(samples) < 128:
        return samples.copy().astype(np.float32)

    n_orig = len(samples)
    pad_len = min(n_orig, sr // 2)
    x_mean = float(np.mean(samples))
    x_demeaned = samples - x_mean
    x_padded = np.pad(x_demeaned, pad_len, mode="reflect")
    n = len(x_padded)

    fft = np.fft.rfft(x_padded)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    mag2 = np.abs(fft) ** 2

    # 1. Unvoiced Spectral Resonance Analysis
    e_low = float(np.sum(mag2[(freqs >= 70.0) & (freqs < 450.0)])) + 1e-12
    e_whisper = float(np.sum(mag2[(freqs >= 1500.0) & (freqs < 4000.0)])) + 1e-12
    usri = e_whisper / e_low

    is_whisper = force_whisper or (usri >= 1.6)

    # 2. Synthesize Equalization Curve
    gain_db = np.zeros_like(freqs, dtype=np.float32)

    g_whisper = min(boost_db + 3.0, 8.5) if is_whisper else boost_db
    g_low_cut = -8.0 if is_whisper else -3.0
    g_fricative = 3.5

    # Band 1: 0 - 150 Hz (Low rumble suppression)
    gain_db[freqs < 150.0] = g_low_cut

    # Band 2: 150 - 350 Hz (Smooth transition to unity)
    r1 = (freqs >= 150.0) & (freqs < 350.0)
    w1 = (freqs[r1] - 150.0) / 200.0
    gain_db[r1] = g_low_cut * 0.5 * (1.0 + np.cos(np.pi * w1))

    # Band 3: 350 - 1500 Hz (Smooth transition up to whisper formant peak)
    r2 = (freqs >= 350.0) & (freqs < 1500.0)
    w2 = (freqs[r2] - 350.0) / 1150.0
    gain_db[r2] = g_whisper * 0.5 * (1.0 - np.cos(np.pi * w2))

    # Band 4: 1500 - 3800 Hz (Flat whisper formant plate)
    gain_db[(freqs >= 1500.0) & (freqs < 3800.0)] = g_whisper

    # Band 5: 3800 - 5000 Hz (Transition down to consonant shelf)
    r3 = (freqs >= 3800.0) & (freqs < 5000.0)
    w3 = (freqs[r3] - 3800.0) / 1200.0
    gain_db[r3] = g_fricative + (g_whisper - g_fricative) * 0.5 * (1.0 + np.cos(np.pi * w3))

    # Band 6: 5000 - 7200 Hz (Consonant frication shelf)
    gain_db[(freqs >= 5000.0) & (freqs < 7200.0)] = g_fricative

    # Band 7: 7200 - 8000 Hz (Anti-aliasing Nyquist roll-off)
    r4 = (freqs >= 7200.0)
    w4 = (freqs[r4] - 7200.0) / 800.0
    gain_db[r4] = g_fricative * 0.5 * (1.0 + np.cos(np.pi * w4))

    # 3. Apply Zero-Phase Linear Filtering
    linear_gain = 10.0 ** (gain_db / 20.0)
    filtered = np.fft.irfft(fft * linear_gain, n=n)

    out = (filtered[pad_len : pad_len + n_orig] + x_mean).astype(np.float32)
    return out


def detect_formant_modulation_spans(
    samples: np.ndarray,
    sr: int = SR,
    min_smi: float = 1.85,
    span_sec: float = 2.2,
) -> list[tuple[int, int]]:
    """Proactively detect sub-interval vocal bursts (whispers, short murmurs) using
    Spectral Modulation Index (SMI) in the human speech formant band (1.2 kHz - 4.0 kHz).

    Returns list of (start_sample, end_sample) indices for targeted micro-windowing.
    """
    if len(samples) < int(0.5 * sr):
        return []

    block_len = int(0.25 * sr)
    n_blocks = len(samples) // block_len
    if n_blocks < 2:
        return []

    n_orig = len(samples)
    pad_len = min(n_orig, sr // 2)
    x_padded = np.pad(samples - float(np.mean(samples)), pad_len, mode="reflect")
    n_fft = len(x_padded)

    fft = np.fft.rfft(x_padded)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    mask = np.zeros_like(freqs)
    r_low = (freqs >= 1000.0) & (freqs < 1400.0)
    w_low = (freqs[r_low] - 1000.0) / 400.0
    mask[r_low] = 0.5 * (1.0 - np.cos(np.pi * w_low))
    mask[(freqs >= 1400.0) & (freqs <= 3800.0)] = 1.0
    r_high = (freqs > 3800.0) & (freqs <= 4200.0)
    w_high = (freqs[r_high] - 3800.0) / 400.0
    mask[r_high] = 0.5 * (1.0 + np.cos(np.pi * w_high))

    bp_sig = np.fft.irfft(fft * mask, n=n_fft)[pad_len : pad_len + n_orig]

    rms = np.zeros(n_blocks, dtype=np.float32)
    for k in range(n_blocks):
        sub = bp_sig[k * block_len : (k + 1) * block_len]
        rms[k] = np.sqrt(np.mean(sub ** 2) + 1e-12)

    med_rms = float(np.median(rms))
    smi = rms / max(med_rms, 1e-6)

    span_len = int(span_sec * sr)
    half_span = span_len // 2

    spans = []
    for k in range(n_blocks):
        if smi[k] >= min_smi:
            center = int((k + 0.5) * block_len)
            s = max(0, center - half_span)
            e = min(n_orig, s + span_len)
            if e - s < span_len and s > 0:
                s = max(0, e - span_len)
            spans.append((s, e))

    if not spans:
        return []

    merged = [spans[0]]
    for s, e in spans[1:]:
        prev_s, prev_e = merged[-1]
        if s <= prev_e:
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))

    return merged


def compute_speech_presence_prob(
    samples: np.ndarray,
    denoised: np.ndarray | None = None,
    sr: int = SR,
    hop_sec: float = 0.020,
    win_sec: float = 0.032,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multi-cue Speech Presence Probability (SPP) detector for sub-noise & background speech.

    Combines 4 pure-NumPy acoustic cues at `hop_sec` resolution (default 20ms = 50 Hz):
      1. Post-denoiser emergence (or Log-MMSE proxy) relative to local P15 floor.
      2. 2-8 Hz syllabic log-envelope modulation index (1.5s sliding window).
      3. Formant-band (300-3400 Hz) spectral structure (1 - spectral flatness).
      4. Low-band (70-900 Hz) harmonic periodicity peak.

    Returns:
      (spp, local_snr_db, local_floor_rms) each of shape (n_frames,).
    """
    win_len = max(128, int(win_sec * sr))
    hop_len = max(64, int(hop_sec * sr))
    if len(samples) < win_len:
        z = np.zeros(1, dtype=np.float32)
        return z, z - 20.0, z + 1e-5

    n_frames = 1 + (len(samples) - win_len) // hop_len
    frames_idx = np.arange(win_len)[None, :] + np.arange(n_frames)[:, None] * hop_len
    window = np.hanning(win_len).astype(np.float32)

    raw_frames = samples[frames_idx]
    stft = np.fft.rfft(raw_frames * window, axis=1)
    mag2 = (np.abs(stft) ** 2) + 1e-12
    freqs = np.fft.rfftfreq(win_len, 1.0 / sr)

    formant_bins = (freqs >= 300.0) & (freqs <= 3400.0)
    if not np.any(formant_bins):
        formant_bins = slice(None)

    formant_pow = mag2[:, formant_bins]
    raw_formant_rms = np.sqrt(np.mean(formant_pow, axis=1))

    # Cue 3: Formant spectral structure (tonal/harmonic peaks vs flat Gaussian hiss)
    geo_mean = np.exp(np.mean(np.log(formant_pow), axis=1))
    arith_mean = np.mean(formant_pow, axis=1) + 1e-12
    flatness = geo_mean / arith_mean
    # White noise in Hann STFT has frame flatness ~0.55; formant ridges drop to 0.05-0.38
    cue_structure = np.clip((0.50 - flatness) / 0.28, 0.0, 1.0).astype(np.float32)

    # Local background floor over ~6s sliding context (P15 of raw formant RMS)
    ctx_half = max(4, int(3.0 / hop_sec))
    local_floor = np.empty(n_frames, dtype=np.float32)
    stride = max(1, ctx_half // 4)
    ctrl_idx = np.arange(0, n_frames, stride)
    ctrl_vals = np.empty(len(ctrl_idx), dtype=np.float32)
    for k, m in enumerate(ctrl_idx):
        lo = max(0, m - ctx_half)
        hi = min(n_frames, m + ctx_half + 1)
        sub = raw_formant_rms[lo:hi]
        nz = sub[sub > 1e-7]
        ctrl_vals[k] = float(np.percentile(nz, 15)) if len(nz) >= 3 else float(np.percentile(sub, 15))
    local_floor = np.interp(np.arange(n_frames, dtype=np.float32), ctrl_idx.astype(np.float32), ctrl_vals).astype(np.float32)
    local_floor = np.maximum(local_floor, 1e-6)
    local_snr_db = (20.0 * np.log10(np.maximum(raw_formant_rms, 1e-7) / local_floor)).astype(np.float32)

    # Cue 1: Post-denoiser emergence (neural denoiser suppresses stationary noise by 15-30 dB
    # while leaving buried speech islands at 3x-20x the denoised floor)
    if denoised is not None and len(denoised) >= len(samples):
        den_frames = denoised[: len(samples)][frames_idx]
        den_stft = np.fft.rfft(den_frames * window, axis=1)
        den_pow = (np.abs(den_stft[:, formant_bins]) ** 2) + 1e-12
        den_rms = np.sqrt(np.mean(den_pow, axis=1))
        den_ctrl = np.empty(len(ctrl_idx), dtype=np.float32)
        for k, m in enumerate(ctrl_idx):
            lo = max(0, m - ctx_half)
            hi = min(n_frames, m + ctx_half + 1)
            den_ctrl[k] = float(np.percentile(den_rms[lo:hi], 20))
        den_floor = np.maximum(
            np.interp(np.arange(n_frames, dtype=np.float32), ctrl_idx.astype(np.float32), den_ctrl),
            1e-6,
        )
        den_ratio_db = 20.0 * np.log10(np.maximum(den_rms, 1e-7) / den_floor)
        cue_den = np.clip((den_ratio_db - 2.5) / 9.5, 0.0, 1.0).astype(np.float32)
    else:
        cue_den = np.clip((local_snr_db + 1.5) / 8.0, 0.0, 1.0).astype(np.float32)

    # Cue 2: 2-8 Hz syllabic modulation of log-formant envelope over 1.28s windows (64 frames @ 50Hz)
    env_fps = 1.0 / hop_sec
    log_env = np.log(raw_formant_rms + 1e-6)
    mod_win = min(n_frames, max(16, int(1.28 * env_fps)))
    cue_mod = np.zeros(n_frames, dtype=np.float32)
    if n_frames >= 16:
        mod_step = max(1, mod_win // 4)
        mod_freqs = np.fft.rfftfreq(mod_win, hop_sec)
        syl_mask = (mod_freqs >= 2.0) & (mod_freqs <= 8.0)
        rest_mask = (mod_freqs >= 0.8) & ~syl_mask
        if np.any(syl_mask):
            w_mod = np.hanning(mod_win).astype(np.float32)
            acc = np.zeros(n_frames, dtype=np.float32)
            cnt = np.zeros(n_frames, dtype=np.float32)
            for s in range(0, max(1, n_frames - mod_win + 1), mod_step):
                seg = log_env[s : s + mod_win]
                if len(seg) < mod_win:
                    break
                seg0 = (seg - float(np.mean(seg))) * w_mod
                mp = (np.abs(np.fft.rfft(seg0)) ** 2) + 1e-12
                e_syl = float(np.mean(mp[syl_mask]))
                e_rest = float(np.mean(mp[rest_mask])) if np.any(rest_mask) else 1e-6
                # Also require minimal envelope variance so pure digital silence doesn't score high
                env_std = float(np.std(seg))
                ratio = (e_syl / max(e_rest, 1e-8)) * min(1.0, env_std / 0.14)
                val = float(np.clip((ratio - 1.15) / 2.0, 0.0, 1.0))
                acc[s : s + mod_win] += val
                cnt[s : s + mod_win] += 1.0
            valid_c = cnt > 0
            cue_mod[valid_c] = acc[valid_c] / cnt[valid_c]

    # Cue 4: Vectorized low-band (80-900 Hz) autocorrelation periodicity
    pitch_bins = (freqs >= 80.0) & (freqs <= 900.0)
    low_stft = np.zeros_like(stft)
    low_stft[:, pitch_bins] = stft[:, pitch_bins]
    acorr = np.fft.irfft(np.abs(low_stft) ** 2, n=win_len, axis=1)
    lag_lo = max(1, int(sr / 340.0))
    lag_hi = min(win_len // 2, int(sr / 75.0))
    if lag_hi > lag_lo:
        r0 = np.maximum(acorr[:, 0], 1e-12)
        peak_r = np.max(acorr[:, lag_lo:lag_hi], axis=1) / r0
        cue_pitch = np.clip((peak_r - 0.28) / 0.35, 0.0, 1.0).astype(np.float32)
    else:
        cue_pitch = np.zeros(n_frames, dtype=np.float32)

    spp = (0.42 * cue_den + 0.24 * cue_mod + 0.19 * cue_structure + 0.15 * cue_pitch).astype(np.float32)
    if n_frames >= 5:
        spp = np.convolve(spp, [0.15, 0.20, 0.30, 0.20, 0.15], mode="same").astype(np.float32)
    return np.clip(spp, 0.0, 1.0), local_snr_db, local_floor


def find_subnoise_speech_spans(
    samples: np.ndarray,
    denoised: np.ndarray | None = None,
    sr: int = SR,
    spp_threshold: float = 0.36,
    min_dur_sec: float = 0.35,
    merge_gap_sec: float = 0.90,
    pad_sec: float = 0.45,
) -> list[dict]:
    """Locate candidate speech islands (including sub-noise-floor background speech) via 4-cue SPP.

    Returns list of dicts with keys:
      start_sec, end_sec, peak_spp, mean_spp, local_snr_db, local_floor_rms
    """
    hop_sec = 0.020
    dur_sec = len(samples) / sr
    if dur_sec < min_dur_sec:
        return []

    spp, snr_frames, floor_frames = compute_speech_presence_prob(samples, denoised=denoised, sr=sr, hop_sec=hop_sec)
    active = spp >= spp_threshold
    if not np.any(active):
        return []

    raw_spans: list[tuple[int, int]] = []
    in_span = False
    s_fr = 0
    for idx, flag in enumerate(active):
        if flag and not in_span:
            s_fr = idx
            in_span = True
        elif not flag and in_span:
            raw_spans.append((s_fr, idx))
            in_span = False
    if in_span:
        raw_spans.append((s_fr, len(active)))

    merge_frames = int(merge_gap_sec / hop_sec)
    merged: list[tuple[int, int]] = []
    for s_f, e_f in raw_spans:
        if merged and (s_f - merged[-1][1]) <= merge_frames:
            merged[-1] = (merged[-1][0], e_f)
        else:
            merged.append((s_f, e_f))

    min_frames = max(2, int(min_dur_sec / hop_sec))
    results: list[dict] = []
    for s_f, e_f in merged:
        if (e_f - s_f) < min_frames:
            continue
        span_spp = spp[s_f:e_f]
        span_snr = snr_frames[s_f:e_f]
        span_floor = floor_frames[s_f:e_f]
        t0 = max(0.0, s_f * hop_sec - pad_sec)
        t1 = min(dur_sec, e_f * hop_sec + pad_sec)
        results.append({
            "start_sec": round(t0, 3),
            "end_sec": round(t1, 3),
            "peak_spp": round(float(np.max(span_spp)), 3),
            "mean_spp": round(float(np.mean(span_spp)), 3),
            "local_snr_db": round(float(np.percentile(span_snr, 80)), 2),
            "local_floor_rms": float(np.median(span_floor)),
        })
    return results


class StreamingDeclicker:
    """Stateful, real-time streaming de-clicker for live microphone streams.

    Removes mechanical keyboard switch transients and impulse spikes across
    streaming chunk boundaries with zero edge pops and fixed 4.0ms algorithmic latency.
    """

    def __init__(self, sr: int = SR, margin_samples: int = 64, threshold_sigma: float = 4.8):
        self.sr = sr
        self.m = margin_samples
        self.threshold_sigma = threshold_sigma
        self.tail = np.zeros(2 * self.m, dtype=np.float32)
        self.clicks_detected = 0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Process incoming 32ms-64ms audio block. Returns cleaned chunk of same length."""
        if len(chunk) == 0:
            return chunk

        b_len = len(chunk)
        composite = np.concatenate([self.tail, chunk.astype(np.float32)])

        cleaned_composite, n_clicks = suppress_impulsive_clicks(
            composite,
            sr=self.sr,
            threshold_sigma=self.threshold_sigma,
            max_click_samples=24,
        )
        self.clicks_detected += n_clicks

        clean_out = cleaned_composite[self.m : self.m + b_len]

        # Preserve cleaned left context (already emitted) alongside raw un-emitted right margin
        # so boundary clicks wait for true future context instead of one-sided edge extrapolation.
        self.tail = np.concatenate([
            cleaned_composite[b_len : b_len + self.m],
            composite[b_len + self.m : b_len + 2 * self.m],
        ])

        return clean_out.astype(np.float32)

    def reset(self):
        self.tail.fill(0.0)
        self.clicks_detected = 0

