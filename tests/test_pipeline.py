"""
End-to-end pipeline integration tests.

Validates: TTS → audio.decode → robust.transcribe → transcript check.
Three noise types: clean / white / pink.  Aggressiveness levels 0-3.
Languages: en (always via Windows SAPI or espeak-ng), es/pt (skipped if no
TTS voice installed).

Run:
    pytest tests/test_pipeline.py -v

Skip reason legend:
    "models not installed"   → run scripts/install_models.sh / .ps1 first
    "no TTS voice for 'es'"  → install Windows Spanish language pack or espeak-ng
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from stt import audio, asr, config, robust


# ── model availability ────────────────────────────────────────────────────────

_CFG = config.load()
_MODELS_DIR = Path(_CFG["models_dir"])


def _models_ok() -> bool:
    m = _MODELS_DIR / _CFG["offline_model"]
    v = _MODELS_DIR / _CFG["vad_model"]
    return m.is_dir() and v.exists()


SKIP_NO_MODELS = pytest.mark.skipif(
    not _models_ok(),
    reason=f"offline model or VAD not installed in {_MODELS_DIR}",
)


# ── Windows SAPI TTS ─────────────────────────────────────────────────────────

def _sapi_installed_voices() -> dict:
    """Return {bcp47_culture: voice_name} for all Windows SAPI voices."""
    if sys.platform != "win32":
        return {}
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.GetInstalledVoices() | "
        "ForEach-Object { $_.VoiceInfo.Culture.Name + '|' + $_.VoiceInfo.Name }"
    )
    try:
        r = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-Command", ps],
            capture_output=True, text=True, timeout=10,
        )
        out = {}
        for line in r.stdout.strip().splitlines():
            if "|" in line:
                cult, name = line.split("|", 1)
                out[cult.strip().lower()] = name.strip()
        return out
    except Exception:
        return {}


_SAPI_VOICES: dict = _sapi_installed_voices()


def _sapi_say(text: str, culture_prefix: str) -> "Path | None":
    """Generate WAV via Windows SAPI; returns temp path or None."""
    voice = next(
        (name for cult, name in _SAPI_VOICES.items()
         if cult.startswith(culture_prefix.lower())),
        None,
    )
    if not voice:
        return None

    out_wav = Path(tempfile.mktemp(suffix=".wav"))
    # Forward-slashes work fine in PowerShell string paths
    wav_fwd = str(out_wav).replace("\\", "/")
    # Escape single quotes in voice name and text
    voice_esc = voice.replace("'", "''")
    text_esc = text.replace("'", "''")

    ps = (
        f"Add-Type -AssemblyName System.Speech\n"
        f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
        f"$s.Rate = -2\n"
        f"$s.SelectVoice('{voice_esc}')\n"
        f"$s.SetOutputToWaveFile('{wav_fwd}')\n"
        f"$s.Speak('{text_esc}')\n"
        f"$s.Dispose()\n"
    )
    ps_file = Path(tempfile.mktemp(suffix=".ps1"))
    try:
        ps_file.write_text(ps, encoding="utf-8")
        r = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ps_file)],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0 or not out_wav.exists() or out_wav.stat().st_size < 1000:
            out_wav.unlink(missing_ok=True)
            return None
        return out_wav
    except Exception:
        out_wav.unlink(missing_ok=True)
        return None
    finally:
        ps_file.unlink(missing_ok=True)


def _espeak_say(text: str, lang: str) -> "Path | None":
    """Generate WAV via espeak-ng; returns temp path or None."""
    if not shutil.which("espeak-ng"):
        return None
    out_wav = Path(tempfile.mktemp(suffix=".wav"))
    try:
        r = subprocess.run(
            ["espeak-ng", "-v", lang, "-s", "130", "-w", str(out_wav), "--", text],
            capture_output=True, timeout=15,
        )
        if r.returncode != 0 or not out_wav.exists() or out_wav.stat().st_size < 500:
            out_wav.unlink(missing_ok=True)
            return None
        return out_wav
    except Exception:
        out_wav.unlink(missing_ok=True)
        return None


def tts_samples(text: str, lang: str) -> "np.ndarray | None":
    """TTS → 16 kHz float32.  Returns None if no TTS available for lang."""
    wav = _sapi_say(text, lang) or _espeak_say(text, lang)
    if wav is None:
        return None
    try:
        return audio.decode(str(wav))
    except Exception:
        return None
    finally:
        wav.unlink(missing_ok=True)


# ── noise injection ───────────────────────────────────────────────────────────

def add_noise(samples: np.ndarray, snr_db: float, noise_type: str = "white") -> np.ndarray:
    """Add white or pink noise at the specified SNR (dB)."""
    rng = np.random.default_rng(42)
    white = rng.normal(0, 1, len(samples)).astype(np.float32)
    if noise_type == "pink":
        freqs = np.fft.rfftfreq(len(white))
        freqs[0] = 1e-9
        white = np.fft.irfft(np.fft.rfft(white) / np.sqrt(freqs), n=len(white)).astype(np.float32)
    rms_sig = float(np.sqrt(np.mean(samples ** 2))) + 1e-9
    rms_n = float(np.sqrt(np.mean(white ** 2))) + 1e-9
    scale = rms_sig / rms_n / (10 ** (snr_db / 20))
    return np.clip(samples + white * scale, -1.0, 1.0).astype(np.float32)


# ── language corpus ───────────────────────────────────────────────────────────

CORPUS = {
    "en": (
        "the quick brown fox jumps over the lazy dog",
        ["fox", "quick", "lazy", "dog", "jumps"],
    ),
    "es": (
        "el zorro rápido salta sobre el perro perezoso",
        ["zorro", "perro", "rápido", "salta"],
    ),
    "pt": (
        "a raposa ágil pula sobre o cão preguiçoso",
        ["raposa", "cão", "ágil", "pula"],
    ),
}


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg():
    return config.load()


@pytest.fixture(scope="module")
def rec(cfg):
    """Offline recognizer, created once per module (expensive — ~2 s)."""
    if not _models_ok():
        pytest.skip("models not installed")
    return asr.offline(cfg, blank_penalty=cfg.get("blank_penalty", 0.6))


# ── clean audio tests (en / es / pt) ─────────────────────────────────────────

@SKIP_NO_MODELS
@pytest.mark.parametrize("lang", ["en", "es", "pt"])
def test_clean_transcription(rec, cfg, lang):
    """TTS clean audio → level-0 VAD → Parakeet → at least one keyword found."""
    text, keywords = CORPUS[lang]
    samples = tts_samples(text, lang)
    if samples is None:
        pytest.skip(
            f"no TTS voice for '{lang}' — install the language pack or espeak-ng"
        )

    assert len(samples) / audio.SR >= 1.0, "TTS produced suspiciously short audio"

    triples = robust.transcribe(rec, samples, cfg, level=0)
    transcript = " ".join(t for _, _, t in triples).lower()

    assert transcript, (
        f"[{lang}] empty transcript from clean TTS audio "
        f"({len(samples)/audio.SR:.1f} s)"
    )

    matched = [kw for kw in keywords if kw in transcript]
    assert matched, (
        f"[{lang}] none of {keywords} found in: {transcript!r}"
    )


# ── noisy audio tests (white noise) ──────────────────────────────────────────

@SKIP_NO_MODELS
@pytest.mark.parametrize("snr_db,level", [
    (10.0, 1),   # mild noise — loose VAD
    ( 5.0, 2),   # moderate noise — neural denoise
    ( 3.0, 3),   # heavy noise — tuned recipe
    ( 0.0, 3),   # at the model's hard floor
])
def test_noisy_white(rec, cfg, snr_db, level):
    """White noise at given SNR, processed at matching aggressiveness level."""
    text, keywords = CORPUS["en"]
    samples = tts_samples(text, "en")
    if samples is None:
        pytest.skip("no English TTS available")

    noisy = add_noise(samples, snr_db=snr_db, noise_type="white")
    triples = robust.transcribe(rec, noisy, cfg, level=level)
    transcript = " ".join(t for _, _, t in triples).lower()

    # At 0 dB SNR (hard floor) we may get nothing — that is expected and documented.
    if snr_db <= 0.0:
        pytest.xfail(reason="0 dB white noise is at/below the model's hard floor")

    assert transcript, (
        f"no output at {snr_db}dB white noise / level {level}"
    )
    # Sanity: no hallucination storm (> 6 words/second)
    n_words = sum(len(t.split()) for _, _, t in triples)
    dur = len(noisy) / audio.SR
    assert n_words / dur < 6.0, (
        f"possible hallucination: {n_words} words over {dur:.1f}s"
    )


# ── noisy audio tests (pink noise) ───────────────────────────────────────────

@SKIP_NO_MODELS
@pytest.mark.parametrize("snr_db,level", [
    (5.0, 2),
    (2.0, 3),
])
def test_noisy_pink(rec, cfg, snr_db, level):
    """Pink noise at given SNR processed at matching aggressiveness level."""
    samples = tts_samples(CORPUS["en"][0], "en")
    if samples is None:
        pytest.skip("no English TTS available")

    noisy = add_noise(samples, snr_db=snr_db, noise_type="pink")
    triples = robust.transcribe(rec, noisy, cfg, level=level)
    transcript = " ".join(t for _, _, t in triples).lower()
    assert transcript, f"no output at {snr_db}dB pink noise / level {level}"


# ── level progression ─────────────────────────────────────────────────────────

@SKIP_NO_MODELS
def test_level_progression(rec, cfg):
    """On 5 dB white noise, levels 1-3 must each produce non-empty output.

    Levels are not required to improve monotonically word-count-wise (the
    denoiser can fragment utterances), but none should return nothing.
    """
    samples = tts_samples(CORPUS["en"][0], "en")
    if samples is None:
        pytest.skip("no English TTS available")

    noisy = add_noise(samples, snr_db=5.0, noise_type="white")
    for level in (0, 1, 2, 3):
        triples = robust.transcribe(rec, noisy, cfg, level=level)
        transcript = " ".join(t for _, _, t in triples).lower()
        assert transcript, f"level {level} produced no output on 5 dB noisy audio"


# ── disk round-trip ───────────────────────────────────────────────────────────

@SKIP_NO_MODELS
def test_file_round_trip(rec, cfg, tmp_path):
    """Write WAV → audio.decode() → transcribe → check keywords survive disk round-trip."""
    text, keywords = CORPUS["en"]
    samples = tts_samples(text, "en")
    if samples is None:
        pytest.skip("no English TTS available")

    wav = tmp_path / "roundtrip.wav"
    sf.write(str(wav), samples, audio.SR)

    decoded = audio.decode(str(wav))
    assert len(decoded) > 0
    # Length should match within ±1 s (resampling tolerance)
    assert abs(len(decoded) - len(samples)) < audio.SR, (
        f"decoded length {len(decoded)} differs from original {len(samples)} by more than 1 s"
    )

    triples = robust.transcribe(rec, decoded, cfg, level=0)
    transcript = " ".join(t for _, _, t in triples).lower()
    assert transcript, "empty transcript after disk round-trip"
    assert any(kw in transcript for kw in keywords), (
        f"no keywords {keywords} in: {transcript!r}"
    )


# ── VAD segmentation sanity ───────────────────────────────────────────────────

@SKIP_NO_MODELS
def test_vad_segments_clean(cfg):
    """Clean TTS audio should yield at least one VAD segment."""
    samples = tts_samples(CORPUS["en"][0], "en")
    if samples is None:
        pytest.skip("no English TTS available")

    segs = list(audio.segments(samples, cfg))
    assert segs, "Silero VAD found no speech in clean TTS audio"
    # Each segment must be a 3-tuple (start_sec, end_sec, samples_array)
    for s, e, chunk in segs:
        assert e > s, f"segment end {e} <= start {s}"
        assert len(chunk) > 0, "empty segment samples"


@SKIP_NO_MODELS
def test_vad_segments_silence(cfg):
    """Pure silence should yield zero VAD segments."""
    silence = np.zeros(int(3 * audio.SR), dtype=np.float32)
    segs = list(audio.segments(silence, cfg))
    assert segs == [], f"VAD found {len(segs)} segment(s) in pure silence"
