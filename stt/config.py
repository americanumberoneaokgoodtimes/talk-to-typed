"""Config loading: ~/.config/stt/config.toml (or %APPDATA%/stt/config.toml on Windows), written with defaults on first run."""
import os
import sys
import tomllib
from pathlib import Path

IS_WIN = sys.platform == "win32"

if IS_WIN:
    CONFIG_PATH = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "stt" / "config.toml"
    DATA_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "stt"
else:
    CONFIG_PATH = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "stt" / "config.toml"
    DATA_DIR = Path.home() / ".local" / "share" / "stt"

DEFAULTS = {
    "threads": 4,
    "models_dir": str(DATA_DIR / "models"),
    "offline_model": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    "streaming_model": "sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25",
    "vad_model": "silero_vad.onnx",
    "vad_threshold": 0.5,
    "vad_min_silence": 0.5,
    "vad_min_speech": 0.25,
    "vad_max_speech": 20.0,
    "robust_denoiser": "dpdfnet8.onnx",  # sherpa-onnx speech-enhancement-models release
    "robust_mix": 0.3,                   # fraction of original audio blended back after denoising
    "robust_vad_threshold": 0.1,
    "robust_vad_min_silence": 0.5,
    "robust_vad_min_speech": 0.1,
    "robust_vad_max_speech": 5.0,
    "mic_calibrate_seconds": 2.0,
    "mic_agc": True,
    "mic_agc_target": 0.06,
    "mic_agc_max_gain": 40.0,
    "mic_preroll": 2.0,
    "mic_vad_threshold": 0.35,
    "mic_vad_min_speech": 0.1,
    "aggressiveness": 0,
    "mic_backend": "auto",
    "mic_chunk_seconds": 5,
    "pulse_port": 4714,
    "termux_bin": "/data/data/com.termux/files/usr/bin",
    "termux_tmp": "/data/data/com.termux/files/home/.cache/stt",
    "windows_mic_device": "default",
    "windows_backend": "auto",
}

TEMPLATE = """# stt configuration (created on first run; edit freely)
threads = {threads}
models_dir = "{models_dir}"
offline_model = "{offline_model}"       # used by `stt file` and the termux mic backend
streaming_model = "{streaming_model}"   # used by `stt mic` (pulse backend)
vad_model = "{vad_model}"               # relative to models_dir
vad_threshold = {vad_threshold}
vad_min_silence = {vad_min_silence}   # seconds of silence that end an utterance
vad_min_speech = {vad_min_speech}
vad_max_speech = {vad_max_speech}     # force-split segments longer than this (s)
# --robust mode (noisy audio): denoise -> blend original back -> normalize -> permissive VAD
robust_denoiser = "{robust_denoiser}"   # relative to models_dir
robust_mix = {robust_mix}               # 0 = denoised only; higher hides denoiser artifacts
robust_vad_threshold = {robust_vad_threshold}
robust_vad_min_silence = {robust_vad_min_silence}
robust_vad_min_speech = {robust_vad_min_speech}
robust_vad_max_speech = {robust_vad_max_speech}   # shorter pieces: the recognizer gives up early on long noisy ones
mic_calibrate_seconds = {mic_calibrate_seconds}   # --whisper/--distant: seconds of silence measured at start (0 = skip)
mic_agc = {mic_agc_toml}               # auto-gain for quiet speech / whispering (live mic)
mic_agc_target = {mic_agc_target}
mic_agc_max_gain = {mic_agc_max_gain}  # linear; 40 = +32 dB
mic_preroll = {mic_preroll}             # seconds of audio replayed from before the VAD fired (live mic)
mic_vad_threshold = {mic_vad_threshold}   # live-mic VAD sensitivity; lower catches soft/slow starts
mic_vad_min_speech = {mic_vad_min_speech}
aggressiveness = {aggressiveness}       # default noisy-audio level 0-4 (see `stt file -h`)
mic_backend = "{mic_backend}"           # auto | pulse | termux
mic_chunk_seconds = {mic_chunk_seconds}  # termux backend chunk length
pulse_port = {pulse_port}               # private PulseAudio server port used by stt
termux_bin = "{termux_bin}"
termux_tmp = "{termux_tmp}"             # must be a path the Termux:API app can write
# Windows mic configuration
windows_mic_device = "{windows_mic_device}"  # audio device name or "default" (DirectShow/WASAPI)
windows_backend = "{windows_backend}"        # auto | dshow | sounddevice
"""


def load() -> dict:
    cfg = dict(DEFAULTS)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(TEMPLATE.format(**DEFAULTS, mic_agc_toml=str(DEFAULTS['mic_agc']).lower()))
    else:
        with open(CONFIG_PATH, "rb") as f:
            cfg.update(tomllib.load(f))
    return cfg
