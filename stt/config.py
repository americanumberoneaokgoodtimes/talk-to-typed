"""Config loading: ~/.config/stt/config.toml (or %APPDATA%/stt/config.toml on Windows), written with defaults on first run."""
import os
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
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
    "models_dir": (DATA_DIR / "models").as_posix(),
    "offline_model": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    "streaming_model": "sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25",
    "vad_model": "silero_vad.onnx",
    "vad_threshold": 0.5,
    "vad_min_silence": 0.5,
    "vad_min_speech": 0.25,
    "vad_max_speech": 20.0,
    "vad_window_size": 512,                # VAD window size in samples (512 = 32ms at 16kHz)
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
    "mic_vad_threshold_whisper": 0.25,   # VAD threshold for --whisper mode
    "mic_vad_threshold_distant": 0.20,   # VAD threshold for --distant mode
    "blank_penalty": 0.6,                # Transducer blank penalty (0.0-1.5)
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
models_dir = '{models_dir}'
offline_model = '{offline_model}'       # used by `stt file` and the termux mic backend
streaming_model = '{streaming_model}'   # used by `stt mic` (pulse backend)
vad_model = '{vad_model}'               # relative to models_dir
vad_threshold = {vad_threshold}
vad_min_silence = {vad_min_silence}   # seconds of silence that end an utterance
vad_min_speech = {vad_min_speech}
vad_max_speech = {vad_max_speech}     # force-split segments longer than this (s)
vad_window_size = {vad_window_size}   # VAD window size in samples (512 = 32ms at 16kHz; 256/1024 also valid)
# --robust mode (noisy audio): denoise -> blend original back -> normalize -> permissive VAD
robust_denoiser = '{robust_denoiser}'   # relative to models_dir
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
mic_vad_threshold = {mic_vad_threshold}   # live-mic VAD sensitivity (streaming); lower catches soft/slow starts
mic_vad_threshold_whisper = {mic_vad_threshold_whisper}  # VAD threshold for --whisper mode
mic_vad_threshold_distant = {mic_vad_threshold_distant}  # VAD threshold for --distant mode
mic_vad_min_speech = {mic_vad_min_speech}
blank_penalty = {blank_penalty}               # transducer blank penalty (0.0-1.5)
aggressiveness = {aggressiveness}       # default noisy-audio level 0-4 (see `stt file -h`)
mic_backend = '{mic_backend}'           # auto | pulse | windows | termux
mic_chunk_seconds = {mic_chunk_seconds}  # termux backend chunk length
pulse_port = {pulse_port}               # private PulseAudio server port used by stt
termux_bin = '{termux_bin}'
termux_tmp = '{termux_tmp}'             # must be a path the Termux:API app can write
# Windows mic configuration
windows_mic_device = '{windows_mic_device}'  # audio device name or "default" (DirectShow/WASAPI)
windows_backend = '{windows_backend}'        # auto | dshow | sounddevice
"""


# Boolean config keys that should be normalized from string "true"/"false"
_BOOL_KEYS = {
    "mic_agc",
}

def _normalize_bool_config(cfg: dict) -> dict:
    """Convert string 'true'/'false' to boolean for known boolean keys."""
    for key in _BOOL_KEYS:
        if key in cfg and isinstance(cfg[key], str):
            val = cfg[key].strip().lower()
            if val in ("true", "false"):
                cfg[key] = val == "true"
    return cfg


def load() -> dict:
    cfg = dict(DEFAULTS)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(TEMPLATE.format(**DEFAULTS, mic_agc_toml=str(DEFAULTS['mic_agc']).lower()))
    else:
        with open(CONFIG_PATH, "rb") as f:
            cfg.update(tomllib.load(f))
    return _normalize_bool_config(cfg)


def _toml_line(key: str, value) -> str:
    """Serialize one key=value pair as valid TOML."""
    if isinstance(value, bool):
        return f"{key} = {str(value).lower()}"
    elif isinstance(value, (int, float)):
        return f"{key} = {value}"
    elif isinstance(value, str):
        # Escape backslashes and double-quotes so the result is valid TOML
        escaped = value.replace('\\', '\\\\').replace('"', '\\"')
        return f'{key} = "{escaped}"'
    return f"{key} = {value}"


def update_config(updates: dict) -> None:
    """Update config file, replacing only changed keys and preserving all comments."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not CONFIG_PATH.exists():
        # First-time write: use the full commented template
        cfg = dict(DEFAULTS)
        cfg.update(updates)
        CONFIG_PATH.write_text(TEMPLATE.format(**cfg, mic_agc_toml=str(cfg['mic_agc']).lower()))
        return

    remaining = dict(updates)
    new_lines = []
    for line in CONFIG_PATH.read_text().splitlines():
        stripped = line.strip()
        # Preserve blank lines and comment lines unchanged
        if not stripped or stripped.startswith('#') or '=' not in stripped:
            new_lines.append(line)
            continue
        key = stripped.split('=', 1)[0].strip()
        if key in remaining:
            new_lines.append(_toml_line(key, remaining.pop(key)))
        else:
            new_lines.append(line)

    # Append any keys not already present in the file
    for key, value in remaining.items():
        new_lines.append(_toml_line(key, value))

    CONFIG_PATH.write_text('\n'.join(new_lines) + '\n')
