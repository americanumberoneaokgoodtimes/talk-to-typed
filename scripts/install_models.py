#!/usr/bin/env python3
"""Cross-platform model installer for stt (Linux, Android Termux/Kali, Windows).

Downloads and unpacks required models into the platform-appropriate models directory:
  - Windows: %LOCALAPPDATA%\\stt\\models (or %APPDATA%\\stt\\models)
  - Linux/Android: ~/.local/share/stt/models

Usage:
  python scripts/install_models.py [--all] [--dest DIR] [--retry N]
"""
import argparse
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ASR_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
ENH_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models"

DEFAULT_RETRIES = 3
RETRY_DELAY = 2  # seconds


def _file_size_mb(path: Path) -> str:
    return f"{path.stat().st_size / (1024 * 1024):.1f} MB" if path.exists() else "0.0 MB"


def is_valid_onnx(path: Path) -> bool:
    """Quick sanity check: non-empty ONNX protobuf file (starts with ir_version field tag 0x08)."""
    try:
        if not path.is_file() or path.stat().st_size < 64:
            return False
        with path.open("rb") as f:
            header = f.read(4)
        return len(header) == 4 and header[0] == 0x08 and not header.startswith((b"<!", b"<h", b"{"))
    except OSError:
        return False


def get_default_models_dir() -> Path:
    env_dir = os.environ.get("STT_MODELS")
    if env_dir:
        return Path(env_dir)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or (Path.home() / "AppData" / "Local")
        return Path(base) / "stt" / "models"
    return Path.home() / ".local" / "share" / "stt" / "models"


def download_file(url: str, dest_path: Path, desc: str, retries: int = DEFAULT_RETRIES):
    temp_file = dest_path.with_suffix(dest_path.suffix + ".download")
    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading {desc} (attempt {attempt}/{retries})...")

            def reporthook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, block_num * block_size * 100 // total_size)
                    mb = (block_num * block_size) / (1024 * 1024)
                    tot_mb = total_size / (1024 * 1024)
                    sys.stdout.write(f"\r  [{percent:3d}%] {mb:.1f} MB / {tot_mb:.1f} MB")
                    sys.stdout.flush()

            urllib.request.urlretrieve(url, str(temp_file), reporthook=reporthook)
            print()
            temp_file.replace(dest_path)
            return
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)
            if attempt < retries:
                print(f"  Attempt {attempt} failed: {e}; retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"failed to download {url} after {retries} attempts: {e}")


def install_tarball(url: str, filename: str, target_dir: Path, retries: int = DEFAULT_RETRIES):
    folder_name = filename.replace(".tar.bz2", "")
    final_folder = target_dir / folder_name
    if final_folder.is_dir():
        onnx_files = list(final_folder.glob("*.onnx"))
        if onnx_files and (final_folder / "tokens.txt").is_file() and all(is_valid_onnx(f) for f in onnx_files):
            print(f"Already installed: {folder_name}")
            return

    tar_path = target_dir / filename
    download_file(f"{url}/{filename}", tar_path, filename, retries=retries)

    print(f"Extracting {filename}...")
    try:
        with tarfile.open(tar_path, "r:bz2") as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(path=target_dir, filter="data")
            else:
                tar.extractall(path=target_dir)
        print(f"OK: {folder_name}")
    finally:
        if tar_path.exists():
            tar_path.unlink(missing_ok=True)


def install_raw_file(url: str, filename: str, target_dir: Path, retries: int = DEFAULT_RETRIES):
    final_path = target_dir / filename
    if is_valid_onnx(final_path):
        print(f"Already installed: {filename}")
        return
    download_file(f"{url}/{filename}", final_path, filename, retries=retries)
    if not is_valid_onnx(final_path):
        raise RuntimeError(f"Downloaded file {filename} is not a valid ONNX model")
    print(f"OK: {filename} ({_file_size_mb(final_path)})")


NLLB_URL = "https://huggingface.co/justheuristic/nllb-200-distilled-600M-ctranslate2/resolve/main"

def install_translation_model(target_dir: Path, retries: int = DEFAULT_RETRIES):
    trans_dir = target_dir / "nllb-200-distilled-600M"
    trans_dir.mkdir(parents=True, exist_ok=True)
    files = [
        "config.json",
        "model.bin",
        "sentencepiece.bpe.model",
        "shared_vocabulary.json",
        "special_tokens_map.json",
    ]
    for fn in files:
        fpath = trans_dir / fn
        if fpath.is_file() and fpath.stat().st_size > 0:
            print(f"Already installed: nllb-200-distilled-600M/{fn}")
            continue
        download_file(f"{NLLB_URL}/{fn}", fpath, f"nllb-200-distilled-600M/{fn}", retries=retries)
    print(f"OK: nllb-200-distilled-600M translation model ({_file_size_mb(trans_dir / 'model.bin')})")


def main():
    parser = argparse.ArgumentParser(description="Install speech-to-text models")
    parser.add_argument("--all", action="store_true", help="install all extra models (kroko, gtcrn, extra denoisers)")
    parser.add_argument("--translation", action="store_true", help="install NLLB-200 distilled 600M CTranslate2 model for --translate")
    parser.add_argument("--dest", type=Path, default=None, help="override destination directory")
    parser.add_argument("--retry", type=int, default=DEFAULT_RETRIES, help=f"download retries (default {DEFAULT_RETRIES})")
    args = parser.parse_args()

    dest = args.dest or get_default_models_dir()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Target models directory: {dest}")
    retries = max(1, args.retry)

    # Core required models
    # 1. Offline Parakeet (Multilingual: en, es, pt, etc.)
    install_tarball(ASR_URL, "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2", dest, retries=retries)
    # 2. Streaming Nemotron
    install_tarball(ASR_URL, "sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25.tar.bz2", dest, retries=retries)
    # 3. Silero VAD
    install_raw_file(ASR_URL, "silero_vad.onnx", dest, retries=retries)
    # 4. Default DPDFNet8 denoiser
    install_raw_file(ENH_URL, "dpdfnet8.onnx", dest, retries=retries)

    if args.all:
        # Extra fast streaming model: Kroko
        install_tarball(ASR_URL, "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06.tar.bz2", dest, retries=retries)
        # Extra denoisers: DPDFNet2, DPDFNet4, GTCRN simple
        for extra in ("dpdfnet2.onnx", "dpdfnet4.onnx", "gtcrn_simple.onnx"):
            install_raw_file(ENH_URL, extra, dest, retries=retries)

    if args.translation:
        install_translation_model(dest, retries=retries)

    print("\nAll requested models installed successfully.")


if __name__ == "__main__":
    main()
