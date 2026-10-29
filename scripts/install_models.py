#!/usr/bin/env python3
"""Cross-platform model installer for stt (Linux, Android Termux/Kali, Windows).

Downloads and unpacks required models into the platform-appropriate models directory:
  - Windows: %LOCALAPPDATA%\\stt\\models (or %APPDATA%\\stt\\models)
  - Linux/Android: ~/.local/share/stt/models

Usage:
  python scripts/install_models.py [--all] [--dest DIR]
"""
import argparse
import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ASR_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
ENH_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models"


def get_default_models_dir() -> Path:
    env_dir = os.environ.get("STT_MODELS")
    if env_dir:
        return Path(env_dir)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or (Path.home() / "AppData" / "Local")
        return Path(base) / "stt" / "models"
    return Path.home() / ".local" / "share" / "stt" / "models"


def download_file(url: str, dest_path: Path, desc: str):
    print(f"Downloading {desc}...")
    temp_file = dest_path.with_suffix(dest_path.suffix + ".download")
    try:
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
    except Exception as e:
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download {url}: {e}")


def install_tarball(url: str, filename: str, target_dir: Path):
    folder_name = filename.replace(".tar.bz2", "")
    final_folder = target_dir / folder_name
    if final_folder.is_dir() and any(final_folder.glob("*.onnx")):
        print(f"Already installed: {folder_name}")
        return

    tar_path = target_dir / filename
    download_file(f"{url}/{filename}", tar_path, filename)

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


def install_raw_file(url: str, filename: str, target_dir: Path):
    final_path = target_dir / filename
    if final_path.is_file() and final_path.stat().st_size > 0:
        print(f"Already installed: {filename}")
        return
    download_file(f"{url}/{filename}", final_path, filename)
    print(f"OK: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Install speech-to-text models")
    parser.add_argument("--all", action="store_true", help="install all extra models (kroko, gtcrn, extra denoisers)")
    parser.add_argument("--dest", type=Path, default=None, help="override destination directory")
    args = parser.parse_args()

    dest = args.dest or get_default_models_dir()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Target models directory: {dest}")

    # Core required models
    # 1. Offline Parakeet (Multilingual: en, es, pt, etc.)
    install_tarball(ASR_URL, "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2", dest)
    # 2. Streaming Nemotron
    install_tarball(ASR_URL, "sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25.tar.bz2", dest)
    # 3. Silero VAD
    install_raw_file(ASR_URL, "silero_vad.onnx", dest)
    # 4. Default DPDFNet8 denoiser
    install_raw_file(ENH_URL, "dpdfnet8.onnx", dest)

    if args.all:
        # Extra fast streaming model: Kroko
        install_tarball(ASR_URL, "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06.tar.bz2", dest)
        # Extra denoisers: DPDFNet2, DPDFNet4, GTCRN simple
        for extra in ("dpdfnet2.onnx", "dpdfnet4.onnx", "gtcrn_simple.onnx"):
            install_raw_file(ENH_URL, extra, dest)

    print("\nAll requested models installed successfully.")


if __name__ == "__main__":
    main()
