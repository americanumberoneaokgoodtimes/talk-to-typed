#!/usr/bin/env python3
"""Cross-platform model installer for stt (Linux, Android Termux/Kali, Windows).

Downloads and unpacks required models into the platform-appropriate models directory:
  - Windows: %LOCALAPPDATA%\\stt\\models (or %APPDATA%\\stt\\models)
  - Linux/Android: ~/.local/share/stt/models

Usage:
  python scripts/install_models.py [--all] [--dest DIR] [--retry N]
"""
import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

ASR_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
ENH_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models"

# Max retries for failed downloads
DEFAULT_RETRIES = 3
RETRY_DELAY = 2  # seconds


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _file_size_mb(path: Path) -> str:
    return f"{path.stat().st_size / (1024 * 1024):.1f} MB" if path.exists() else "0.0 MB"


def download_file(url: str, dest_path: Path, desc: str, retries: int = DEFAULT_RETRIES) -> None:
    """Download a file with progress reporting and retry on failure."""
    temp_file = dest_path.with_suffix(dest_path.suffix + ".download")

    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading {desc} (attempt {attempt}/{retries})...")

            def reporthook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, block_num * block_size * 100 // total_size)
                    mb = (block_num * block_size) / (1024 * 1024)
                    tot_mb = total_size / (1024 * 1024)
                    bar = "=" * (percent // 2) + " " * (50 - percent // 2)
                    sys.stdout.write(f"\r  [{percent:3d}%] [{bar}] {mb:.1f} MB / {tot_mb:.1f} MB")
                    sys.stdout.flush()

            urllib.request.urlretrieve(url, str(temp_file), reporthook=reporthook)
            print()
            temp_file.replace(dest_path)
            print(f"  OK: {desc} ({_file_size_mb(dest_path)})")
            return
        except Exception as e:
            print(f"  Attempt {attempt} failed: {e}")
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            if attempt < retries:
                print(f"  Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"failed to download {url} after {retries} attempts: {e}")


# ---------------------------------------------------------------------------
# Install helpers
# ---------------------------------------------------------------------------

def is_valid_onnx(path: Path) -> bool:
    """Quick sanity check: file starts with ONNX magic bytes."""
    try:
        with path.open("rb") as f:
            header = f.read(4)
        return header == b"\x7fONNX" or header == b"ONNX"[:4]
    except OSError:
        return False


def install_tarball(url: str, filename: str, target_dir: Path, retries: int = DEFAULT_RETRIES) -> None:
    folder_name = filename.replace(".tar.bz2", "")
    final_folder = target_dir / folder_name

    # Skip if already complete and valid
    if final_folder.is_dir():
        onnx_files = list(final_folder.glob("*.onnx"))
        tokens_file = final_folder / "tokens.txt"
        if onnx_files and tokens_file.is_file():
            if all(is_valid_onnx(f) for f in onnx_files):
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

        # Verify extraction produced valid files
        if not final_folder.is_dir():
            raise RuntimeError(f"Expected folder {folder_name} not found after extraction")
        onnx_files = list(final_folder.glob("*.onnx"))
        if not onnx_files:
            raise RuntimeError(f"No .onnx files found in {folder_name}")
        print(f"OK: {folder_name} ({len(onnx_files)} model file(s))")
    finally:
        if tar_path.exists():
            tar_path.unlink(missing_ok=True)


def install_raw_file(url: str, filename: str, target_dir: Path, retries: int = DEFAULT_RETRIES) -> None:
    final_path = target_dir / filename

    # Skip if already present and valid
    if final_path.is_file() and final_path.stat().st_size > 0:
        if is_valid_onnx(final_path):
            print(f"Already installed: {filename}")
            return

    download_file(f"{url}/{filename}", final_path, filename, retries=retries)

    # Verify the downloaded file looks like ONNX
    if not is_valid_onnx(final_path):
        raise RuntimeError(f"Downloaded file {filename} is not a valid ONNX model")
    print(f"OK: {filename} ({_file_size_mb(final_path)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Install speech-to-text models")
    parser.add_argument("--all", action="store_true",
                        help="install all extra models (kroko, gtcrn, extra denoisers)")
    parser.add_argument("--dest", type=Path, default=None,
                        help="override destination directory")
    parser.add_argument("--retry", type=int, default=DEFAULT_RETRIES,
                        help=f"number of download retries (default {DEFAULT_RETRIES})")
    args = parser.parse_args()

    dest = args.dest
    if dest is None:
        env_dir = os.environ.get("STT_MODELS")
        if env_dir:
            dest = Path(env_dir)
        elif sys.platform == "win32":
            base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or (Path.home() / "AppData" / "Local")
            dest = Path(base) / "stt" / "models"
        else:
            dest = Path.home() / ".local" / "share" / "stt" / "models"

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Target models directory: {dest}")
    print()

    retries = max(1, args.retry)

    installed = []

    # ------------------------------------------------------------------
    # Core required models
    # ------------------------------------------------------------------

    # 1. Offline Parakeet (Multilingual: en, es, pt, etc.)
    install_tarball(
        ASR_URL,
        "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2",
        dest,
        retries=retries,
    )
    installed.append("sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 (offline ASR)")

    # 2. Streaming Nemotron
    install_tarball(
        ASR_URL,
        "sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25.tar.bz2",
        dest,
        retries=retries,
    )
    installed.append("sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8 (streaming ASR)")

    # 3. Silero VAD
    install_raw_file(ASR_URL, "silero_vad.onnx", dest, retries=retries)
    installed.append("silero_vad.onnx (VAD)")

    # 4. Default DPDFNet8 denoiser
    install_raw_file(ENH_URL, "dpdfnet8.onnx", dest, retries=retries)
    installed.append("dpdfnet8.onnx (default denoiser)")

    # ------------------------------------------------------------------
    # Extra models (--all)
    # ------------------------------------------------------------------

    if args.all:
        # Extra fast streaming model: Kroko
        install_tarball(
            ASR_URL,
            "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06.tar.bz2",
            dest,
            retries=retries,
        )
        installed.append("sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06 (streaming ASR)")

        # Extra denoisers used by subnoise mode
        for extra in ("dpdfnet2.onnx", "dpdfnet4.onnx", "gtcrn_simple.onnx"):
            install_raw_file(ENH_URL, extra, dest, retries=retries)
            installed.append(f"{extra} (denoiser)")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print()
    print("=" * 60)
    print("Installation complete.")
    print(f"Models directory: {dest}")
    print("Installed models:")
    for item in installed:
        print(f"  • {item}")
    print("=" * 60)


if __name__ == "__main__":
    main()
