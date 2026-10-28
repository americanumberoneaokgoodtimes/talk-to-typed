#!/usr/bin/env python3
"""Standalone binary build helper for talk-to-typed / stt using PyInstaller.

Bundles Python interpreter, sherpa-onnx, numpy, soundfile, and the stt package
into a self-contained executable.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    print("=" * 65)
    print("  talk-to-typed / stt: Standalone Executable Builder")
    print("=" * 65)

    pyinstaller = shutil.which("pyinstaller") or (
        str(Path(sys.executable).parent / "pyinstaller")
        if (Path(sys.executable).parent / "pyinstaller").exists()
        else None
    )

    if not pyinstaller:
        print("[build_standalone] PyInstaller is not installed in the current environment.")
        print("To install PyInstaller and create a standalone runner:")
        print(f"  {sys.executable} -m pip install pyinstaller")
        print("\nOnce installed, run this script again:")
        print("  python scripts/build_standalone.py\n")
        return

    entry_point = ROOT / "stt" / "cli.py"
    dist_dir = ROOT / "dist_standalone"

    cmd = [
        pyinstaller,
        "--name=stt",
        "--onefile",
        "--clean",
        f"--distpath={dist_dir}",
        f"--workpath={ROOT / 'build_standalone'}",
        "--collect-all=sherpa_onnx",
        "--collect-all=soundfile",
        str(entry_point),
    ]

    print(f"\n[build_standalone] Building standalone executable into {dist_dir} ...")
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode == 0:
        exe_path = dist_dir / ("stt.exe" if sys.platform == "win32" else "stt")
        print("\n" + "=" * 65)
        print("  Standalone Build Succeeded!")
        print(f"  Binary location: {exe_path}")
        print("=" * 65 + "\n")
    else:
        sys.exit(res.returncode)


if __name__ == "__main__":
    main()
