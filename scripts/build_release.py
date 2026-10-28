#!/usr/bin/env python3
"""Build and check distributable packages (wheel and sdist) for PyPI."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd, **kwargs):
    print(f"\n[build_release] Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=ROOT, **kwargs)
    if res.returncode != 0:
        sys.exit(f"[build_release] Error: command failed with code {res.returncode}")
    return res


def main():
    print("=" * 65)
    print("  talk-to-typed / stt: PyPI Build & Package Release Tool")
    print("=" * 65)

    # 1. Clean dist/ and build/
    dist_dir = ROOT / "dist"
    build_dir = ROOT / "build"
    egg_info = list(ROOT.glob("*.egg-info"))
    if dist_dir.exists():
        print(f"Cleaning {dist_dir} ...")
        shutil.rmtree(dist_dir)
    if build_dir.exists():
        print(f"Cleaning {build_dir} ...")
        shutil.rmtree(build_dir)
    for ei in egg_info:
        if ei.exists():
            shutil.rmtree(ei)

    # 2. Build sdist and wheel
    run([sys.executable, "-m", "build"])

    # 3. List built artifacts
    artifacts = list(dist_dir.glob("*"))
    print("\n[build_release] Successfully generated packages:")
    for a in sorted(artifacts):
        size_kb = a.stat().st_size / 1024
        print(f"  • {a.name} ({size_kb:.1f} KB)")

    # 4. Twine check if available
    try:
        import twine
        print("\n[build_release] Running twine check on dist artifacts...")
        run([sys.executable, "-m", "twine", "check", "dist/*"])
    except ImportError:
        print("\n[build_release] Note: 'twine' is not installed. To check artifacts before upload:")
        print("  pip install twine && twine check dist/*")

    print("\n" + "=" * 65)
    print("  Release Build Complete!")
    print("=" * 65)
    print("To publish to TestPyPI:")
    print("  twine upload --repository testpypi dist/*\n")
    print("To publish to Production PyPI:")
    print("  twine upload dist/*\n")


if __name__ == "__main__":
    main()
