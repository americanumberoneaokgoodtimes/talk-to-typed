#!/usr/bin/env python3
"""Convenience alias for scripts/install_models.py."""
import sys
from pathlib import Path

# Add project root to sys.path so scripts.install_models can be imported
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    from scripts.install_models import *
except ImportError:
    # If installed as site-packages without scripts/
    from scripts.install_models import main

if __name__ == "__main__":
    main()
