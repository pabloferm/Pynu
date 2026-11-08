#!/usr/bin/env python3
"""Scan command line; same as ``python -m pynu.binned.scan``."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pynu.binned.scan import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
