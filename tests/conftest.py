"""Shared pytest configuration.

Adds the repository root to sys.path so that ``import fft`` works when
pytest is launched from any directory.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
