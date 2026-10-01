"""Pytest configuration: make the ``mar`` package importable from ``src``.

The project uses a ``src`` layout, so the package lives at ``src/mardynamics``
instead of at the project root. Installing the project (``pip install -e .``)
puts it on the path permanently. Adding ``src`` here as well means the test
suite also runs straight from a fresh clone, before anything has been
installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
