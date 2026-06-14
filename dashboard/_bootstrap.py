"""Ensure the repo root is on sys.path for Streamlit Cloud page scripts.

Streamlit adds the main script directory (``dashboard/``) to sys.path, so
``from dashboard.components...`` imports fail unless the repository root is
also present. Import this module before any ``dashboard.*`` imports:

    import _bootstrap  # noqa: F401
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_root_str = str(_ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)
