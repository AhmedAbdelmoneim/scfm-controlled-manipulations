"""Put the app directory on sys.path so ``import metrics_dashboard`` resolves correctly.

Streamlit Cloud adds the repository root to sys.path. That makes Python treat the
outer ``metrics_dashboard/`` app folder as a namespace package and breaks
``from metrics_dashboard.catalog import ...`` with KeyError. Prepend this app dir first.
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
