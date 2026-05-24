"""App startup: logging, headless matplotlib, and import path for Streamlit Cloud."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Headless backend must be set before any other module imports pyplot.
import matplotlib

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("scfm_dashboard")

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

log.info("bootstrap: app_dir=%s cwd=%s", _APP_DIR, Path.cwd())
print(f"[scfm_dashboard] bootstrap ok app_dir={_APP_DIR} cwd={Path.cwd()}", flush=True)
