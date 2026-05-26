"""App startup: logging and import path for Streamlit Cloud."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("scfm_dashboard")

_APP_DIR = Path(__file__).resolve().parent
_APP_DIR_STR = str(_APP_DIR)
if _APP_DIR_STR not in sys.path:
    sys.path.insert(0, _APP_DIR_STR)

# Streamlit Cloud may install an older `metrics-dashboard` wheel into site-packages.
# Drop any cached imports so the in-repo package under this app directory wins.
_PKG = "metrics_dashboard"
for name in list(sys.modules):
    if name == _PKG or name.startswith(f"{_PKG}."):
        del sys.modules[name]

log.info("bootstrap: app_dir=%s cwd=%s", _APP_DIR, Path.cwd())
print(f"[scfm_dashboard] bootstrap ok app_dir={_APP_DIR} cwd={Path.cwd()}", flush=True)
