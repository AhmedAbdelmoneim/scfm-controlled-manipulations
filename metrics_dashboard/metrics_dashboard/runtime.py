"""Runtime diagnostics for deployment debugging."""

from __future__ import annotations

import logging
import platform
import sys
from pathlib import Path

from metrics_dashboard.config import bundle_root

log = logging.getLogger("scfm_dashboard.runtime")


def log_startup_context() -> None:
    root = bundle_root()
    datasets = sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / "metrics.parquet").is_file()
    ) if root.is_dir() else []
    log.info("python=%s platform=%s", sys.version.split()[0], platform.platform())
    log.info("cwd=%s", Path.cwd())
    log.info("bundle_root=%s exists=%s", root, root.is_dir())
    log.info("datasets=%s", datasets)
    if root.is_dir() and not datasets:
        log.warning("bundle_root exists but no */metrics.parquet found")
        log.info("bundle_root children=%s", [p.name for p in root.iterdir()])
