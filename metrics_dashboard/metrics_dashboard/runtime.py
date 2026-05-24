"""Runtime diagnostics for deployment debugging."""

from __future__ import annotations

import logging
import platform
import sys
from pathlib import Path

from metrics_dashboard.config import BUNDLE_ROOT, bundle_root

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


def bundle_diagnostics() -> dict:
    root = bundle_root()
    out: dict = {
        "bundle_root": str(root),
        "bundle_root_exists": root.is_dir(),
        "cwd": str(Path.cwd()),
    }
    if not root.is_dir():
        out["error"] = "data/dashboard_bundles not found — commit bundles to repo root"
        return out
    datasets = []
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        pq = p / "metrics.parquet"
        datasets.append(
            {
                "dataset": p.name,
                "metrics_parquet": pq.is_file(),
                "size_mb": round(pq.stat().st_size / 1e6, 2) if pq.is_file() else None,
            }
        )
    out["datasets"] = datasets
    return out
