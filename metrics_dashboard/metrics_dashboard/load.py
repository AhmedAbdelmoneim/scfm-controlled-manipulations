"""Load metrics and summaries from checked-in Parquet bundles."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from metrics_dashboard.bundle import METRICS_FILENAME, bundle_metrics_path, bundle_summary_path
from metrics_dashboard.config import MODEL_ORDER, bundle_root

log = logging.getLogger("scfm_dashboard.load")


def _cache_key(dataset_ids: tuple[str, ...], models: tuple[str, ...], root: Path) -> str:
    parts = [str(root), ",".join(dataset_ids), ",".join(models)]
    for ds in dataset_ids:
        path = bundle_metrics_path(root, ds)
        if path.is_file():
            parts.append(f"{ds}:{path.stat().st_mtime_ns}")
    return "|".join(parts)


def load_metrics(dataset_id: str, models: list[str], root: Path | None = None) -> pd.DataFrame:
    base = root or bundle_root()
    path = bundle_metrics_path(base, dataset_id)
    if not path.is_file():
        log.warning("missing metrics parquet: %s", path)
        return pd.DataFrame()
    t0 = time.perf_counter()
    log.info("reading parquet %s (%.2f MB)", path, path.stat().st_size / 1e6)
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        log.exception("failed to read parquet %s", path)
        raise RuntimeError(
            f"Could not read {path.name} ({exc}). "
            "Re-export with `make export-dashboard-bundle ... --compression snappy` if ZSTD is unsupported."
        ) from exc
    if models:
        df = df[df["model"].astype(str).isin(models)]
    df["model"] = pd.Categorical(df["model"].astype(str), categories=MODEL_ORDER, ordered=True)
    log.info("loaded %s: %d rows in %.2fs", dataset_id, len(df), time.perf_counter() - t0)
    return df


def load_dataset_summary(dataset_id: str, root: Path | None = None) -> dict:
    base = root or bundle_root()
    path = bundle_summary_path(base, dataset_id)
    if path.is_file():
        return json.loads(path.read_text())
    return {
        "dataset_id": dataset_id,
        "n_cells": 0,
        "n_genes": 0,
        "n_cell_types": 0,
        "n_batches": 0,
        "error": "summary.json missing — re-export bundle",
    }


@st.cache_data(show_spinner="Loading metrics…")
def load_metrics_cached(
    dataset_ids: tuple[str, ...],
    models: tuple[str, ...],
    root_str: str,
    cache_version: str,
) -> pd.DataFrame:
    log.info("cache load datasets=%s models=%s root=%s", dataset_ids, models, root_str)
    root = Path(root_str)
    frames = [load_metrics(ds, list(models), root) for ds in dataset_ids]
    frames = [f for f in frames if not f.empty]
    if not frames:
        log.warning("no metrics frames loaded")
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    log.info("concatenated %d rows", len(out))
    return out


def load_multi_dataset_metrics(
    dataset_ids: list[str],
    models: list[str],
    root: Path | None = None,
) -> pd.DataFrame:
    base = root or bundle_root()
    version = _cache_key(tuple(sorted(dataset_ids)), tuple(sorted(models)), base)
    return load_metrics_cached(
        tuple(sorted(dataset_ids)),
        tuple(sorted(models)),
        str(base),
        version,
    )
