"""Load metrics and summaries from checked-in Parquet bundles."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from metrics_dashboard.bundle import METRICS_FILENAME, bundle_metrics_path, bundle_summary_path
from metrics_dashboard.config import MODEL_ORDER, bundle_root


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
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if models:
        df = df[df["model"].astype(str).isin(models)]
    df["model"] = pd.Categorical(df["model"].astype(str), categories=MODEL_ORDER, ordered=True)
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
    root = Path(root_str)
    frames = [load_metrics(ds, list(models), root) for ds in dataset_ids]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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
