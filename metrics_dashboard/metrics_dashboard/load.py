"""Load dashboard metrics from Parquet bundles or legacy SCEval trees."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from metrics_dashboard.bundle import (
    METRICS_FILENAME,
    SUMMARY_FILENAME,
    bundle_metrics_path,
    bundle_summary_path,
    extract_dataset_summary,
    is_legacy_dataset_dir,
    load_metrics_table,
)
from metrics_dashboard.config import evaluation_dir


def _eval_cache_key(dataset_id: str, models: tuple[str, ...], root: Path) -> str:
    bundle = bundle_metrics_path(root, dataset_id)
    if bundle.is_file():
        return f"{root}|{dataset_id}|{','.join(models)}|{bundle.stat().st_mtime_ns}"

    ev = evaluation_dir(dataset_id, root)
    parts = [str(root), dataset_id, ",".join(models)]
    if ev.is_dir():
        for p in sorted(ev.glob("*_metrics.csv")):
            if p.stem.removesuffix("_metrics") in models:
                parts.append(f"{p.name}:{p.stat().st_mtime_ns}")
    return "|".join(parts)


def load_dataset_metrics(
    dataset_id: str,
    models: list[str],
    root: Path,
) -> pd.DataFrame:
    return load_metrics_table(dataset_id, root, models)


@st.cache_data(show_spinner="Loading metrics…")
def load_dataset_metrics_cached(
    dataset_id: str,
    models: tuple[str, ...],
    root_str: str,
    cache_version: str,
) -> pd.DataFrame:
    root = Path(root_str)
    return load_dataset_metrics(dataset_id, list(models), root)


def load_model_metrics_across_datasets(
    model: str,
    dataset_ids: list[str],
    root: Path,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ds in dataset_ids:
        df = load_metrics_table(ds, root, [model])
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner="Loading cross-dataset metrics…")
def load_model_metrics_across_datasets_cached(
    model: str,
    dataset_ids: tuple[str, ...],
    root_str: str,
    cache_version: str,
) -> pd.DataFrame:
    root = Path(root_str)
    return load_model_metrics_across_datasets(model, list(dataset_ids), root)


def load_dataset_summary(
    dataset_id: str,
    root: Path,
    *,
    cell_type_col: str = "cell_type",
    batch_col: str = "batch",
) -> dict[str, int | str | float]:
    summary_path = bundle_summary_path(root, dataset_id)
    if summary_path.is_file():
        import json

        return json.loads(summary_path.read_text())

    legacy = root / dataset_id
    if is_legacy_dataset_dir(legacy):
        return extract_dataset_summary(
            dataset_id, legacy, cell_type_col=cell_type_col, batch_col=batch_col
        )

    return {
        "dataset_id": dataset_id,
        "n_cells": 0,
        "n_genes": 0,
        "n_cell_types": 0,
        "n_batches": 0,
        "error": "summary not found (export bundle or provide reference.h5ad)",
    }


@st.cache_data(show_spinner="Loading dataset summary…")
def load_dataset_summary_cached(
    dataset_id: str,
    root_str: str,
    ref_mtime_ns: int,
) -> dict[str, int | str | float]:
    root = Path(root_str)
    summary_path = bundle_summary_path(root, dataset_id)
    if summary_path.is_file():
        ref_mtime_ns = summary_path.stat().st_mtime_ns
    return load_dataset_summary(dataset_id, root)


def load_multi_dataset_metrics(
    dataset_ids: list[str],
    models: list[str],
    root: Path,
) -> pd.DataFrame:
    frames = [load_dataset_metrics(ds, models, root) for ds in dataset_ids]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner="Loading metrics…")
def load_multi_dataset_metrics_cached(
    dataset_ids: tuple[str, ...],
    models: tuple[str, ...],
    root_str: str,
    cache_version: str,
) -> pd.DataFrame:
    root = Path(root_str)
    return load_multi_dataset_metrics(list(dataset_ids), list(models), root)
