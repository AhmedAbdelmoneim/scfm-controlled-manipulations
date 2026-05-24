"""Load and enrich evaluation metrics CSVs."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import streamlit as st

from metrics_dashboard.config import (
    MODEL_ORDER,
    PARAM_KEYS,
    evaluation_dir,
    manipulations_dir,
    reference_h5ad_path,
)


def _eval_cache_key(dataset_id: str, models: tuple[str, ...], root: Path) -> str:
    ev = evaluation_dir(dataset_id, root)
    parts = [str(root), dataset_id, ",".join(models)]
    if ev.is_dir():
        for p in sorted(ev.glob("*_metrics.csv")):
            if p.stem.removesuffix("_metrics") in models:
                parts.append(f"{p.name}:{p.stat().st_mtime_ns}")
    return "|".join(parts)


def load_intervention_params(
    manip_dir: Path,
    intervention_ids: list[str],
    id_to_name: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for iid in intervention_ids:
        h5ad_path = manip_dir / f"{iid}.h5ad"
        if not h5ad_path.is_file():
            continue
        name = id_to_name.get(iid, iid.split("_")[0])
        adata = ad.read_h5ad(h5ad_path, backed="r")
        params = adata.uns.get("scfm_intervention", {}).get(name, {})
        key = PARAM_KEYS.get(name)
        value = params.get(key) if key else None
        if value is None and params:
            scalar = {
                k: v
                for k, v in params.items()
                if not isinstance(v, (list, np.ndarray))
                or (hasattr(v, "__len__") and len(v) < 10)
            }
            if len(scalar) == 1:
                key, value = next(iter(scalar.items()))
            else:
                key, value = "intervention_id", iid
        rows.append(
            {
                "intervention_id": iid,
                "intervention_name": name,
                "param_key": key,
                "param_value": value,
            }
        )
        adata.file.close()
    return pd.DataFrame(rows)


def load_dataset_metrics(
    dataset_id: str,
    models: list[str],
    root: Path,
) -> pd.DataFrame:
    """Load and concat per-model CSVs; join intervention sweep parameters."""
    ev = evaluation_dir(dataset_id, root)
    frames: list[pd.DataFrame] = []
    for model in models:
        path = ev / f"{model}_metrics.csv"
        if path.is_file():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()

    metrics_df = pd.concat(frames, ignore_index=True)
    metrics_df["model"] = pd.Categorical(
        metrics_df["model"], categories=MODEL_ORDER, ordered=True
    )

    id_to_name = (
        metrics_df[["intervention_id", "intervention_name"]]
        .drop_duplicates()
        .set_index("intervention_id")["intervention_name"]
        .to_dict()
    )
    params_df = load_intervention_params(
        manipulations_dir(dataset_id, root),
        metrics_df["intervention_id"].drop_duplicates().tolist(),
        id_to_name,
    )
    if not params_df.empty:
        metrics_df = metrics_df.merge(
            params_df, on=["intervention_id", "intervention_name"], how="left"
        )
    return metrics_df


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
    """Concat one model's metrics across datasets (for model card)."""
    frames: list[pd.DataFrame] = []
    for ds in dataset_ids:
        path = evaluation_dir(ds, root) / f"{model}_metrics.csv"
        if path.is_file():
            df = pd.read_csv(path)
            if "dataset_id" not in df.columns or df["dataset_id"].isna().all():
                df["dataset_id"] = ds
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["model"] = pd.Categorical(out["model"], categories=MODEL_ORDER, ordered=True)
    return out


def _cross_dataset_cache_key(model: str, dataset_ids: tuple[str, ...], root: Path) -> str:
    parts = [str(root), model, ",".join(dataset_ids)]
    for ds in dataset_ids:
        path = evaluation_dir(ds, root) / f"{model}_metrics.csv"
        if path.is_file():
            parts.append(f"{path}:{path.stat().st_mtime_ns}")
    return "|".join(parts)


def load_dataset_summary(
    dataset_id: str,
    root: Path,
    *,
    cell_type_col: str = "cell_type",
    batch_col: str = "batch",
) -> dict[str, int | str | float]:
    """Summary stats from reference.h5ad."""
    path = reference_h5ad_path(dataset_id, root)
    if not path.is_file():
        return {
            "dataset_id": dataset_id,
            "n_cells": 0,
            "n_genes": 0,
            "n_cell_types": 0,
            "n_batches": 0,
            "error": "reference.h5ad not found",
        }
    adata = ad.read_h5ad(path, backed="r")
    n_cells = int(adata.n_obs)
    n_genes = int(adata.n_vars)
    n_ct = (
        int(adata.obs[cell_type_col].nunique())
        if cell_type_col in adata.obs.columns
        else 0
    )
    n_batches = (
        int(adata.obs[batch_col].nunique()) if batch_col in adata.obs.columns else 0
    )
    adata.file.close()
    return {
        "dataset_id": dataset_id,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "n_cell_types": n_ct,
        "n_batches": n_batches,
    }


@st.cache_data(show_spinner="Loading dataset summary…")
def load_dataset_summary_cached(
    dataset_id: str,
    root_str: str,
    ref_mtime_ns: int,
) -> dict[str, int | str | float]:
    return load_dataset_summary(dataset_id, Path(root_str))


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


@st.cache_data(show_spinner="Loading cross-dataset metrics…")
def load_model_metrics_across_datasets_cached(
    model: str,
    dataset_ids: tuple[str, ...],
    root_str: str,
    cache_version: str,
) -> pd.DataFrame:
    root = Path(root_str)
    return load_model_metrics_across_datasets(model, list(dataset_ids), root)
