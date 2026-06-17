"""Minimal dashboard bundles (Parquet + JSON) without manipulation h5ads."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from metrics_dashboard.config import MODEL_ORDER, PARAM_KEYS
from metrics_dashboard.obs_columns import (
    resolve_batch_column,
    resolve_cell_type_column_for_dataset,
)

METRICS_FILENAME = "metrics.parquet"
SUMMARY_FILENAME = "summary.json"
MANIFEST_FILENAME = "manifest.json"


def coerce_metrics_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes so Parquet can store mixed sweep axes (numeric + categorical)."""
    out = df.copy()
    if "param_value" in out.columns:
        out["param_value"] = out["param_value"].map(_param_value_as_str).astype("string")
    for col in out.columns:
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].astype(str)
    return out


def _param_value_as_str(value: object) -> str | pd.NA:
    if value is None:
        return pd.NA
    if isinstance(value, float) and np.isnan(value):
        return pd.NA
    return str(value)


def dataset_bundle_dir(artifacts_root: Path, dataset_id: str) -> Path:
    return artifacts_root / dataset_id


def bundle_metrics_path(artifacts_root: Path, dataset_id: str) -> Path:
    return dataset_bundle_dir(artifacts_root, dataset_id) / METRICS_FILENAME


def bundle_summary_path(artifacts_root: Path, dataset_id: str) -> Path:
    return dataset_bundle_dir(artifacts_root, dataset_id) / SUMMARY_FILENAME


def is_legacy_dataset_dir(path: Path) -> bool:
    """Full SCEval tree with ``results/evaluation``."""
    return (path / "results" / "evaluation").is_dir()


def is_bundle_dataset_dir(path: Path) -> bool:
    """Exported minimal bundle."""
    return (path / METRICS_FILENAME).is_file()


def resolve_dataset_dirs(source: Path) -> list[tuple[str, Path]]:
    """Return ``(dataset_id, dataset_root)`` pairs to export or load."""
    source = source.resolve()
    if is_legacy_dataset_dir(source) or is_bundle_dataset_dir(source):
        return [(source.name, source)]
    if not source.is_dir():
        return []
    out: list[tuple[str, Path]] = []
    for child in sorted(source.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if is_legacy_dataset_dir(child) or is_bundle_dataset_dir(child):
            out.append((child.name, child))
    return out


def extract_intervention_params(
    manip_dir: Path,
    intervention_ids: list[str],
    id_to_name: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for iid in intervention_ids:
        h5ad_path = manip_dir / f"{iid}.h5ad"
        if not h5ad_path.is_file():
            continue
        name = id_to_name.get(iid, iid.split("_")[0])
        import anndata as ad

        adata = ad.read_h5ad(h5ad_path, backed="r")
        try:
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
        finally:
            adata.file.close()
    return pd.DataFrame(rows)


def extract_dataset_summary(
    dataset_id: str,
    dataset_root: Path,
    *,
    cell_type_col: str = "cell_type",
    batch_col: str = "batch",
) -> dict[str, Any]:
    """Summary from reference.h5ad (legacy layout) or existing summary.json."""
    bundle_summary = dataset_root / SUMMARY_FILENAME
    if is_bundle_dataset_dir(dataset_root) and bundle_summary.is_file():
        return json.loads(bundle_summary.read_text())

    if is_legacy_dataset_dir(dataset_root):
        ref_path = dataset_root / "results" / "manipulations" / "reference.h5ad"
    else:
        ref_path = dataset_root / "reference.h5ad"

    if not ref_path.is_file():
        return {
            "dataset_id": dataset_id,
            "n_cells": 0,
            "n_genes": 0,
            "n_cell_types": 0,
            "n_batches": 0,
            "error": "reference.h5ad not found",
        }

    import anndata as ad

    adata = ad.read_h5ad(ref_path, backed="r")
    try:
        obs_cols = adata.obs.columns
        resolved_cell_type = resolve_cell_type_column_for_dataset(
            obs_cols,
            cell_type_col,
            dataset_id=dataset_id,
        )
        resolved_batch = resolve_batch_column(obs_cols, batch_col)
        out: dict[str, Any] = {
            "dataset_id": dataset_id,
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "n_cell_types": (
                int(adata.obs[resolved_cell_type].nunique())
                if resolved_cell_type is not None
                else 0
            ),
            "n_batches": (
                int(adata.obs[resolved_batch].nunique())
                if resolved_batch is not None
                else 0
            ),
        }
        if resolved_cell_type is not None:
            out["cell_type_column"] = resolved_cell_type
        if resolved_batch is not None:
            out["batch_column"] = resolved_batch
        if cell_type_col and resolved_cell_type != cell_type_col:
            out["cell_type_column_configured"] = cell_type_col
        return out
    finally:
        adata.file.close()


def load_metrics_from_legacy(
    dataset_id: str,
    dataset_root: Path,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Load CSV metrics and join sweep params from manipulation h5ads."""
    ev = dataset_root / "results" / "evaluation"

    csv_paths = sorted(
        p for p in ev.glob("*_metrics.csv") if "_scib_metrics" not in p.stem
    )
    scib_paths = sorted(ev.glob("*_scib_metrics.csv"))
    if models is not None:
        model_set = set(models)
        csv_paths = [p for p in csv_paths if p.stem.removesuffix("_metrics") in model_set]
        scib_paths = [
            p for p in scib_paths if p.stem.removesuffix("_scib_metrics") in model_set
        ]

    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        frames.append(pd.read_csv(path))
    for path in scib_paths:
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()

    metrics_df = pd.concat(frames, ignore_index=True)
    if "dataset_id" not in metrics_df.columns or metrics_df["dataset_id"].isna().all():
        metrics_df["dataset_id"] = dataset_id

    metrics_df["model"] = pd.Categorical(
        metrics_df["model"].astype(str), categories=MODEL_ORDER, ordered=True
    )

    id_to_name = (
        metrics_df[["intervention_id", "intervention_name"]]
        .drop_duplicates()
        .set_index("intervention_id")["intervention_name"]
        .to_dict()
    )
    manip = dataset_root / "results" / "manipulations"

    params_df = extract_intervention_params(
        manip,
        metrics_df["intervention_id"].drop_duplicates().tolist(),
        id_to_name,
    )
    if not params_df.empty:
        metrics_df = metrics_df.merge(
            params_df, on=["intervention_id", "intervention_name"], how="left"
        )
    return metrics_df


def load_metrics_table(
    dataset_id: str,
    artifacts_root: Path,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Load metrics from a Parquet bundle or legacy CSV tree."""
    bundle_path = bundle_metrics_path(artifacts_root, dataset_id)
    if bundle_path.is_file():
        df = pd.read_parquet(bundle_path)
        if models is not None:
            df = df[df["model"].astype(str).isin(models)]
        df["model"] = pd.Categorical(
            df["model"].astype(str), categories=MODEL_ORDER, ordered=True
        )
        return df

    legacy_root = artifacts_root / dataset_id
    if is_legacy_dataset_dir(legacy_root):
        return load_metrics_from_legacy(dataset_id, legacy_root, models)

    return pd.DataFrame()


def build_metrics_table(
    dataset_id: str,
    dataset_root: Path,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Build enriched metrics frame from a legacy SCEval dataset directory."""
    if is_bundle_dataset_dir(dataset_root):
        df = pd.read_parquet(dataset_root / METRICS_FILENAME)
        if models is not None:
            df = df[df["model"].astype(str).isin(models)]
        return df
    return load_metrics_from_legacy(dataset_id, dataset_root, models)


def export_dataset_bundle(
    dataset_id: str,
    dataset_root: Path,
    output_root: Path,
    *,
    compression: str = "zstd",
) -> Path:
    """Export one dataset to a minimal Parquet + JSON bundle."""
    metrics_df = build_metrics_table(dataset_id, dataset_root)
    if metrics_df.empty:
        raise FileNotFoundError(
            f"No metrics CSVs under {dataset_root}/results/evaluation"
        )

    out_dir = dataset_bundle_dir(output_root, dataset_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / METRICS_FILENAME
    coerce_metrics_for_parquet(metrics_df).to_parquet(
        metrics_path, compression=compression, index=False
    )

    summary = extract_dataset_summary(dataset_id, dataset_root)
    (out_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2) + "\n")

    ev = dataset_root / "results" / "evaluation"
    source_files = {
        p.name: int(p.stat().st_mtime_ns)
        for p in sorted(
            list(ev.glob("*_metrics.csv")) + list(ev.glob("*_scib_metrics.csv"))
        )
        if p.is_file()
    }
    manifest = {
        "dataset_id": dataset_id,
        "source_dir": str(dataset_root.resolve()),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "models": sorted(metrics_df["model"].astype(str).unique()),
        "n_rows": int(len(metrics_df)),
        "metrics_file": METRICS_FILENAME,
        "compression": compression,
        "source_files_mtime_ns": source_files,
    }
    (out_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")

    return out_dir
