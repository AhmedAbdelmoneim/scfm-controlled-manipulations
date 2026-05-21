"""Cell-type and batch integration metrics via scIB (ASW, iLISI)."""

from __future__ import annotations

import logging
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scib

from scfm_controlled_manipulations.evaluation.metrics_common import (
    scalar_summary,
    summary_to_row_fields,
)

logger = logging.getLogger(__name__)

EMBED_KEY = "X_eval"
METRIC_CATEGORY = "cell_type_and_batch_metrics"


def _obs_col_present(obs_df: pd.DataFrame, col: str | None) -> bool:
    return col is not None and col in obs_df.columns


def log_cell_batch_obs_columns(
    obs_df: pd.DataFrame,
    *,
    cell_type_col: str | None,
    batch_col: str | None,
) -> None:
    """Log whether configured cell-type / batch columns exist in reference ``obs`` (once per dataset)."""
    if cell_type_col is None and batch_col is None:
        logger.info("cell_type_and_batch_metrics: disabled in config (no column names)")
        return
    if cell_type_col is not None:
        if _obs_col_present(obs_df, cell_type_col):
            logger.info(
                "cell_type_and_batch_metrics: cell_type column %r found in reference obs",
                cell_type_col,
            )
        else:
            logger.info(
                "cell_type_and_batch_metrics: cell_type column %r not in reference obs "
                "(cell_type_asw skipped)",
                cell_type_col,
            )
    if batch_col is not None:
        if _obs_col_present(obs_df, batch_col):
            logger.info(
                "cell_type_and_batch_metrics: batch column %r found in reference obs",
                batch_col,
            )
        else:
            logger.info(
                "cell_type_and_batch_metrics: batch column %r not in reference obs "
                "(batch_ilisi skipped)",
                batch_col,
            )


def _as_dense_embedding(mat: Any) -> np.ndarray:
    if hasattr(mat, "todense"):
        return np.asarray(mat.todense(), dtype=np.float32)
    return np.asarray(mat, dtype=np.float32)


def _matrix_to_adata(mat: Any, obs_df: pd.DataFrame) -> ad.AnnData:
    embed = _as_dense_embedding(mat)
    n_cells = embed.shape[0]
    adata = ad.AnnData(X=np.zeros((n_cells, 1), dtype=np.float32))
    adata.obsm[EMBED_KEY] = embed
    adata.obs = obs_df.copy()
    return adata


def _base_row(
    *,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    space_label: str,
    seed: int,
    n_cells: int,
    distance_metric: str,
    k: int,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "model": model,
        "intervention_id": intervention_id,
        "intervention_name": intervention_name,
        "metric_category": METRIC_CATEGORY,
        "space": space_label,
        "seed": seed,
        "n_cells": n_cells,
        "distance_metric": distance_metric,
        "k": k,
    }


def _append_metric_row(
    rows: list[dict[str, Any]],
    *,
    base: dict[str, Any],
    metric_name: str,
    value: float,
) -> None:
    rows.append(
        {
            **base,
            "metric_name": metric_name,
            **summary_to_row_fields(scalar_summary(value)),
            "null_value": np.nan,
        }
    )


def _safe_scib_metric(fn: Any, metric_name: str, space_label: str) -> float:
    try:
        return float(fn())
    except Exception as exc:
        logger.warning(
            "cell_type_and_batch_metrics: %s failed for space=%s: %s",
            metric_name,
            space_label,
            exc,
        )
        return float("nan")


def _compute_scib_integration_rows(
    *,
    mat: Any,
    obs_df: pd.DataFrame,
    space_label: str,
    cell_type_col: str | None,
    batch_col: str | None,
    k_values: list[int],
    distance_metrics: list[str],
    seed: int,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    n_cells: int,
) -> list[dict[str, Any]]:
    has_cell_type = _obs_col_present(obs_df, cell_type_col)
    has_batch = _obs_col_present(obs_df, batch_col)
    if not has_cell_type and not has_batch:
        return []

    k_meta = max(int(k) for k in k_values) if k_values else 0
    distance_metric = distance_metrics[0] if distance_metrics else "cosine"
    base = _base_row(
        dataset_id=dataset_id,
        model=model,
        intervention_id=intervention_id,
        intervention_name=intervention_name,
        space_label=space_label,
        seed=seed,
        n_cells=n_cells,
        distance_metric=distance_metric,
        k=k_meta,
    )

    adata = _matrix_to_adata(mat, obs_df)
    rows: list[dict[str, Any]] = []

    if has_cell_type and cell_type_col is not None:
        asw = _safe_scib_metric(
            lambda: scib.metrics.silhouette(adata, label_key=cell_type_col, embed=EMBED_KEY),
            "cell_type_asw",
            space_label,
        )
        _append_metric_row(rows, base=base, metric_name="cell_type_asw", value=asw)

    if has_batch and batch_col is not None:
        ilisi = _safe_scib_metric(
            lambda: scib.metrics.ilisi_graph(
                adata,
                batch_key=batch_col,
                type_="embed",
                use_rep=EMBED_KEY,
                n_cores=1,
            ),
            "batch_ilisi",
            space_label,
        )
        _append_metric_row(rows, base=base, metric_name="batch_ilisi", value=ilisi)

    return rows


def compute_cell_batch_reference_rows(
    *,
    mat: Any,
    obs_df: pd.DataFrame,
    space_label: str,
    dataset_id: str,
    model: str,
    seed: int,
    cell_type_col: str | None,
    batch_col: str | None,
    k_values: list[int],
    distance_metrics: list[str],
    n_cells: int,
) -> list[dict[str, Any]]:
    """Metrics for reference embedding (constant across interventions)."""
    logger.info("cell_type_and_batch_metrics: precomputing reference space=%s", space_label)
    return _compute_scib_integration_rows(
        mat=mat,
        obs_df=obs_df,
        space_label=space_label,
        cell_type_col=cell_type_col,
        batch_col=batch_col,
        k_values=k_values,
        distance_metrics=distance_metrics,
        seed=seed,
        dataset_id=dataset_id,
        model=model,
        intervention_id="__static__",
        intervention_name="__static__",
        n_cells=n_cells,
    )


def stamp_cell_batch_rows(
    rows: list[dict[str, Any]],
    *,
    intervention_id: str,
    intervention_name: str,
) -> list[dict[str, Any]]:
    stamped: list[dict[str, Any]] = []
    for row in rows:
        stamped.append(
            {
                **row,
                "intervention_id": intervention_id,
                "intervention_name": intervention_name,
            }
        )
    return stamped


def compute_cell_type_and_batch_metrics(
    *,
    bundle: Any,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    seed: int,
    cell_type_col: str | None,
    batch_col: str | None,
    k_values: list[int],
    distance_metrics: list[str],
    static_row_templates: list[list[dict[str, Any]]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    obs = bundle.obs
    n_cells = bundle.emb_ref.shape[0]

    if static_row_templates:
        for template in static_row_templates:
            rows.extend(
                stamp_cell_batch_rows(
                    template,
                    intervention_id=intervention_id,
                    intervention_name=intervention_name,
                )
            )

    logger.info(
        "cell_type_and_batch_metrics: intervention=%s n_cells=%d (embedding_manipulated)",
        intervention_id,
        n_cells,
    )
    rows.extend(
        _compute_scib_integration_rows(
            mat=bundle.emb_man,
            obs_df=obs,
            space_label="embedding_manipulated",
            cell_type_col=cell_type_col,
            batch_col=batch_col,
            k_values=k_values,
            distance_metrics=distance_metrics,
            seed=seed,
            dataset_id=dataset_id,
            model=model,
            intervention_id=intervention_id,
            intervention_name=intervention_name,
            n_cells=n_cells,
        )
    )

    if not rows:
        logger.info("No cell_type/batch columns matched; skipping cell_type_and_batch_metrics")
    return pd.DataFrame(rows)
