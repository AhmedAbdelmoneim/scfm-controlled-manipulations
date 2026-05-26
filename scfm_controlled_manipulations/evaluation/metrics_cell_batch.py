"""Cell-type and batch integration metrics via scib-metrics (ASW, iLISI)."""

from __future__ import annotations

import logging
from typing import Any, Literal
import warnings

import numpy as np
import pandas as pd
import scib_metrics
from scib_metrics.nearest_neighbors import NeighborsResults, pynndescent

from scfm_controlled_manipulations.evaluation.data import _as_dense_embedding
from scfm_controlled_manipulations.evaluation.metrics_common import (
    make_metric_row,
    scalar_summary,
)

# scib-metrics 0.5.x: deprecated pandas.value_counts in graph_connectivity (upstream).
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*value_counts is deprecated.*",
    module=r"scib_metrics\..*",
)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

METRIC_CATEGORY = "cell_type_and_batch_metrics"
SilhouetteMetric = Literal["euclidean", "cosine"]


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
                "(cell_type_asw and graph_connectivity skipped)",
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


def _embedding_matrix(mat: Any) -> np.ndarray:
    return _as_dense_embedding(mat).astype(np.float64, copy=False)


def _silhouette_metric(distance_metric: str) -> SilhouetteMetric:
    if distance_metric == "cosine":
        return "cosine"
    if distance_metric != "euclidean":
        logger.warning(
            "cell_type_and_batch_metrics: unknown distance_metric %r; using euclidean for ASW",
            distance_metric,
        )
    return "euclidean"


def _neighbors_for_knn_metrics(
    embed: np.ndarray,
    *,
    n_neighbors: int,
    distance_metric: str,
    seed: int,
) -> NeighborsResults:
    """Build a kNN graph for iLISI / graph connectivity (pynndescent; cosine via L2 norm)."""
    x = embed
    if distance_metric == "cosine":
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        x = x / norms
    elif distance_metric != "euclidean":
        logger.warning(
            "cell_type_and_batch_metrics: unknown distance_metric %r; using euclidean for iLISI",
            distance_metric,
        )
    return pynndescent(x, n_neighbors=n_neighbors, random_state=seed, n_jobs=1)


def _append_metric_row(
    rows: list[dict[str, Any]],
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
    metric_name: str,
    value: float,
) -> None:
    rows.append(
        make_metric_row(
            dataset_id=dataset_id,
            model=model,
            intervention_id=intervention_id,
            intervention_name=intervention_name,
            metric_category=METRIC_CATEGORY,
            metric_name=metric_name,
            space=space_label,
            summary=scalar_summary(value),
            n_cells=n_cells,
            seed=seed,
            extra={"distance_metric": distance_metric, "k": k},
        )
    )


def _safe_metric(fn: Any, metric_name: str, space_label: str) -> float:
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


def _compute_integration_rows(
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

    k_meta = max(int(k) for k in k_values) if k_values else 15
    distance_metric = distance_metrics[0] if distance_metrics else "euclidean"
    embed = _embedding_matrix(mat)
    rows: list[dict[str, Any]] = []
    neighbors: NeighborsResults | None = None

    def _neighbors() -> NeighborsResults:
        nonlocal neighbors
        if neighbors is None:
            neighbors = _neighbors_for_knn_metrics(
                embed,
                n_neighbors=k_meta,
                distance_metric=distance_metric,
                seed=seed,
            )
        return neighbors

    if has_cell_type and cell_type_col is not None:
        labels = obs_df[cell_type_col].to_numpy()
        sil_metric = _silhouette_metric(distance_metric)
        asw = _safe_metric(
            lambda: scib_metrics.silhouette_label(
                embed,
                labels,
                rescale=True,
                metric=sil_metric,
            ),
            "cell_type_asw",
            space_label,
        )
        _append_metric_row(
            rows,
            dataset_id=dataset_id,
            model=model,
            intervention_id=intervention_id,
            intervention_name=intervention_name,
            space_label=space_label,
            seed=seed,
            n_cells=n_cells,
            distance_metric=distance_metric,
            k=k_meta,
            metric_name="cell_type_asw",
            value=asw,
        )
        nbrs = _neighbors()
        connectivity = _safe_metric(
            lambda: scib_metrics.graph_connectivity(nbrs, labels),
            "graph_connectivity",
            space_label,
        )
        _append_metric_row(
            rows,
            dataset_id=dataset_id,
            model=model,
            intervention_id=intervention_id,
            intervention_name=intervention_name,
            space_label=space_label,
            seed=seed,
            n_cells=n_cells,
            distance_metric=distance_metric,
            k=k_meta,
            metric_name="graph_connectivity",
            value=connectivity,
        )

    if has_batch and batch_col is not None:
        batches = obs_df[batch_col].to_numpy()
        ilisi = _safe_metric(
            lambda: scib_metrics.ilisi_knn(_neighbors(), batches),
            "batch_ilisi",
            space_label,
        )
        _append_metric_row(
            rows,
            dataset_id=dataset_id,
            model=model,
            intervention_id=intervention_id,
            intervention_name=intervention_name,
            space_label=space_label,
            seed=seed,
            n_cells=n_cells,
            distance_metric=distance_metric,
            k=k_meta,
            metric_name="batch_ilisi",
            value=ilisi,
        )

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
    return _compute_integration_rows(
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
        _compute_integration_rows(
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
