"""Evaluate a single intervention against cached reference context."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any

import pandas as pd

from scfm_controlled_manipulations.evaluation.context import (
    DatasetEvaluateContext,
    ModelEvaluateContext,
    load_intervention_bundle,
)
from scfm_controlled_manipulations.evaluation.knn_cache import KnnIndexCache
from scfm_controlled_manipulations.evaluation.metrics_cell_batch import (
    compute_cell_type_and_batch_metrics,
)
from scfm_controlled_manipulations.evaluation.metrics_clustering import compute_clustering_metrics
from scfm_controlled_manipulations.evaluation.metrics_knn import compute_knn_metrics
from scfm_controlled_manipulations.evaluation.metrics_stats_shift import (
    compute_embedding_shift,
    compute_embedding_stats,
)

logger = logging.getLogger(__name__)


def evaluate_intervention(
    *,
    name: str,
    iid: str,
    int_index: int,
    n_planned: int,
    dataset_ctx: DatasetEvaluateContext,
    model_ctx: ModelEvaluateContext,
    results_dir: Path,
    embeddings_root: Path,
    model: str,
    dataset_id: str,
    seed: int,
    k_values: list[int],
    distance_metrics: list[str],
    diffusion_t_values: list[int],
    leiden_resolutions: list[float],
    cache_path: Path,
    knn_cache: KnnIndexCache,
    cell_type_col: str | None,
    batch_col: str | None,
    static_row_templates: list[list[dict[str, Any]]],
    stats_shift_pairwise_max_pairs: int | None = None,
    knn_alpha: float = 10.0,
    knn_bandwidth_k: int | None = None,
    knn_n_null_permutations: int = 1,
) -> list[pd.DataFrame]:
    job_started = time.perf_counter()
    logger.info(
        "  [%d/%d] %s (%s) — loading manipulation matrices",
        int_index,
        n_planned,
        iid,
        name,
    )

    try:
        bundle = load_intervention_bundle(
            dataset_ctx=dataset_ctx,
            model_ctx=model_ctx,
            results_dir=results_dir,
            embeddings_root=embeddings_root,
            model=model,
            intervention_id=iid,
        )
    except FileNotFoundError as err:
        logger.warning("  [%d/%d] %s skipped: %s", int_index, n_planned, iid, err)
        return []
    except ValueError as err:
        logger.error("Alignment failed for %s: %s", iid, err)
        raise

    n_cells = bundle.emb_ref.shape[0]
    logger.info(
        "  [%d/%d] %s — %d cells; running metric blocks",
        int_index,
        n_planned,
        iid,
        n_cells,
    )

    frames: list[pd.DataFrame] = []

    t0 = time.perf_counter()
    frames.append(
        compute_embedding_stats(
            bundle=bundle,
            dataset_id=dataset_id,
            model=model,
            intervention_id=iid,
            intervention_name=name,
            seed=seed,
            ref_cache=model_ctx.ref_stats_cache,
        )
    )
    logger.info(
        "  [%d/%d] %s — embedding_stats done (%.1fs)",
        int_index,
        n_planned,
        iid,
        time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    frames.append(
        compute_embedding_shift(
            bundle=bundle,
            dataset_id=dataset_id,
            model=model,
            intervention_id=iid,
            intervention_name=name,
            seed=seed,
            ref_cache=model_ctx.ref_stats_cache,
            pairwise_max_pairs=stats_shift_pairwise_max_pairs,
        )
    )
    logger.info(
        "  [%d/%d] %s — embedding_shift done (%.1fs)",
        int_index,
        n_planned,
        iid,
        time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    frames.append(
        compute_knn_metrics(
            bundle=bundle,
            dataset_id=dataset_id,
            model=model,
            intervention_id=iid,
            intervention_name=name,
            seed=seed,
            distance_metrics=distance_metrics,
            k_values=k_values,
            diffusion_t_values=diffusion_t_values,
            cache_dir=cache_path,
            knn_cache=knn_cache,
            alpha=knn_alpha,
            bandwidth_k=knn_bandwidth_k,
            n_null_permutations=knn_n_null_permutations,
        )
    )
    logger.info(
        "  [%d/%d] %s — knn_metrics done (%.1fs)",
        int_index,
        n_planned,
        iid,
        time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    frames.append(
        compute_clustering_metrics(
            bundle=bundle,
            dataset_id=dataset_id,
            model=model,
            intervention_id=iid,
            intervention_name=name,
            seed=seed,
            distance_metrics=distance_metrics,
            k_values=k_values,
            leiden_resolutions=leiden_resolutions,
            cache_dir=cache_path,
            leiden_cache=model_ctx.leiden_cache,
        )
    )
    logger.info(
        "  [%d/%d] %s — clustering_metrics done (%.1fs)",
        int_index,
        n_planned,
        iid,
        time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    frames.append(
        compute_cell_type_and_batch_metrics(
            bundle=bundle,
            dataset_id=dataset_id,
            model=model,
            intervention_id=iid,
            intervention_name=name,
            seed=seed,
            cell_type_col=cell_type_col,
            batch_col=batch_col,
            k_values=k_values,
            distance_metrics=distance_metrics,
            static_row_templates=static_row_templates or None,
        )
    )
    logger.info(
        "  [%d/%d] %s — cell_type_and_batch_metrics done (%.1fs); intervention total %.1fs",
        int_index,
        n_planned,
        iid,
        time.perf_counter() - t0,
        time.perf_counter() - job_started,
    )

    return frames
