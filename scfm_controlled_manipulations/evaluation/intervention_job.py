"""Evaluate a single intervention against cached reference context."""

from __future__ import annotations

import logging
from pathlib import Path
import time

import pandas as pd

from scfm_controlled_manipulations.evaluation.context import (
    DatasetEvaluateContext,
    ModelEvaluateContext,
    load_intervention_bundle,
)
from scfm_controlled_manipulations.evaluation.metrics_clustering import compute_clustering_metrics
from scfm_controlled_manipulations.evaluation.metrics_stats_shift import (
    compute_embedding_shift,
    compute_embedding_stats,
)
from scfm_controlled_manipulations.evaluation.metrics_structure import compute_structure_metrics

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
    leiden_resolutions: list[float],
    cache_path: Path,
    stats_shift_pairwise_max_pairs: int | None = None,
    distance_correlation_subsample_n: int | None = None,
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
    ref_stats_cache = model_ctx.ref_stats_cache if bundle.uses_full_reference else None

    t0 = time.perf_counter()
    frames.append(
        compute_embedding_stats(
            bundle=bundle,
            dataset_id=dataset_id,
            model=model,
            intervention_id=iid,
            intervention_name=name,
            seed=seed,
            ref_cache=ref_stats_cache,
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
            ref_cache=ref_stats_cache,
            pairwise_max_pairs=stats_shift_pairwise_max_pairs,
            distance_correlation_subsample_n=distance_correlation_subsample_n,
            distance_metrics=distance_metrics,
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
        compute_structure_metrics(
            bundle=bundle,
            dataset_id=dataset_id,
            model=model,
            intervention_id=iid,
            intervention_name=name,
            seed=seed,
        )
    )
    logger.info(
        "  [%d/%d] %s — structure_metrics done (%.1fs)",
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
        "  [%d/%d] %s — clustering_metrics done (%.1fs); intervention total %.1fs",
        int_index,
        n_planned,
        iid,
        time.perf_counter() - t0,
        time.perf_counter() - job_started,
    )

    return frames
