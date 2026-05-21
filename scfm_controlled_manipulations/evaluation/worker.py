"""Process-pool workers for intervention-level evaluation (fork-shared or spawn-reload)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from scfm_controlled_manipulations.compute_env import apply_thread_limits
from scfm_controlled_manipulations.evaluation.context import (
    DatasetEvaluateContext,
    ModelEvaluateContext,
    load_dataset_context,
    load_model_context,
)
from scfm_controlled_manipulations.evaluation.intervention_job import evaluate_intervention
from scfm_controlled_manipulations.evaluation.leiden_cache import init_leiden_isolate_pool
from scfm_controlled_manipulations.evaluation.metrics_cell_batch import (
    ClassifierCacheKey,
    ClassifierCacheValue,
    compute_cell_batch_static_rows,
)
from scfm_controlled_manipulations.evaluation.reference_prep import precompute_reference_leiden
from scfm_controlled_manipulations.evaluation.reference_stats_shift import (
    precompute_reference_stats_shift,
)

_SHARED: SharedEvalContext | None = None


@dataclass
class InterventionTask:
    int_index: int
    name: str
    intervention_id: str
    n_planned: int


@dataclass
class SharedEvalContext:
    dataset_ctx: DatasetEvaluateContext
    model_ctx: ModelEvaluateContext
    results_dir: Path
    embeddings_root: Path
    model: str
    ref_id: str
    dataset_id: str
    seed: int
    k_values: list[int]
    distance_metrics: list[str]
    diffusion_t_values: list[int]
    leiden_resolutions: list[float]
    leiden_resolution_cell_batch: float
    cache_path: Path
    cell_type_col: str | None
    batch_col: str | None
    stats_shift_pairwise_cell_subsample_n: int
    stats_shift_pairwise_max_pairs: int | None
    knn_alpha: float
    knn_bandwidth_k: int | None
    knn_n_null_permutations: int
    reference_cache: dict[ClassifierCacheKey, ClassifierCacheValue]
    static_row_templates: list[list[dict[str, Any]]]


@dataclass(frozen=True)
class SharedEvalPayload:
    """Pickle-friendly paths for spawn-based workers (macOS / Windows)."""

    results_dir: str
    embeddings_root: str
    model: str
    ref_id: str
    dataset_id: str
    seed: int
    k_values: list[int]
    distance_metrics: list[str]
    diffusion_t_values: list[int]
    leiden_resolutions: list[float]
    leiden_resolution_cell_batch: float
    cache_path: str
    cell_type_col: str | None
    batch_col: str | None
    stats_shift_pairwise_cell_subsample_n: int
    stats_shift_pairwise_max_pairs: int | None
    knn_alpha: float
    knn_bandwidth_k: int | None
    knn_n_null_permutations: int


def install_shared_context(ctx: SharedEvalContext | None) -> None:
    global _SHARED
    _SHARED = ctx


def build_shared_context(
    *,
    dataset_ctx: DatasetEvaluateContext,
    results_dir: Path,
    embeddings_root: Path,
    model: str,
    ref_id: str,
    dataset_id: str,
    seed: int,
    k_values: list[int],
    distance_metrics: list[str],
    diffusion_t_values: list[int],
    leiden_resolutions: list[float],
    leiden_resolution_cell_batch: float,
    cache_path: Path,
    cell_type_col: str | None,
    batch_col: str | None,
    stats_shift_pairwise_cell_subsample_n: int,
    stats_shift_pairwise_max_pairs: int | None,
    knn_alpha: float,
    knn_bandwidth_k: int | None,
    knn_n_null_permutations: int,
) -> SharedEvalContext:
    model_ctx = load_model_context(
        embeddings_root, model, ref_id, target_obs=dataset_ctx.obs.index
    )
    if k_values:
        k_max = max(int(k) for k in k_values)
        for metric in distance_metrics:
            dataset_ctx.knn_cache.neighbors(dataset_ctx.raw_ref, k_max, metric)
            dataset_ctx.knn_cache.neighbors(model_ctx.emb_ref, k_max, metric)
    precompute_reference_leiden(
        model_ctx,
        k_values=k_values,
        distance_metrics=distance_metrics,
        leiden_resolutions=leiden_resolutions,
        leiden_resolution_cell_batch=leiden_resolution_cell_batch,
        seed=seed,
    )
    model_ctx.ref_stats_cache = precompute_reference_stats_shift(
        model_ctx,
        dataset_ctx,
        seed=seed,
        pairwise_cell_subsample_n=stats_shift_pairwise_cell_subsample_n,
        pairwise_max_pairs=stats_shift_pairwise_max_pairs,
    )

    reference_cache: dict[ClassifierCacheKey, ClassifierCacheValue] = {}
    static_row_templates: list[list[dict[str, Any]]] = []
    if cell_type_col or batch_col:
        static_row_templates.append(
            compute_cell_batch_static_rows(
                mat=dataset_ctx.raw_ref,
                obs_df=dataset_ctx.obs,
                space_label="raw_reference",
                dataset_id=dataset_id,
                model=model,
                seed=seed,
                cell_type_col=cell_type_col,
                batch_col=batch_col,
                k_values=k_values,
                distance_metrics=distance_metrics,
                leiden_resolution=leiden_resolution_cell_batch,
                reference_cache=reference_cache,
                knn_cache=dataset_ctx.knn_cache,
                leiden_cache=model_ctx.leiden_cache,
                n_cells=dataset_ctx.n_cells,
            )
        )
        static_row_templates.append(
            compute_cell_batch_static_rows(
                mat=model_ctx.emb_ref,
                obs_df=dataset_ctx.obs,
                space_label="embedding_reference",
                dataset_id=dataset_id,
                model=model,
                seed=seed,
                cell_type_col=cell_type_col,
                batch_col=batch_col,
                k_values=k_values,
                distance_metrics=distance_metrics,
                leiden_resolution=leiden_resolution_cell_batch,
                reference_cache=reference_cache,
                knn_cache=dataset_ctx.knn_cache,
                leiden_cache=model_ctx.leiden_cache,
                n_cells=dataset_ctx.n_cells,
            )
        )

    return SharedEvalContext(
        dataset_ctx=dataset_ctx,
        model_ctx=model_ctx,
        results_dir=results_dir,
        embeddings_root=embeddings_root,
        model=model,
        ref_id=ref_id,
        dataset_id=dataset_id,
        seed=seed,
        k_values=k_values,
        distance_metrics=distance_metrics,
        diffusion_t_values=diffusion_t_values,
        leiden_resolutions=leiden_resolutions,
        leiden_resolution_cell_batch=leiden_resolution_cell_batch,
        cache_path=cache_path,
        cell_type_col=cell_type_col,
        batch_col=batch_col,
        stats_shift_pairwise_cell_subsample_n=stats_shift_pairwise_cell_subsample_n,
        stats_shift_pairwise_max_pairs=stats_shift_pairwise_max_pairs,
        knn_alpha=knn_alpha,
        knn_bandwidth_k=knn_bandwidth_k,
        knn_n_null_permutations=knn_n_null_permutations,
        reference_cache=reference_cache,
        static_row_templates=static_row_templates,
    )


def worker_initializer() -> None:
    """Fork worker: inherit parent ``_SHARED``; pin threads; spawn pool for Leiden."""
    apply_thread_limits(threads_per_process=1)
    init_leiden_isolate_pool()


def worker_initializer_spawn(payload: SharedEvalPayload) -> None:
    """Spawn worker: load reference data and rebuild per-model caches."""
    apply_thread_limits(threads_per_process=1)
    dataset_ctx = load_dataset_context(Path(payload.results_dir))
    shared = build_shared_context(
        dataset_ctx=dataset_ctx,
        results_dir=Path(payload.results_dir),
        embeddings_root=Path(payload.embeddings_root),
        model=payload.model,
        ref_id=payload.ref_id,
        dataset_id=payload.dataset_id,
        seed=payload.seed,
        k_values=payload.k_values,
        distance_metrics=payload.distance_metrics,
        diffusion_t_values=payload.diffusion_t_values,
        leiden_resolutions=payload.leiden_resolutions,
        leiden_resolution_cell_batch=payload.leiden_resolution_cell_batch,
        cache_path=Path(payload.cache_path),
        cell_type_col=payload.cell_type_col,
        batch_col=payload.batch_col,
        stats_shift_pairwise_cell_subsample_n=payload.stats_shift_pairwise_cell_subsample_n,
        stats_shift_pairwise_max_pairs=payload.stats_shift_pairwise_max_pairs,
        knn_alpha=payload.knn_alpha,
        knn_bandwidth_k=payload.knn_bandwidth_k,
        knn_n_null_permutations=payload.knn_n_null_permutations,
    )
    install_shared_context(shared)


def run_intervention_task(task: InterventionTask) -> list[pd.DataFrame]:
    if _SHARED is None:
        raise RuntimeError("Evaluation worker context was not initialized")
    ctx = _SHARED
    return evaluate_intervention(
        name=task.name,
        iid=task.intervention_id,
        int_index=task.int_index,
        n_planned=task.n_planned,
        dataset_ctx=ctx.dataset_ctx,
        model_ctx=ctx.model_ctx,
        results_dir=ctx.results_dir,
        embeddings_root=ctx.embeddings_root,
        model=ctx.model,
        dataset_id=ctx.dataset_id,
        seed=ctx.seed,
        k_values=ctx.k_values,
        distance_metrics=ctx.distance_metrics,
        diffusion_t_values=ctx.diffusion_t_values,
        leiden_resolutions=ctx.leiden_resolutions,
        cache_path=ctx.cache_path,
        knn_cache=ctx.dataset_ctx.knn_cache,
        reference_cache=ctx.reference_cache,
        cell_type_col=ctx.cell_type_col,
        batch_col=ctx.batch_col,
        leiden_resolution_cell_batch=ctx.leiden_resolution_cell_batch,
        static_row_templates=ctx.static_row_templates,
        stats_shift_pairwise_max_pairs=ctx.stats_shift_pairwise_max_pairs,
        knn_alpha=ctx.knn_alpha,
        knn_bandwidth_k=ctx.knn_bandwidth_k,
        knn_n_null_permutations=ctx.knn_n_null_permutations,
    )
