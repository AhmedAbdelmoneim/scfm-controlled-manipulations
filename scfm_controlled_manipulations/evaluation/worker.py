"""Process-pool workers for intervention-level evaluation (spawn)."""

from __future__ import annotations

import hashlib
import json
import logging
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
from scfm_controlled_manipulations.evaluation.disk_cache import read_pickle_cache, write_pickle_cache
from scfm_controlled_manipulations.evaluation.intervention_job import evaluate_intervention
from scfm_controlled_manipulations.evaluation.metrics_cell_batch import (
    compute_cell_batch_reference_rows,
)
from scfm_controlled_manipulations.evaluation.metrics_clustering import run_leiden_labels
from scfm_controlled_manipulations.evaluation.metrics_knn import prewarm_reference_evaluation_disk_cache
from scfm_controlled_manipulations.evaluation.reference_stats_shift import (
    precompute_reference_stats_shift,
)

logger = logging.getLogger(__name__)

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
    trustworthiness_k_values: list[int]
    distance_metrics: list[str]
    diffusion_t_values: list[int]
    leiden_resolutions: list[float]
    cache_path: Path
    cell_type_col: str | None
    batch_col: str | None
    stats_shift_pairwise_max_pairs: int | None
    knn_alpha: float
    knn_bandwidth_k: int | None
    knn_n_null_permutations: int
    static_row_templates: list[list[dict[str, Any]]]


@dataclass(frozen=True)
class SharedEvalPayload:
    """Pickle-friendly paths for spawn-based workers."""

    results_dir: str
    embeddings_root: str
    model: str
    ref_id: str
    dataset_id: str
    seed: int
    k_values: list[int]
    trustworthiness_k_values: list[int]
    distance_metrics: list[str]
    diffusion_t_values: list[int]
    leiden_resolutions: list[float]
    cache_path: str
    cell_type_col: str | None
    batch_col: str | None
    stats_shift_pairwise_cell_subsample_n: int
    stats_shift_pairwise_max_pairs: int | None
    knn_alpha: float
    knn_bandwidth_k: int | None
    knn_n_null_permutations: int
    bootstrap_path: str | None = None
    worker_threads: int = 1


def install_shared_context(ctx: SharedEvalContext | None) -> None:
    global _SHARED
    _SHARED = ctx


def worker_bootstrap_path(cache_path: Path, *, model: str, fingerprint: str) -> Path:
    return cache_path / f"worker_bootstrap_{model}_{fingerprint}.pkl"


def worker_bootstrap_fingerprint(
    *,
    dataset_id: str,
    model: str,
    seed: int,
    k_values: list[int],
    trustworthiness_k_values: list[int],
    distance_metrics: list[str],
    diffusion_t_values: list[int],
    leiden_resolutions: list[float],
    knn_alpha: float,
    knn_bandwidth_k: int | None,
    knn_n_null_permutations: int,
    cell_type_col: str | None,
    batch_col: str | None,
) -> str:
    payload = {
        "dataset_id": dataset_id,
        "model": model,
        "seed": seed,
        "k_values": k_values,
        "trustworthiness_k_values": trustworthiness_k_values,
        "distance_metrics": distance_metrics,
        "diffusion_t_values": diffusion_t_values,
        "leiden_resolutions": leiden_resolutions,
        "knn_alpha": knn_alpha,
        "knn_bandwidth_k": knn_bandwidth_k,
        "knn_n_null_permutations": knn_n_null_permutations,
        "cell_type_col": cell_type_col,
        "batch_col": batch_col,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def write_worker_bootstrap(path: Path, shared: SharedEvalContext) -> None:
    write_pickle_cache(path, shared)
    logger.info("Wrote worker bootstrap snapshot to %s", path)


def _rebind_knn_cache_after_unpickle(
    *,
    dataset_ctx: DatasetEvaluateContext,
    model_ctx: ModelEvaluateContext,
    distance_metrics: list[str],
    k_values: list[int],
    cache_path: Path,
    dataset_id: str,
    model: str,
    knn_n_jobs: int,
) -> None:
    """Re-seed reference kNN entries with live matrix object ids (spawn unpickle invalidates ``id()`` keys)."""
    if not k_values:
        return
    k_max = max(int(k) for k in k_values)
    dataset_ctx.knn_cache._store.clear()
    for space, mat in (("raw", dataset_ctx.raw_ref), ("embedding", model_ctx.emb_ref)):
        for metric in distance_metrics:
            dataset_ctx.knn_cache.warm_reference_from_disk(
                mat,
                space=space,
                k_max=k_max,
                metric=metric,
                cache_dir=cache_path,
                dataset_id=dataset_id,
                model=model,
                n_cells=dataset_ctx.n_cells,
                knn_n_jobs=knn_n_jobs,
            )


def _rebind_leiden_cache_after_unpickle(model_ctx: ModelEvaluateContext) -> None:
    """Remap Leiden label cache keys to the unpickled embedding matrix object id."""
    old_labels = dict(model_ctx.leiden_cache._labels)
    if not old_labels:
        return
    model_ctx.leiden_cache._labels.clear()
    new_mat_id = id(model_ctx.emb_ref)
    for (_old_id, metric, k, resolution, seed), labels in old_labels.items():
        model_ctx.leiden_cache._labels[(new_mat_id, metric, k, resolution, seed)] = labels


def load_worker_bootstrap(path: Path, *, knn_n_jobs: int = 1) -> SharedEvalContext:
    shared = read_pickle_cache(path)
    if not isinstance(shared, SharedEvalContext):
        raise TypeError(f"Expected SharedEvalContext in {path}, got {type(shared)!r}")
    _rebind_leiden_cache_after_unpickle(shared.model_ctx)
    _rebind_knn_cache_after_unpickle(
        dataset_ctx=shared.dataset_ctx,
        model_ctx=shared.model_ctx,
        distance_metrics=shared.distance_metrics,
        k_values=shared.k_values,
        cache_path=shared.cache_path,
        dataset_id=shared.dataset_id,
        model=shared.model,
        knn_n_jobs=knn_n_jobs,
    )
    return shared


def _warm_reference_leiden_cache(
    *,
    model_ctx: ModelEvaluateContext,
    k_values: list[int],
    distance_metrics: list[str],
    leiden_resolutions: list[float],
    seed: int,
) -> None:
    resolutions = sorted(set(float(r) for r in leiden_resolutions))
    for metric in distance_metrics:
        for k in k_values:
            for resolution in resolutions:
                run_leiden_labels(
                    model_ctx.emb_ref,
                    k=int(k),
                    metric=metric,
                    resolution=resolution,
                    seed=seed,
                    leiden_cache=model_ctx.leiden_cache,
                )


def _warm_reference_knn_cache(
    *,
    dataset_ctx: DatasetEvaluateContext,
    model_ctx: ModelEvaluateContext,
    k_values: list[int],
    distance_metrics: list[str],
    cache_path: Path,
    dataset_id: str,
    model: str,
    diffusion_t_values: list[int],
    knn_alpha: float,
    knn_bandwidth_k: int | None,
    knn_n_jobs: int,
) -> None:
    if not k_values:
        return
    k_max = max(int(k) for k in k_values)
    prewarm_reference_evaluation_disk_cache(
        cache_dir=cache_path,
        dataset_id=dataset_id,
        model=model,
        raw_ref=dataset_ctx.raw_ref,
        emb_ref=model_ctx.emb_ref,
        n_cells=dataset_ctx.n_cells,
        k_values=k_values,
        distance_metrics=distance_metrics,
        diffusion_t_values=diffusion_t_values,
        alpha=knn_alpha,
        bandwidth_k=knn_bandwidth_k,
        knn_n_jobs=knn_n_jobs,
    )
    for space, mat in (("raw", dataset_ctx.raw_ref), ("embedding", model_ctx.emb_ref)):
        for metric in distance_metrics:
            dataset_ctx.knn_cache.warm_reference_from_disk(
                mat,
                space=space,
                k_max=k_max,
                metric=metric,
                cache_dir=cache_path,
                dataset_id=dataset_id,
                model=model,
                n_cells=dataset_ctx.n_cells,
                knn_n_jobs=knn_n_jobs,
            )


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
    trustworthiness_k_values: list[int],
    distance_metrics: list[str],
    diffusion_t_values: list[int],
    leiden_resolutions: list[float],
    cache_path: Path,
    cell_type_col: str | None,
    batch_col: str | None,
    stats_shift_pairwise_cell_subsample_n: int,
    stats_shift_pairwise_max_pairs: int | None,
    knn_alpha: float,
    knn_bandwidth_k: int | None,
    knn_n_null_permutations: int,
    knn_build_threads: int = 1,
) -> SharedEvalContext:
    model_ctx = load_model_context(
        embeddings_root, model, ref_id, target_obs=dataset_ctx.obs.index
    )
    _warm_reference_knn_cache(
        dataset_ctx=dataset_ctx,
        model_ctx=model_ctx,
        k_values=k_values,
        distance_metrics=distance_metrics,
        cache_path=cache_path,
        dataset_id=dataset_id,
        model=model,
        diffusion_t_values=diffusion_t_values,
        knn_alpha=knn_alpha,
        knn_bandwidth_k=knn_bandwidth_k,
        knn_n_jobs=max(1, int(knn_build_threads)),
    )
    # Leiden uses scanpy/pynndescent/numba (must stay at 1 thread after process init).
    apply_thread_limits(threads_per_process=1)
    _warm_reference_leiden_cache(
        model_ctx=model_ctx,
        k_values=k_values,
        distance_metrics=distance_metrics,
        leiden_resolutions=leiden_resolutions,
        seed=seed,
    )
    model_ctx.ref_stats_cache = precompute_reference_stats_shift(
        model_ctx,
        dataset_ctx,
        seed=seed,
        pairwise_cell_subsample_n=stats_shift_pairwise_cell_subsample_n,
        pairwise_max_pairs=stats_shift_pairwise_max_pairs,
    )
    static_row_templates: list[list[dict[str, Any]]] = []
    if cell_type_col or batch_col:
        static_row_templates.append(
            compute_cell_batch_reference_rows(
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
        trustworthiness_k_values=trustworthiness_k_values,
        distance_metrics=distance_metrics,
        diffusion_t_values=diffusion_t_values,
        leiden_resolutions=leiden_resolutions,
        cache_path=cache_path,
        cell_type_col=cell_type_col,
        batch_col=batch_col,
        stats_shift_pairwise_max_pairs=stats_shift_pairwise_max_pairs,
        knn_alpha=knn_alpha,
        knn_bandwidth_k=knn_bandwidth_k,
        knn_n_null_permutations=knn_n_null_permutations,
        static_row_templates=static_row_templates,
    )


def worker_initializer_spawn(payload: SharedEvalPayload) -> None:
    """Spawn worker: load pre-built shared context from bootstrap (fast) or rebuild (fallback)."""
    apply_thread_limits(threads_per_process=max(1, int(payload.worker_threads)))
    bootstrap = payload.bootstrap_path
    if bootstrap:
        path = Path(bootstrap)
        if path.is_file():
            logger.debug("Worker loading bootstrap snapshot from %s", path)
            install_shared_context(
                load_worker_bootstrap(path, knn_n_jobs=max(1, int(payload.worker_threads)))
            )
            return
        logger.warning(
            "Worker bootstrap missing at %s; falling back to full context rebuild",
            path,
        )

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
        trustworthiness_k_values=payload.trustworthiness_k_values,
        distance_metrics=payload.distance_metrics,
        diffusion_t_values=payload.diffusion_t_values,
        leiden_resolutions=payload.leiden_resolutions,
        cache_path=Path(payload.cache_path),
        cell_type_col=payload.cell_type_col,
        batch_col=payload.batch_col,
        stats_shift_pairwise_cell_subsample_n=payload.stats_shift_pairwise_cell_subsample_n,
        stats_shift_pairwise_max_pairs=payload.stats_shift_pairwise_max_pairs,
        knn_alpha=payload.knn_alpha,
        knn_bandwidth_k=payload.knn_bandwidth_k,
        knn_n_null_permutations=payload.knn_n_null_permutations,
        knn_build_threads=1,
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
        trustworthiness_k_values=ctx.trustworthiness_k_values,
        distance_metrics=ctx.distance_metrics,
        diffusion_t_values=ctx.diffusion_t_values,
        leiden_resolutions=ctx.leiden_resolutions,
        cache_path=ctx.cache_path,
        knn_cache=ctx.dataset_ctx.knn_cache,
        cell_type_col=ctx.cell_type_col,
        batch_col=ctx.batch_col,
        static_row_templates=ctx.static_row_templates,
        stats_shift_pairwise_max_pairs=ctx.stats_shift_pairwise_max_pairs,
        knn_alpha=ctx.knn_alpha,
        knn_bandwidth_k=ctx.knn_bandwidth_k,
        knn_n_null_permutations=ctx.knn_n_null_permutations,
    )
