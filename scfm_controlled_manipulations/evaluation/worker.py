"""Process-pool workers for intervention-level evaluation (spawn)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from scfm_controlled_manipulations.compute_env import apply_thread_limits
from scfm_controlled_manipulations.evaluation.context import (
    DatasetEvaluateContext,
    ModelEvaluateContext,
    load_dataset_context,
    load_model_context,
)
from scfm_controlled_manipulations.evaluation.disk_cache import (
    read_pickle_cache,
    write_pickle_cache,
)
from scfm_controlled_manipulations.evaluation.intervention_job import evaluate_intervention
from scfm_controlled_manipulations.evaluation.metrics_clustering import run_leiden_labels
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
    distance_metrics: list[str]
    leiden_resolutions: list[float]
    cache_path: Path
    stats_shift_pairwise_max_pairs: int | None
    distance_correlation_subsample_n: int


@dataclass(frozen=True)
class SharedEvalPayload:
    """Pickle-friendly paths for spawn-based workers."""

    results_dir: str
    manipulations_dir: str
    embeddings_root: str
    model: str
    ref_id: str
    dataset_id: str
    seed: int
    k_values: list[int]
    distance_metrics: list[str]
    leiden_resolutions: list[float]
    cache_path: str
    stats_shift_pairwise_cell_subsample_n: int
    stats_shift_pairwise_max_pairs: int | None
    distance_correlation_subsample_n: int
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
    distance_metrics: list[str],
    leiden_resolutions: list[float],
    distance_correlation_subsample_n: int,
) -> str:
    payload = {
        "dataset_id": dataset_id,
        "model": model,
        "seed": seed,
        "k_values": k_values,
        "distance_metrics": distance_metrics,
        "leiden_resolutions": leiden_resolutions,
        "distance_correlation_subsample_n": distance_correlation_subsample_n,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def write_worker_bootstrap(path: Path, shared: SharedEvalContext) -> None:
    write_pickle_cache(path, shared)
    logger.info("Wrote worker bootstrap snapshot to %s", path)


def _rebind_leiden_cache_after_unpickle(model_ctx: ModelEvaluateContext) -> None:
    """Remap Leiden label cache keys to the unpickled embedding matrix object id."""
    old_labels = dict(model_ctx.leiden_cache._labels)
    if not old_labels:
        return
    model_ctx.leiden_cache._labels.clear()
    new_mat_id = id(model_ctx.emb_ref)
    for (_old_id, metric, k, resolution, seed), labels in old_labels.items():
        model_ctx.leiden_cache._labels[(new_mat_id, metric, k, resolution, seed)] = labels


def load_worker_bootstrap(path: Path) -> SharedEvalContext:
    shared = read_pickle_cache(path)
    if not isinstance(shared, SharedEvalContext):
        raise TypeError(f"Expected SharedEvalContext in {path}, got {type(shared)!r}")
    _rebind_leiden_cache_after_unpickle(shared.model_ctx)
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
    leiden_resolutions: list[float],
    cache_path: Path,
    stats_shift_pairwise_cell_subsample_n: int,
    stats_shift_pairwise_max_pairs: int | None,
    distance_correlation_subsample_n: int,
) -> SharedEvalContext:
    model_ctx = load_model_context(
        embeddings_root, model, ref_id, target_obs=dataset_ctx.obs.index
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
        leiden_resolutions=leiden_resolutions,
        cache_path=cache_path,
        stats_shift_pairwise_max_pairs=stats_shift_pairwise_max_pairs,
        distance_correlation_subsample_n=distance_correlation_subsample_n,
    )


def worker_initializer_spawn(payload: SharedEvalPayload) -> None:
    """Spawn worker: load pre-built shared context from bootstrap (fast) or rebuild (fallback)."""
    apply_thread_limits(threads_per_process=max(1, int(payload.worker_threads)))
    bootstrap = payload.bootstrap_path
    if bootstrap:
        path = Path(bootstrap)
        if path.is_file():
            logger.debug("Worker loading bootstrap snapshot from %s", path)
            install_shared_context(load_worker_bootstrap(path))
            return
        logger.warning(
            "Worker bootstrap missing at %s; falling back to full context rebuild",
            path,
        )

    dataset_ctx = load_dataset_context(Path(payload.results_dir), Path(payload.manipulations_dir))
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
        leiden_resolutions=payload.leiden_resolutions,
        cache_path=Path(payload.cache_path),
        stats_shift_pairwise_cell_subsample_n=payload.stats_shift_pairwise_cell_subsample_n,
        stats_shift_pairwise_max_pairs=payload.stats_shift_pairwise_max_pairs,
        distance_correlation_subsample_n=payload.distance_correlation_subsample_n,
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
        leiden_resolutions=ctx.leiden_resolutions,
        cache_path=ctx.cache_path,
        stats_shift_pairwise_max_pairs=ctx.stats_shift_pairwise_max_pairs,
        distance_correlation_subsample_n=ctx.distance_correlation_subsample_n,
    )
