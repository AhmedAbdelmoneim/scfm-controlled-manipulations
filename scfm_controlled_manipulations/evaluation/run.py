"""CLI entry for paired reference vs manipulation structure metrics."""

from __future__ import annotations

import multiprocessing as mp

from scfm_controlled_manipulations.compute_env import apply_thread_limits

apply_thread_limits(threads_per_process=1)

from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from scfm_controlled_manipulations.evaluation.context import load_dataset_context
from scfm_controlled_manipulations.evaluation.metrics_cell_batch import log_cell_batch_obs_columns
from scfm_controlled_manipulations.evaluation.metrics_common import VALUE_SUMMARY_COLUMNS
from scfm_controlled_manipulations.evaluation.worker import (
    InterventionTask,
    SharedEvalPayload,
    build_shared_context,
    install_shared_context,
    run_intervention_task,
    worker_initializer_spawn,
)
from scfm_controlled_manipulations.io import (
    embedding_path,
    evaluation_cache_dir,
    evaluation_dir,
    evaluation_metrics_csv_path,
    intervention_id,
    manipulation_path,
)
from scfm_controlled_manipulations.obs_columns import (
    resolve_batch_column,
    resolve_cell_type_column,
)
from scfm_controlled_manipulations.sweep_config import (
    expand_intervention_specs,
    reference_intervention_id,
)

logger = logging.getLogger(__name__)

DEFAULT_EVALUATION: dict[str, Any] = {
    "k_values": [5, 15, 50],
    "distance_metrics": ["euclidean"],
    "diffusion_t_values": [1, 2, 4, 8, 16, 32],
    "leiden_resolutions": [0.25, 0.5, 1.0, 2.0],
    "cell_type_col": "cell_type",
    "batch_col": "batch",
    "dataset_id": None,
    "evaluation_workers": 1,
    "stats_shift_pairwise_cell_subsample_n": 500,
    "stats_shift_pairwise_max_pairs": 10_000,
    "knn_alpha": 10.0,
    "knn_bandwidth_k": None,
    "knn_n_null_permutations": 1,
}


def _require_positive_int_list(values: Any, *, key: str) -> list[int]:
    out = [int(v) for v in values]
    if not out or any(v <= 0 for v in out):
        raise ValueError(f"evaluation.{key} must be a non-empty list of positive integers")
    return out


def _require_positive_float_list(values: Any, *, key: str) -> list[float]:
    out = [float(v) for v in values]
    if not out or any(v <= 0 for v in out):
        raise ValueError(f"evaluation.{key} must be a non-empty list of positive floats")
    return out


def validate_evaluation_config(ev: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize evaluation config values once at run start."""
    validated = dict(ev)
    validated["k_values"] = _require_positive_int_list(validated["k_values"], key="k_values")
    validated["diffusion_t_values"] = _require_positive_int_list(
        validated["diffusion_t_values"], key="diffusion_t_values"
    )
    validated["leiden_resolutions"] = _require_positive_float_list(
        validated["leiden_resolutions"], key="leiden_resolutions"
    )
    distance_metrics = [str(m).strip() for m in validated["distance_metrics"]]
    if not distance_metrics or any(not m for m in distance_metrics):
        raise ValueError("evaluation.distance_metrics must be a non-empty list of strings")
    validated["distance_metrics"] = distance_metrics

    validated["evaluation_workers"] = max(1, int(validated["evaluation_workers"]))

    validated["stats_shift_pairwise_cell_subsample_n"] = int(
        validated["stats_shift_pairwise_cell_subsample_n"]
    )
    if validated["stats_shift_pairwise_cell_subsample_n"] <= 0:
        raise ValueError("evaluation.stats_shift_pairwise_cell_subsample_n must be > 0")
    pairwise_max = validated.get("stats_shift_pairwise_max_pairs")
    if pairwise_max is not None:
        pairwise_max = int(pairwise_max)
        if pairwise_max <= 0:
            raise ValueError("evaluation.stats_shift_pairwise_max_pairs must be > 0 or null")
    validated["stats_shift_pairwise_max_pairs"] = pairwise_max

    validated["knn_alpha"] = float(validated["knn_alpha"])
    bw_raw = validated.get("knn_bandwidth_k")
    validated["knn_bandwidth_k"] = int(bw_raw) if bw_raw is not None else None
    validated["knn_n_null_permutations"] = max(1, int(validated["knn_n_null_permutations"]))
    return validated


def merge_evaluation_config(cfg: dict[str, Any]) -> dict[str, Any]:
    merged = dict(DEFAULT_EVALUATION)
    merged.update(cfg.get("evaluation") or {})
    return validate_evaluation_config(merged)


def _optional_obs_column(value: Any) -> str | None:
    """Return a column name, or ``None`` to skip metadata metrics for that field."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def _dataset_id(cfg: dict[str, Any], ev: dict[str, Any]) -> str:
    if ev.get("dataset_id"):
        return str(ev["dataset_id"])
    return Path(cfg["input_h5ad"]).stem


def _planned_interventions(
    specs: list[dict[str, Any]],
    *,
    ref_id: str,
    results_dir: Path,
    embeddings_root: Path,
    model: str,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Interventions with on-disk manipulated raw data and an embedding for ``model``."""
    planned: list[tuple[str, str, dict[str, Any]]] = []
    for spec in specs:
        name = str(spec["name"])
        kwargs = dict(spec.get("kwargs") or {})
        iid = intervention_id(name, kwargs)
        if iid == ref_id:
            continue
        if not manipulation_path(results_dir, iid).is_file():
            continue
        if embedding_path(embeddings_root, model, iid).is_file():
            planned.append((name, iid, spec))
    return planned


def append_raw_embedding_gain_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Append rows with ``metric_category`` suffix ``_gain`` (embedding minus raw) where both exist."""
    gain_rows: list[dict[str, Any]] = []
    for cat in ("embedding_shift", "knn_metrics"):
        sub = df[df["metric_category"] == cat]
        if sub.empty or "space" not in sub.columns:
            continue
        raw = sub[sub["space"] == "raw"].copy()
        emb = sub[sub["space"] == "embedding"].copy()
        if raw.empty or emb.empty:
            continue
        keys = [
            "dataset_id",
            "model",
            "intervention_id",
            "intervention_name",
            "metric_name",
        ]
        for optional in ("distance_metric", "k", "diffusion_t", "resolution"):
            if optional in raw.columns and optional in emb.columns:
                keys.append(optional)
        keys = [k for k in keys if k in raw.columns and k in emb.columns]
        merged = emb.merge(raw, on=keys, how="inner", suffixes=("_emb", "_raw"))
        if merged.empty:
            continue
        block_data: dict[str, Any] = {
            "dataset_id": merged["dataset_id"],
            "model": merged["model"],
            "intervention_id": merged["intervention_id"],
            "intervention_name": merged["intervention_name"],
            "metric_category": f"{cat}_gain",
            "metric_name": merged["metric_name"],
            "space": "embedding_minus_raw",
            "null_value": np.nan,
            "n_cells": merged["n_cells_emb"].astype(int),
            "seed": merged["seed_emb"].astype(int),
        }
        for col in VALUE_SUMMARY_COLUMNS:
            emb_col = f"{col}_emb"
            raw_col = f"{col}_raw"
            if col == "value_std":
                block_data[col] = np.nan
            elif emb_col in merged.columns and raw_col in merged.columns:
                block_data[col] = merged[emb_col] - merged[raw_col]
            else:
                block_data[col] = np.nan
        block = pd.DataFrame(block_data)
        for optional in ("distance_metric", "k", "diffusion_t", "resolution"):
            col = optional
            if col in merged.columns:
                block[col] = merged[col]
        gain_rows.extend(block.to_dict("records"))
    if not gain_rows:
        return df
    return pd.concat([df, pd.DataFrame(gain_rows)], ignore_index=True)


def _log_job_progress(
    *,
    completed_jobs: int,
    total_jobs: int,
    run_started: float,
    knn_cache_size: int,
    int_index: int | None = None,
    n_planned: int | None = None,
    iid: str | None = None,
) -> None:
    elapsed_run = time.perf_counter() - run_started
    if completed_jobs <= 0 or total_jobs <= 0:
        return
    eta_s = elapsed_run / completed_jobs * (total_jobs - completed_jobs)
    if int_index is not None and iid is not None and n_planned is not None:
        logger.info(
            "  [%d/%d] %s complete; overall %d/%d jobs (%.0f%%), ETA ~%.0f min; kNN cache=%d",
            int_index,
            n_planned,
            iid,
            completed_jobs,
            total_jobs,
            100.0 * completed_jobs / total_jobs,
            eta_s / 60.0,
            knn_cache_size,
        )
    else:
        logger.info(
            "  overall %d/%d jobs (%.0f%%), ETA ~%.0f min; kNN cache=%d",
            completed_jobs,
            total_jobs,
            100.0 * completed_jobs / total_jobs,
            eta_s / 60.0,
            knn_cache_size,
        )


def run_evaluate(cfg: dict[str, Any]) -> None:
    ev = merge_evaluation_config(cfg)
    run_started = time.perf_counter()

    results_dir = Path(cfg["results_dir"])
    embeddings_root = Path(cfg["embeddings_root"])
    ref_id = reference_intervention_id(cfg)
    specs = expand_intervention_specs(cfg["interventions"])
    seed = int(cfg.get("seed", 42))
    dataset_id = _dataset_id(cfg, ev)
    cache_path = evaluation_cache_dir(results_dir)
    evaluation_workers = max(1, int(ev.get("evaluation_workers", 1)))

    evaluation_dir(results_dir).mkdir(parents=True, exist_ok=True)

    models = list(cfg["models"])
    k_values = [int(k) for k in ev["k_values"]]
    distance_metrics = list(ev["distance_metrics"])
    diffusion_t_values = [int(t) for t in ev["diffusion_t_values"]]
    leiden_resolutions = [float(x) for x in ev["leiden_resolutions"]]
    cell_type_col_config = _optional_obs_column(ev.get("cell_type_col"))
    batch_col_config = _optional_obs_column(ev.get("batch_col"))
    stats_shift_pairwise_cell_subsample_n = int(
        ev.get("stats_shift_pairwise_cell_subsample_n", 500)
    )
    pairwise_max_raw = ev.get("stats_shift_pairwise_max_pairs")
    stats_shift_pairwise_max_pairs = (
        int(pairwise_max_raw) if pairwise_max_raw is not None else None
    )
    knn_alpha = float(ev.get("knn_alpha", 10.0))
    bw_raw = ev.get("knn_bandwidth_k")
    knn_bandwidth_k = int(bw_raw) if bw_raw is not None else None
    knn_n_null_permutations = max(1, int(ev.get("knn_n_null_permutations", 1)))

    total_jobs = sum(
        len(
            _planned_interventions(
                specs,
                ref_id=ref_id,
                results_dir=results_dir,
                embeddings_root=embeddings_root,
                model=m,
            )
        )
        for m in models
    )

    logger.info(
        "Evaluate: dataset_id=%s results_dir=%s embeddings_root=%s",
        dataset_id,
        results_dir,
        embeddings_root,
    )
    mp_method = mp.get_context("spawn").get_start_method()
    logger.info(
        "Plan: models=%d interventions_with_embeddings=%d "
        "(k=%s metrics=%s diffusion_t=%s leiden_res=%s cache=%s workers=%d mp=%s)",
        len(models),
        total_jobs,
        k_values,
        distance_metrics,
        diffusion_t_values,
        leiden_resolutions,
        cache_path,
        evaluation_workers,
        mp_method,
    )
    logger.info(
        "Each worker uses 1 BLAS/sklearn thread; set evaluation.evaluation_workers up to your CPU budget. "
        "Process pool start method is spawn and Leiden runs in-process."
    )

    logger.info("Loading shared reference raw matrix and obs (once per dataset)")
    t0 = time.perf_counter()
    dataset_ctx = load_dataset_context(results_dir)
    knn_cache = dataset_ctx.knn_cache
    obs_cols = dataset_ctx.obs.columns
    cell_type_col = resolve_cell_type_column(obs_cols, cell_type_col_config)
    batch_col = resolve_batch_column(obs_cols, batch_col_config)
    if cell_type_col_config and cell_type_col != cell_type_col_config:
        logger.info(
            "Resolved cell_type_col %r -> %r in reference obs",
            cell_type_col_config,
            cell_type_col,
        )
    elif cell_type_col_config and cell_type_col is None:
        logger.warning(
            "cell_type_col %r not found in reference obs (tried aliases); "
            "cell_type_asw and graph_connectivity skipped",
            cell_type_col_config,
        )
    if batch_col_config and batch_col != batch_col_config:
        logger.info(
            "Resolved batch_col %r -> %r in reference obs",
            batch_col_config,
            batch_col,
        )
    logger.info(
        "Reference raw loaded: n_cells=%d (%.1fs)",
        dataset_ctx.n_cells,
        time.perf_counter() - t0,
    )
    log_cell_batch_obs_columns(
        dataset_ctx.obs,
        cell_type_col=cell_type_col,
        batch_col=batch_col,
    )

    completed_jobs = 0

    for model_index, model in enumerate(models, start=1):
        model_started = time.perf_counter()
        planned = _planned_interventions(
            specs,
            ref_id=ref_id,
            results_dir=results_dir,
            embeddings_root=embeddings_root,
            model=model,
        )
        n_planned = len(planned)
        n_non_ref = sum(
            1
            for spec in specs
            if intervention_id(str(spec["name"]), dict(spec.get("kwargs") or {})) != ref_id
        )
        logger.info(
            "Model %d/%d %s: %d interventions to evaluate (%d skipped without embeddings)",
            model_index,
            len(models),
            model,
            n_planned,
            n_non_ref - n_planned,
        )

        tasks = [
            InterventionTask(int_index=idx, name=name, intervention_id=iid, n_planned=n_planned)
            for idx, (name, iid, _spec) in enumerate(planned, start=1)
        ]
        frames: list[pd.DataFrame] = []

        if n_planned == 0:
            logger.warning("No interventions to evaluate for model %s; skipping", model)
            continue

        logger.info("Preparing shared reference context for model %s", model)
        t0 = time.perf_counter()
        shared = build_shared_context(
            dataset_ctx=dataset_ctx,
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
            cache_path=cache_path,
            cell_type_col=cell_type_col,
            batch_col=batch_col,
            stats_shift_pairwise_cell_subsample_n=stats_shift_pairwise_cell_subsample_n,
            stats_shift_pairwise_max_pairs=stats_shift_pairwise_max_pairs,
            knn_alpha=knn_alpha,
            knn_bandwidth_k=knn_bandwidth_k,
            knn_n_null_permutations=knn_n_null_permutations,
        )
        logger.info(
            "Reference context ready (%.1fs; cell/batch templates=%d; Leiden cache=%d; stats cache=%s)",
            time.perf_counter() - t0,
            len(shared.static_row_templates),
            len(shared.model_ctx.leiden_cache),
            shared.model_ctx.ref_stats_cache is not None,
        )

        pool_workers = min(evaluation_workers, n_planned)

        if pool_workers <= 1:
            install_shared_context(shared)
            for task in tasks:
                job_frames = run_intervention_task(task)
                if job_frames:
                    frames.extend(job_frames)
                    completed_jobs += 1
                    _log_job_progress(
                        completed_jobs=completed_jobs,
                        total_jobs=total_jobs,
                        run_started=run_started,
                        knn_cache_size=len(knn_cache),
                    )
        else:
            logger.info(
                "Running %d interventions with %d process workers (mp=%s)",
                n_planned,
                pool_workers,
                mp_method,
            )
            mp_ctx = mp.get_context("spawn")
            payload = SharedEvalPayload(
                results_dir=str(results_dir),
                embeddings_root=str(embeddings_root),
                model=model,
                ref_id=ref_id,
                dataset_id=dataset_id,
                seed=seed,
                k_values=k_values,
                distance_metrics=distance_metrics,
                diffusion_t_values=diffusion_t_values,
                leiden_resolutions=leiden_resolutions,
                cache_path=str(cache_path),
                cell_type_col=cell_type_col,
                batch_col=batch_col,
                stats_shift_pairwise_cell_subsample_n=stats_shift_pairwise_cell_subsample_n,
                stats_shift_pairwise_max_pairs=stats_shift_pairwise_max_pairs,
                knn_alpha=knn_alpha,
                knn_bandwidth_k=knn_bandwidth_k,
                knn_n_null_permutations=knn_n_null_permutations,
            )
            executor_kwargs = {
                "max_workers": pool_workers,
                "mp_context": mp_ctx,
                "initializer": worker_initializer_spawn,
                "initargs": (payload,),
            }

            with ProcessPoolExecutor(**executor_kwargs) as pool:
                futures = {pool.submit(run_intervention_task, task): task for task in tasks}
                for future in as_completed(futures):
                    task = futures[future]
                    job_frames = future.result()
                    if job_frames:
                        frames.extend(job_frames)
                        completed_jobs += 1
                        _log_job_progress(
                            completed_jobs=completed_jobs,
                            total_jobs=total_jobs,
                            run_started=run_started,
                            knn_cache_size=len(knn_cache),
                            int_index=task.int_index,
                            n_planned=n_planned,
                            iid=task.intervention_id,
                        )

        install_shared_context(None)

        if not frames:
            logger.warning("No evaluation rows for model %s", model)
            continue

        t0 = time.perf_counter()
        out_df = pd.concat(frames, ignore_index=True)
        out_df = append_raw_embedding_gain_rows(out_df)
        out_path = evaluation_metrics_csv_path(results_dir, model)
        out_df.to_csv(out_path, index=False)
        logger.info(
            "Model %s finished: wrote %d rows to %s (concat/write %.1fs; model total %.1f min; "
            "kNN cache entries=%d; Leiden ref cache=%d)",
            model,
            len(out_df),
            out_path,
            time.perf_counter() - t0,
            (time.perf_counter() - model_started) / 60.0,
            len(knn_cache),
            len(shared.model_ctx.leiden_cache),
        )

    logger.info(
        "Finished evaluate run for dataset_id=%s (%d/%d jobs in %.1f min)",
        dataset_id,
        completed_jobs,
        total_jobs,
        (time.perf_counter() - run_started) / 60.0,
    )
