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

import pandas as pd

from scfm_controlled_manipulations.evaluation.context import load_dataset_context
from scfm_controlled_manipulations.evaluation.worker import (
    InterventionTask,
    SharedEvalPayload,
    build_shared_context,
    install_shared_context,
    run_intervention_task,
    worker_bootstrap_fingerprint,
    worker_bootstrap_path,
    worker_initializer_spawn,
    write_worker_bootstrap,
)
from scfm_controlled_manipulations.io import (
    embedding_path,
    evaluation_cache_dir,
    evaluation_dir,
    evaluation_metrics_csv_path,
    intervention_id,
    manipulation_path,
    manipulations_dir,
)
from scfm_controlled_manipulations.sweep_config import (
    expand_intervention_specs,
    reference_intervention_id,
)

logger = logging.getLogger(__name__)

DEFAULT_EVALUATION: dict[str, Any] = {
    "k_values": [5, 15, 50],
    "distance_metrics": ["euclidean"],
    "leiden_resolutions": [0.25, 0.5, 1.0, 2.0],
    "cell_type_col": "cell_type",
    "batch_col": "batch",
    "dataset_id": None,
    "evaluation_workers": 1,
    "evaluation_setup_threads": 1,
    "evaluation_worker_threads": 1,
    "stats_shift_pairwise_cell_subsample_n": 500,
    "stats_shift_pairwise_max_pairs": 10_000,
    "scib_benchmark_n_jobs": 1,
    "distance_correlation_subsample_n": None,
    "trajectory_key": "trajectory",
    "trajectory_n_neighbors": 15,
    "trajectory_n_dcs": 10,
    "trajectory_n_permutations": 10,
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
    validated["leiden_resolutions"] = _require_positive_float_list(
        validated["leiden_resolutions"], key="leiden_resolutions"
    )
    distance_metrics = [str(m).strip() for m in validated["distance_metrics"]]
    if not distance_metrics or any(not m for m in distance_metrics):
        raise ValueError("evaluation.distance_metrics must be a non-empty list of strings")
    validated["distance_metrics"] = distance_metrics

    validated["evaluation_workers"] = max(1, int(validated["evaluation_workers"]))
    validated["evaluation_setup_threads"] = max(
        1, int(validated.get("evaluation_setup_threads", 1))
    )
    validated["evaluation_worker_threads"] = max(
        1, int(validated.get("evaluation_worker_threads", 1))
    )

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

    validated["scib_benchmark_n_jobs"] = max(1, int(validated.get("scib_benchmark_n_jobs", 1)))
    trajectory_key = str(validated.get("trajectory_key", "trajectory")).strip()
    if not trajectory_key:
        raise ValueError("evaluation.trajectory_key must be a non-empty string")
    validated["trajectory_key"] = trajectory_key
    validated["trajectory_n_neighbors"] = max(1, int(validated.get("trajectory_n_neighbors", 15)))
    validated["trajectory_n_dcs"] = max(1, int(validated.get("trajectory_n_dcs", 10)))
    trajectory_n_permutations = int(validated.get("trajectory_n_permutations", 10))
    if trajectory_n_permutations < 0:
        raise ValueError("evaluation.trajectory_n_permutations must be >= 0")
    validated["trajectory_n_permutations"] = trajectory_n_permutations
    dist_corr_sub = validated.get("distance_correlation_subsample_n")
    if dist_corr_sub is None:
        validated["distance_correlation_subsample_n"] = validated[
            "stats_shift_pairwise_cell_subsample_n"
        ]
    else:
        dist_corr_sub = int(dist_corr_sub)
        if dist_corr_sub <= 0:
            raise ValueError("evaluation.distance_correlation_subsample_n must be > 0 or null")
        validated["distance_correlation_subsample_n"] = dist_corr_sub

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
    manip_dir: Path,
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
        if not manipulation_path(results_dir, iid, manip_dir).is_file():
            continue
        if embedding_path(embeddings_root, model, iid).is_file():
            planned.append((name, iid, spec))
    return planned


def _log_job_progress(
    *,
    completed_jobs: int,
    total_jobs: int,
    run_started: float,
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
            "  [%d/%d] %s complete; overall %d/%d jobs (%.0f%%), ETA ~%.0f min",
            int_index,
            n_planned,
            iid,
            completed_jobs,
            total_jobs,
            100.0 * completed_jobs / total_jobs,
            eta_s / 60.0,
        )
    else:
        logger.info(
            "  overall %d/%d jobs (%.0f%%), ETA ~%.0f min",
            completed_jobs,
            total_jobs,
            100.0 * completed_jobs / total_jobs,
            eta_s / 60.0,
        )


def run_evaluate(cfg: dict[str, Any]) -> None:
    ev = merge_evaluation_config(cfg)
    run_started = time.perf_counter()

    results_dir = Path(cfg["results_dir"])
    manip_dir = manipulations_dir(results_dir, cfg.get("manipulations_dir"))
    embeddings_root = Path(cfg["embeddings_root"])
    ref_id = reference_intervention_id(cfg)
    specs = expand_intervention_specs(cfg["interventions"])
    seed = int(cfg.get("seed", 42))
    dataset_id = _dataset_id(cfg, ev)
    cache_path = evaluation_cache_dir(results_dir)
    evaluation_workers = max(1, int(ev.get("evaluation_workers", 1)))
    evaluation_setup_threads = max(1, int(ev.get("evaluation_setup_threads", 1)))
    evaluation_worker_threads = max(1, int(ev.get("evaluation_worker_threads", 1)))

    evaluation_dir(results_dir).mkdir(parents=True, exist_ok=True)

    models = list(cfg["models"])
    k_values = [int(k) for k in ev["k_values"]]
    distance_metrics = list(ev["distance_metrics"])
    leiden_resolutions = [float(x) for x in ev["leiden_resolutions"]]
    stats_shift_pairwise_cell_subsample_n = int(
        ev.get("stats_shift_pairwise_cell_subsample_n", 500)
    )
    pairwise_max_raw = ev.get("stats_shift_pairwise_max_pairs")
    stats_shift_pairwise_max_pairs = (
        int(pairwise_max_raw) if pairwise_max_raw is not None else None
    )
    distance_correlation_subsample_n = int(ev["distance_correlation_subsample_n"])

    total_jobs = sum(
        len(
            _planned_interventions(
                specs,
                ref_id=ref_id,
                results_dir=results_dir,
                manip_dir=manip_dir,
                embeddings_root=embeddings_root,
                model=m,
            )
        )
        for m in models
    )

    logger.info(
        "Evaluate: dataset_id=%s results_dir=%s manipulations_dir=%s embeddings_root=%s",
        dataset_id,
        results_dir,
        manip_dir,
        embeddings_root,
    )
    mp_method = mp.get_context("spawn").get_start_method()
    logger.info(
        "Plan: models=%d interventions_with_embeddings=%d "
        "(k=%s metrics=%s leiden_res=%s cache=%s workers=%d mp=%s)",
        len(models),
        total_jobs,
        k_values,
        distance_metrics,
        leiden_resolutions,
        cache_path,
        evaluation_workers,
        mp_method,
    )
    logger.info(
        "Thread budget: setup_threads=%d worker_threads=%d process_workers=%d (spawn). "
        "Workers load a pre-built bootstrap snapshot when using a process pool.",
        evaluation_setup_threads,
        evaluation_worker_threads,
        evaluation_workers,
    )

    logger.info("Loading shared reference obs (once per dataset)")
    t0 = time.perf_counter()
    dataset_ctx = load_dataset_context(results_dir, manip_dir)
    logger.info(
        "Reference obs loaded: n_cells=%d (%.1fs)",
        dataset_ctx.n_cells,
        time.perf_counter() - t0,
    )

    completed_jobs = 0

    for model_index, model in enumerate(models, start=1):
        model_started = time.perf_counter()
        planned = _planned_interventions(
            specs,
            ref_id=ref_id,
            results_dir=results_dir,
            manip_dir=manip_dir,
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

        ref_embedding_path = embedding_path(embeddings_root, model, ref_id)
        if not ref_embedding_path.is_file():
            logger.warning(
                "Reference embedding missing for model %s; skipping model for dataset %s: %s",
                model,
                dataset_id,
                ref_embedding_path,
            )
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
            leiden_resolutions=leiden_resolutions,
            cache_path=cache_path,
            stats_shift_pairwise_cell_subsample_n=stats_shift_pairwise_cell_subsample_n,
            stats_shift_pairwise_max_pairs=stats_shift_pairwise_max_pairs,
            distance_correlation_subsample_n=distance_correlation_subsample_n,
        )
        logger.info(
            "Reference context ready (%.1fs; Leiden cache=%d; stats cache=%s)",
            time.perf_counter() - t0,
            len(shared.model_ctx.leiden_cache),
            shared.model_ctx.ref_stats_cache is not None,
        )

        pool_workers = min(evaluation_workers, n_planned)
        bootstrap_path: Path | None = None
        if pool_workers > 1:
            fingerprint = worker_bootstrap_fingerprint(
                dataset_id=dataset_id,
                model=model,
                seed=seed,
                k_values=k_values,
                distance_metrics=distance_metrics,
                leiden_resolutions=leiden_resolutions,
                distance_correlation_subsample_n=distance_correlation_subsample_n,
            )
            bootstrap_path = worker_bootstrap_path(
                cache_path, model=model, fingerprint=fingerprint
            )
            write_worker_bootstrap(bootstrap_path, shared)

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
                manipulations_dir=str(manip_dir),
                embeddings_root=str(embeddings_root),
                model=model,
                ref_id=ref_id,
                dataset_id=dataset_id,
                seed=seed,
                k_values=k_values,
                distance_metrics=distance_metrics,
                leiden_resolutions=leiden_resolutions,
                cache_path=str(cache_path),
                stats_shift_pairwise_cell_subsample_n=stats_shift_pairwise_cell_subsample_n,
                stats_shift_pairwise_max_pairs=stats_shift_pairwise_max_pairs,
                distance_correlation_subsample_n=distance_correlation_subsample_n,
                bootstrap_path=str(bootstrap_path) if bootstrap_path is not None else None,
                worker_threads=evaluation_worker_threads,
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
        out_path = evaluation_metrics_csv_path(results_dir, model)
        out_df.to_csv(out_path, index=False)
        logger.info(
            "Model %s finished: wrote %d rows to %s (concat/write %.1fs; model total %.1f min; "
            "Leiden ref cache=%d)",
            model,
            len(out_df),
            out_path,
            time.perf_counter() - t0,
            (time.perf_counter() - model_started) / 60.0,
            len(shared.model_ctx.leiden_cache),
        )

    logger.info(
        "Finished evaluate run for dataset_id=%s (%d/%d jobs in %.1f min)",
        dataset_id,
        completed_jobs,
        total_jobs,
        (time.perf_counter() - run_started) / 60.0,
    )
