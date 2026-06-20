"""CLI entry for reference-only scIB Benchmarker metrics."""

from __future__ import annotations

from scfm_controlled_manipulations.compute_env import apply_thread_limits

apply_thread_limits(threads_per_process=1)

import logging
from pathlib import Path
import time
from typing import Any

import pandas as pd

from scfm_controlled_manipulations.evaluation.context import (
    load_dataset_context,
    load_model_context,
)
from scfm_controlled_manipulations.evaluation.data import load_manipulation_counts
from scfm_controlled_manipulations.evaluation.metrics_cell_batch import (
    compute_cell_batch_reference_rows,
    log_cell_batch_obs_columns,
)
from scfm_controlled_manipulations.evaluation.run import (
    _dataset_id,
    _optional_obs_column,
    merge_evaluation_config,
)
from scfm_controlled_manipulations.io import (
    embedding_path,
    evaluation_dir,
    evaluation_scib_metrics_csv_path,
    manipulations_dir,
)
from scfm_controlled_manipulations.obs_columns import (
    resolve_batch_column,
    resolve_cell_type_column_for_dataset,
)
from scfm_controlled_manipulations.sweep_config import reference_intervention_id

logger = logging.getLogger(__name__)


def run_evaluate_scib(cfg: dict[str, Any]) -> None:
    ev = merge_evaluation_config(cfg)
    run_started = time.perf_counter()

    results_dir = Path(cfg["results_dir"])
    manip_dir = manipulations_dir(results_dir, cfg.get("manipulations_dir"))
    embeddings_root = Path(cfg["embeddings_root"])
    ref_id = reference_intervention_id(cfg)
    seed = int(cfg.get("seed", 42))
    dataset_id = _dataset_id(cfg, ev)
    models = list(cfg["models"])
    scib_benchmark_n_jobs = max(1, int(ev.get("scib_benchmark_n_jobs", 1)))
    cell_type_col_config = _optional_obs_column(ev.get("cell_type_col"))
    batch_col_config = _optional_obs_column(ev.get("batch_col"))

    evaluation_dir(results_dir).mkdir(parents=True, exist_ok=True)

    logger.info(
        "Evaluate-scib: dataset_id=%s results_dir=%s manipulations_dir=%s embeddings_root=%s ref_id=%s",
        dataset_id,
        results_dir,
        manip_dir,
        embeddings_root,
        ref_id,
    )

    t0 = time.perf_counter()
    dataset_ctx = load_dataset_context(results_dir, manip_dir)
    obs_cols = dataset_ctx.obs.columns
    cell_type_col = resolve_cell_type_column_for_dataset(
        obs_cols,
        cell_type_col_config,
        dataset_id=dataset_id,
    )
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
            "scib bio/batch metrics skipped",
            cell_type_col_config,
        )
    if batch_col_config and batch_col != batch_col_config:
        logger.info(
            "Resolved batch_col %r -> %r in reference obs",
            batch_col_config,
            batch_col,
        )
    logger.info(
        "Reference obs loaded: n_cells=%d (%.1fs)",
        dataset_ctx.n_cells,
        time.perf_counter() - t0,
    )
    log_cell_batch_obs_columns(
        dataset_ctx.obs,
        cell_type_col=cell_type_col,
        batch_col=batch_col,
    )

    models_written = 0
    for model_index, model in enumerate(models, start=1):
        emb_path = embedding_path(embeddings_root, model, ref_id)
        if not emb_path.is_file():
            logger.warning(
                "Model %d/%d %s: reference embedding missing at %s; skipping",
                model_index,
                len(models),
                model,
                emb_path,
            )
            continue

        logger.info(
            "Model %d/%d %s: running scIB benchmark on reference", model_index, len(models), model
        )
        t0 = time.perf_counter()
        model_ctx = load_model_context(
            embeddings_root, model, ref_id, target_obs=dataset_ctx.obs.index
        )
        rows = compute_cell_batch_reference_rows(
            counts=load_manipulation_counts(results_dir, ref_id, manip_dir),
            mat=model_ctx.emb_ref,
            obs_df=dataset_ctx.obs,
            space_label="embedding_reference",
            dataset_id=dataset_id,
            model=model,
            intervention_id=ref_id,
            intervention_name=ref_id,
            seed=seed,
            cell_type_col=cell_type_col,
            batch_col=batch_col,
            n_cells=dataset_ctx.n_cells,
            n_jobs=scib_benchmark_n_jobs,
        )
        if not rows:
            logger.warning("Model %s: no scIB rows produced; skipping CSV write", model)
            continue

        out_df = pd.DataFrame(rows)
        out_path = evaluation_scib_metrics_csv_path(results_dir, model)
        out_df.to_csv(out_path, index=False)
        models_written += 1
        logger.info(
            "Model %s finished: wrote %d rows to %s (%.1fs)",
            model,
            len(out_df),
            out_path,
            time.perf_counter() - t0,
        )

    logger.info(
        "Finished evaluate-scib for dataset_id=%s (%d/%d models in %.1f min)",
        dataset_id,
        models_written,
        len(models),
        (time.perf_counter() - run_started) / 60.0,
    )
