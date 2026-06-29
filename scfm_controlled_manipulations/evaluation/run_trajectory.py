"""CLI entry for reference-only trajectory inference metrics."""

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
from scfm_controlled_manipulations.evaluation.metrics_trajectory import (
    compute_trajectory_reference_rows,
)
from scfm_controlled_manipulations.evaluation.run import (
    _dataset_id,
    merge_evaluation_config,
)
from scfm_controlled_manipulations.io import (
    embedding_path,
    evaluation_dir,
    evaluation_trajectory_metrics_csv_path,
    manipulations_dir,
)
from scfm_controlled_manipulations.sweep_config import reference_intervention_id

logger = logging.getLogger(__name__)


def run_evaluate_trajectory(cfg: dict[str, Any]) -> None:
    ev = merge_evaluation_config(cfg)
    run_started = time.perf_counter()

    results_dir = Path(cfg["results_dir"])
    manip_dir = manipulations_dir(results_dir, cfg.get("manipulations_dir"))
    embeddings_root = Path(cfg["embeddings_root"])
    ref_id = reference_intervention_id(cfg)
    seed = int(cfg.get("seed", 42))
    dataset_id = _dataset_id(cfg, ev)
    models = list(cfg["models"])
    trajectory_key = str(ev["trajectory_key"])
    n_neighbors = int(ev["trajectory_n_neighbors"])
    n_dcs = int(ev["trajectory_n_dcs"])
    n_permutations = int(ev["trajectory_n_permutations"])

    evaluation_dir(results_dir).mkdir(parents=True, exist_ok=True)

    logger.info(
        "Evaluate-trajectory: dataset_id=%s results_dir=%s manipulations_dir=%s "
        "embeddings_root=%s ref_id=%s trajectory_key=%s n_neighbors=%d n_dcs=%d n_perm=%d",
        dataset_id,
        results_dir,
        manip_dir,
        embeddings_root,
        ref_id,
        trajectory_key,
        n_neighbors,
        n_dcs,
        n_permutations,
    )

    t0 = time.perf_counter()
    dataset_ctx = load_dataset_context(results_dir, manip_dir)
    logger.info(
        "Reference obs loaded: n_cells=%d columns=%d (%.1fs)",
        dataset_ctx.n_cells,
        len(dataset_ctx.obs.columns),
        time.perf_counter() - t0,
    )
    if trajectory_key not in dataset_ctx.obs.columns:
        logger.warning(
            "Trajectory column %r not found in reference obs; no trajectory metrics will be written",
            trajectory_key,
        )
        return

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
            "Model %d/%d %s: running trajectory metrics on reference",
            model_index,
            len(models),
            model,
        )
        t0 = time.perf_counter()
        model_ctx = load_model_context(
            embeddings_root, model, ref_id, target_obs=dataset_ctx.obs.index
        )
        rows = compute_trajectory_reference_rows(
            mat=model_ctx.emb_ref,
            obs_df=dataset_ctx.obs,
            trajectory_key=trajectory_key,
            space_label="embedding_reference",
            dataset_id=dataset_id,
            model=model,
            intervention_id=ref_id,
            intervention_name=ref_id,
            seed=seed,
            n_neighbors=n_neighbors,
            n_dcs=n_dcs,
            n_permutations=n_permutations,
        )
        if not rows:
            logger.warning("Model %s: no trajectory rows produced; skipping CSV write", model)
            continue

        out_df = pd.DataFrame(rows)
        out_path = evaluation_trajectory_metrics_csv_path(results_dir, model)
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
        "Finished evaluate-trajectory for dataset_id=%s (%d/%d models in %.1f min)",
        dataset_id,
        models_written,
        len(models),
        (time.perf_counter() - run_started) / 60.0,
    )
