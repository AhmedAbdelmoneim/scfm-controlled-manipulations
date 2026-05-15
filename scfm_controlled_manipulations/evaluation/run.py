"""CLI entry for paired reference vs manipulation structure metrics."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from scfm_controlled_manipulations.evaluation.data import load_aligned_bundle
from scfm_controlled_manipulations.evaluation.metrics_cell_batch import (
    ClassifierCacheKey,
    ClassifierCacheValue,
    compute_cell_type_and_batch_metrics,
)
from scfm_controlled_manipulations.evaluation.metrics_clustering import compute_clustering_metrics
from scfm_controlled_manipulations.evaluation.metrics_knn import compute_knn_metrics
from scfm_controlled_manipulations.evaluation.metrics_stats_shift import (
    compute_embedding_shift,
    compute_embedding_stats,
)
from scfm_controlled_manipulations.io import (
    embedding_path,
    evaluation_cache_dir,
    evaluation_dir,
    evaluation_metrics_csv_path,
    intervention_id,
)
from scfm_controlled_manipulations.sweep_config import (
    expand_intervention_specs,
    reference_intervention_id,
)

logger = logging.getLogger(__name__)

DEFAULT_EVALUATION: dict[str, Any] = {
    "k_values": [15, 30, 50],
    "distance_metrics": ["euclidean", "cosine"],
    "diffusion_t_values": [1, 4, 8],
    "leiden_resolutions": [0.5, 1.0],
    "cell_type_col": "cell_type",
    "batch_col": "batch",
    "leiden_resolution_cell_batch": 1.0,
    "dataset_id": None,
}


def merge_evaluation_config(cfg: dict[str, Any]) -> dict[str, Any]:
    merged = dict(DEFAULT_EVALUATION)
    merged.update(cfg.get("evaluation") or {})
    return merged


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
    embeddings_root: Path,
    model: str,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Interventions with an on-disk manipulated embedding for ``model``."""
    planned: list[tuple[str, str, dict[str, Any]]] = []
    for spec in specs:
        name = str(spec["name"])
        kwargs = dict(spec.get("kwargs") or {})
        iid = intervention_id(name, kwargs)
        if iid == ref_id:
            continue
        if embedding_path(embeddings_root, model, iid).is_file():
            planned.append((name, iid, spec))
    return planned


def append_raw_embedding_gain_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Append rows with ``metric_category`` suffix ``_gain`` (embedding minus raw) where both exist."""
    gain_rows: list[dict[str, Any]] = []
    for cat in ("embedding_shift", "knn_metrics", "clustering_metrics"):
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
        block = pd.DataFrame(
            {
                "dataset_id": merged["dataset_id"],
                "model": merged["model"],
                "intervention_id": merged["intervention_id"],
                "intervention_name": merged["intervention_name"],
                "metric_category": f"{cat}_gain",
                "metric_name": merged["metric_name"],
                "space": "embedding_minus_raw",
                "value_mean": merged["value_mean_emb"] - merged["value_mean_raw"],
                "value_median": merged["value_median_emb"] - merged["value_median_raw"],
                "value_std": np.nan,
                "null_value": np.nan,
                "n_cells": merged["n_cells_emb"].astype(int),
                "seed": merged["seed_emb"].astype(int),
            }
        )
        for optional in ("distance_metric", "k", "diffusion_t", "resolution"):
            col = optional
            if col in merged.columns:
                block[col] = merged[col]
        gain_rows.extend(block.to_dict("records"))
    if not gain_rows:
        return df
    return pd.concat([df, pd.DataFrame(gain_rows)], ignore_index=True)


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

    evaluation_dir(results_dir).mkdir(parents=True, exist_ok=True)

    models = list(cfg["models"])
    k_values = [int(k) for k in ev["k_values"]]
    distance_metrics = list(ev["distance_metrics"])
    diffusion_t_values = [int(t) for t in ev["diffusion_t_values"]]
    leiden_resolutions = [float(x) for x in ev["leiden_resolutions"]]

    total_jobs = sum(
        len(_planned_interventions(specs, ref_id=ref_id, embeddings_root=embeddings_root, model=m))
        for m in models
    )

    logger.info(
        "Evaluate: dataset_id=%s results_dir=%s embeddings_root=%s",
        dataset_id,
        results_dir,
        embeddings_root,
    )
    logger.info(
        "Plan: models=%d interventions_with_embeddings=%d "
        "(k=%s metrics=%s diffusion_t=%s leiden_res=%s cache=%s)",
        len(models),
        total_jobs,
        k_values,
        distance_metrics,
        diffusion_t_values,
        leiden_resolutions,
        cache_path,
    )

    completed_jobs = 0

    for model_index, model in enumerate(models, start=1):
        model_started = time.perf_counter()
        planned = _planned_interventions(
            specs, ref_id=ref_id, embeddings_root=embeddings_root, model=model
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

        frames: list[pd.DataFrame] = []
        reference_cache: dict[ClassifierCacheKey, ClassifierCacheValue] = {}

        for int_index, (name, iid, _spec) in enumerate(planned, start=1):
            job_started = time.perf_counter()
            logger.info(
                "  [%d/%d] %s (%s) — loading aligned bundle",
                int_index,
                n_planned,
                iid,
                name,
            )

            try:
                bundle = load_aligned_bundle(
                    results_dir=results_dir,
                    embeddings_root=embeddings_root,
                    model=model,
                    intervention_id=iid,
                    reference_intervention_id=ref_id,
                )
            except FileNotFoundError as err:
                logger.warning("  [%d/%d] %s skipped: %s", int_index, n_planned, iid, err)
                continue
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

            t0 = time.perf_counter()
            frames.append(
                compute_embedding_stats(
                    bundle=bundle,
                    dataset_id=dataset_id,
                    model=model,
                    intervention_id=iid,
                    intervention_name=name,
                    seed=seed,
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
                    cell_type_col=_optional_obs_column(ev.get("cell_type_col")),
                    batch_col=_optional_obs_column(ev.get("batch_col")),
                    k_values=k_values,
                    distance_metrics=distance_metrics,
                    leiden_resolution=float(ev.get("leiden_resolution_cell_batch", 1.0)),
                    reference_cache=reference_cache,
                )
            )
            logger.info(
                "  [%d/%d] %s — cell_type_and_batch_metrics done (%.1fs)",
                int_index,
                n_planned,
                iid,
                time.perf_counter() - t0,
            )

            completed_jobs += 1
            elapsed_run = time.perf_counter() - run_started
            if completed_jobs > 0 and total_jobs > 0:
                eta_s = elapsed_run / completed_jobs * (total_jobs - completed_jobs)
                logger.info(
                    "  [%d/%d] %s — intervention complete (%.1fs); "
                    "overall %d/%d jobs (%.0f%%), ETA ~%.0f min",
                    int_index,
                    n_planned,
                    iid,
                    time.perf_counter() - job_started,
                    completed_jobs,
                    total_jobs,
                    100.0 * completed_jobs / total_jobs,
                    eta_s / 60.0,
                )

        if not frames:
            logger.warning("No evaluation rows for model %s", model)
            continue

        t0 = time.perf_counter()
        out_df = pd.concat(frames, ignore_index=True)
        out_df = append_raw_embedding_gain_rows(out_df)
        out_path = evaluation_metrics_csv_path(results_dir, model)
        out_df.to_csv(out_path, index=False)
        logger.info(
            "Model %s finished: wrote %d rows to %s (concat/write %.1fs; model total %.1f min)",
            model,
            len(out_df),
            out_path,
            time.perf_counter() - t0,
            (time.perf_counter() - model_started) / 60.0,
        )

    logger.info(
        "Finished evaluate run for dataset_id=%s (%d/%d jobs in %.1f min)",
        dataset_id,
        completed_jobs,
        total_jobs,
        (time.perf_counter() - run_started) / 60.0,
    )
