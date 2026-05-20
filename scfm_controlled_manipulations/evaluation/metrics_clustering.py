"""Leiden clustering stability on embeddings: independent clustering on ref vs manip, ARI/NMI."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from scfm_controlled_manipulations.evaluation.leiden_cache import LeidenCache
from scfm_controlled_manipulations.evaluation.metrics_common import (
    DistributionSummary,
    scalar_summary,
    summary_to_row_fields,
)

logger = logging.getLogger(__name__)


def run_leiden_labels(
    mat: np.ndarray,
    *,
    k: int,
    metric: str,
    resolution: float,
    seed: int,
    leiden_cache: LeidenCache | None = None,
) -> np.ndarray:
    if leiden_cache is not None:
        return leiden_cache.labels(mat, k=k, metric=metric, resolution=resolution, seed=seed)
    return LeidenCache().labels(mat, k=k, metric=metric, resolution=resolution, seed=seed)


def clustering_stability(
    ref_mat: np.ndarray,
    man_mat: np.ndarray,
    *,
    k: int,
    metric: str,
    resolution: float,
    seed: int,
    leiden_cache: LeidenCache | None = None,
) -> dict[str, float]:
    ref_clusters = run_leiden_labels(
        ref_mat,
        k=k,
        metric=metric,
        resolution=resolution,
        seed=seed,
        leiden_cache=leiden_cache,
    )
    man_clusters = run_leiden_labels(
        man_mat,
        k=k,
        metric=metric,
        resolution=resolution,
        seed=seed,
        leiden_cache=None,
    )
    return {
        "ari": float(adjusted_rand_score(ref_clusters, man_clusters)),
        "nmi": float(normalized_mutual_info_score(ref_clusters, man_clusters)),
        "n_ref_clusters": float(len(np.unique(ref_clusters))),
        "n_manip_clusters": float(len(np.unique(man_clusters))),
    }


def _row(
    *,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    distance_metric: str,
    k: int,
    resolution: float,
    metric_name: str,
    summary: DistributionSummary,
    n_cells: int,
    seed: int,
    n_ref_clusters: float,
    n_manip_clusters: float,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "model": model,
        "intervention_id": intervention_id,
        "intervention_name": intervention_name,
        "metric_category": "clustering_metrics",
        "metric_name": metric_name,
        "space": "embedding",
        **summary_to_row_fields(summary),
        "null_value": np.nan,
        "n_cells": n_cells,
        "seed": seed,
        "distance_metric": distance_metric,
        "k": k,
        "resolution": resolution,
        "n_ref_clusters": n_ref_clusters,
        "n_manip_clusters": n_manip_clusters,
    }


def compute_clustering_metrics(
    *,
    bundle: Any,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    seed: int,
    distance_metrics: list[str],
    k_values: list[int],
    leiden_resolutions: list[float],
    leiden_cache: LeidenCache | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_cells = bundle.emb_ref.shape[0]
    ref, man = bundle.emb_ref, bundle.emb_man

    n_jobs = len(distance_metrics) * len(k_values) * len(leiden_resolutions)
    logger.info(
        "clustering_metrics: intervention=%s n_cells=%d (%d Leiden pairs)",
        intervention_id,
        n_cells,
        n_jobs,
    )

    job_i = 0
    for metric in distance_metrics:
        for k in k_values:
            for resolution in leiden_resolutions:
                job_i += 1
                logger.info(
                    "clustering_metrics: Leiden %d/%d metric=%s k=%d resolution=%s",
                    job_i,
                    n_jobs,
                    metric,
                    k,
                    resolution,
                )
                t0 = time.perf_counter()
                stats = clustering_stability(
                    ref,
                    man,
                    k=k,
                    metric=metric,
                    resolution=resolution,
                    seed=seed,
                    leiden_cache=leiden_cache,
                )
                logger.info(
                    "clustering_metrics: Leiden %d/%d done in %.1fs (ARI=%.4f NMI=%.4f)",
                    job_i,
                    n_jobs,
                    time.perf_counter() - t0,
                    stats["ari"],
                    stats["nmi"],
                )
                for mn, val in (
                    ("leiden_ari", stats["ari"]),
                    ("leiden_nmi", stats["nmi"]),
                ):
                    rows.append(
                        _row(
                            dataset_id=dataset_id,
                            model=model,
                            intervention_id=intervention_id,
                            intervention_name=intervention_name,
                            distance_metric=metric,
                            k=k,
                            resolution=resolution,
                            metric_name=mn,
                            summary=scalar_summary(val),
                            n_cells=n_cells,
                            seed=seed,
                            n_ref_clusters=stats["n_ref_clusters"],
                            n_manip_clusters=stats["n_manip_clusters"],
                        )
                    )

    return pd.DataFrame(rows)
