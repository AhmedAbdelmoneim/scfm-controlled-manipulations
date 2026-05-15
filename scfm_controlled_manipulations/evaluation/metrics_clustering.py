"""Leiden clustering stability on embeddings: independent clustering on ref vs manip, ARI/NMI."""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from scfm_controlled_manipulations.evaluation.metrics_common import scalar_summary


def run_leiden_labels(
    mat: np.ndarray,
    *,
    k: int,
    metric: str,
    resolution: float,
    seed: int,
) -> np.ndarray:
    adata_tmp = ad.AnnData(mat)
    sc.pp.neighbors(
        adata_tmp,
        n_neighbors=k,
        metric=metric,
        use_rep="X",
        random_state=seed,
    )
    sc.tl.leiden(
        adata_tmp,
        resolution=resolution,
        random_state=seed,
        key_added="leiden_eval",
    )
    return adata_tmp.obs["leiden_eval"].astype(str).to_numpy()


def clustering_stability(
    ref_mat: np.ndarray,
    man_mat: np.ndarray,
    *,
    k: int,
    metric: str,
    resolution: float,
    seed: int,
) -> dict[str, float]:
    ref_clusters = run_leiden_labels(ref_mat, k=k, metric=metric, resolution=resolution, seed=seed)
    man_clusters = run_leiden_labels(man_mat, k=k, metric=metric, resolution=resolution, seed=seed)
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
    value_mean: float,
    value_median: float,
    value_std: float,
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
        "value_mean": value_mean,
        "value_median": value_median,
        "value_std": value_std,
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
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_cells = bundle.emb_ref.shape[0]
    ref, man = bundle.emb_ref, bundle.emb_man

    for metric in distance_metrics:
        for k in k_values:
            for resolution in leiden_resolutions:
                stats = clustering_stability(
                    ref, man, k=k, metric=metric, resolution=resolution, seed=seed
                )
                for mn, val in (
                    ("leiden_ari", stats["ari"]),
                    ("leiden_nmi", stats["nmi"]),
                ):
                    vm, vmed, vs = scalar_summary(val)
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
                            value_mean=vm,
                            value_median=vmed,
                            value_std=vs,
                            n_cells=n_cells,
                            seed=seed,
                            n_ref_clusters=stats["n_ref_clusters"],
                            n_manip_clusters=stats["n_manip_clusters"],
                        )
                    )

    return pd.DataFrame(rows)
