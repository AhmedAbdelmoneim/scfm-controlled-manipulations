"""Cell-type and batch structure metrics (silhouette, neighbor purity, AUROC/AP, permutation null)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

from scfm_controlled_manipulations.evaluation.leiden_cache import LeidenCache
from scfm_controlled_manipulations.evaluation.metrics_clustering import run_leiden_labels
from scfm_controlled_manipulations.evaluation.metrics_common import (
    DistributionSummary,
    distribution_summary,
    scalar_summary,
    summary_to_row_fields,
)
from scfm_controlled_manipulations.evaluation.metrics_knn import knn_neighbors

KnnIndexCacheLike = Any

logger = logging.getLogger(__name__)

ClassifierCacheKey = tuple[str, str]
ClassifierCacheValue = tuple[float, float, float, float, float]


def _obs_col_present(obs_df: pd.DataFrame, col: str | None) -> bool:
    return col is not None and col in obs_df.columns


def safe_silhouette(mat: Any, labels: np.ndarray, *, metric: str, seed: int) -> float:
    labels_arr = np.asarray(labels).astype(str)
    if len(np.unique(labels_arr)) < 2:
        return float("nan")
    if len(np.unique(labels_arr)) >= len(labels_arr):
        return float("nan")
    return float(
        silhouette_score(mat, labels_arr, metric=metric, sample_size=None, random_state=seed)
    )


def _encode_labels(labels: np.ndarray) -> tuple[np.ndarray, int]:
    _, inverse = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    return inverse.astype(np.intp), int(inverse.max()) + 1


def _neighbor_same_label_fraction_per_cell(
    inverse_labels: np.ndarray, neighbor_idx: np.ndarray
) -> np.ndarray:
    neighbor_codes = inverse_labels[neighbor_idx]
    return np.mean(neighbor_codes == inverse_labels[:, None], axis=1)


def _neighbor_label_count_matrix(
    inverse_labels: np.ndarray,
    neighbor_idx: np.ndarray,
    n_labels: int,
) -> np.ndarray:
    """Neighbor-label counts per cell, shape ``(n_cells, n_labels)``."""
    n_cells, k = neighbor_idx.shape
    neighbor_codes = inverse_labels[neighbor_idx]
    row_ids = np.broadcast_to(np.arange(n_cells, dtype=np.intp)[:, None], (n_cells, k)).ravel()
    counts = np.zeros((n_cells, n_labels), dtype=np.float64)
    np.add.at(counts, (row_ids, neighbor_codes.ravel()), 1.0)
    return counts


def _neighbor_label_entropy_norm_per_cell(
    inverse_labels: np.ndarray, neighbor_idx: np.ndarray
) -> np.ndarray:
    n_labels = int(inverse_labels.max()) + 1
    counts = _neighbor_label_count_matrix(inverse_labels, neighbor_idx, n_labels)
    k = neighbor_idx.shape[1]
    probs = counts / k
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(probs > 0, np.log(probs), 0.0)
        entropy = -np.sum(probs * log_p, axis=1)
    max_entropy = np.log(n_labels) if n_labels > 1 else 1.0
    return entropy / max_entropy


def _ilisi_like_score_per_cell(inverse_labels: np.ndarray, neighbor_idx: np.ndarray) -> np.ndarray:
    n_labels = int(inverse_labels.max()) + 1
    counts = _neighbor_label_count_matrix(inverse_labels, neighbor_idx, n_labels)
    k = neighbor_idx.shape[1]
    probs = counts / k
    return 1.0 / np.maximum(np.sum(probs**2, axis=1), 1e-12)


def label_cluster_agreement(
    mat: np.ndarray,
    labels: np.ndarray,
    *,
    k: int,
    metric: str,
    resolution: float,
    seed: int,
    leiden_cache: LeidenCache | None = None,
) -> tuple[float, float]:
    labels_arr = np.asarray(labels).astype(str)
    cluster_labels = run_leiden_labels(
        mat,
        k=k,
        metric=metric,
        resolution=resolution,
        seed=seed,
        leiden_cache=leiden_cache,
    )
    return (
        float(adjusted_rand_score(labels_arr, cluster_labels)),
        float(normalized_mutual_info_score(labels_arr, cluster_labels)),
    )


def _cv_splits_for_labels(y: np.ndarray, n_splits: int) -> int | None:
    """Stratified fold count that fits label counts; ``None`` if CV is not viable."""
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2:
        return None
    min_count = int(counts.min())
    if min_count < 2:
        return None
    return min(n_splits, min_count)


def _mask_labels_for_stratified_cv(y: np.ndarray, n_splits: int) -> np.ndarray | None:
    """Drop classes with fewer than ``n_splits`` samples so StratifiedKFold is valid."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    keep_classes = classes[counts >= n_splits]
    if len(keep_classes) < 2:
        return None
    return np.isin(y, keep_classes)


def _ovr_roc_ap_cv(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    n_splits: int = 3,
) -> tuple[float, float, float, float]:
    """Returns mean roc_auc (ovr macro), std, mean ap (macro), std."""
    y = np.asarray(y)
    n_splits_eff = _cv_splits_for_labels(y, n_splits)
    if n_splits_eff is None:
        return float("nan"), float("nan"), float("nan"), float("nan")

    keep = _mask_labels_for_stratified_cv(y, n_splits_eff)
    if keep is None:
        return float("nan"), float("nan"), float("nan"), float("nan")
    x = x[keep]
    y = y[keep]

    pipe = make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(
            LogisticRegression(
                max_iter=400,
                class_weight="balanced",
                random_state=seed,
            )
        ),
    )
    cv = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)
    rocs = []
    aps = []
    for train_idx, test_idx in cv.split(x, y):
        pipe.fit(x[train_idx], y[train_idx])
        y_test = y[test_idx]
        classes = pipe.classes_
        if len(np.unique(y_test)) < 2 or len(np.unique(y_test)) < len(classes):
            continue
        prob = pipe.predict_proba(x[test_idx])
        try:
            rocs.append(
                roc_auc_score(y_test, prob, multi_class="ovr", average="macro", labels=classes)
            )
            y_ohe = label_binarize(y_test, classes=classes)
            if y_ohe.ndim == 1:
                y_ohe = y_ohe.reshape(-1, 1)
            if y_ohe.shape[1] == prob.shape[1] and np.any(y_ohe):
                aps.append(average_precision_score(y_ohe, prob, average="macro"))
        except ValueError:
            continue
    if not rocs:
        return float("nan"), float("nan"), float("nan"), float("nan")
    ap_mean = float(np.mean(aps)) if aps else float("nan")
    ap_std = float(np.std(aps)) if aps else float("nan")
    return float(np.mean(rocs)), float(np.std(rocs)), ap_mean, ap_std


def _permutation_null_roc(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    n_splits: int = 3,
) -> float:
    rng = np.random.default_rng(seed + 99991)
    y_perm = rng.permutation(y)
    roc_mean, _, _, _ = _ovr_roc_ap_cv(x, y_perm, seed=seed + 1, n_splits=n_splits)
    return roc_mean


def _as_dense_if_small(mat: Any, max_features: int) -> np.ndarray | None:
    n_features = mat.shape[1]
    if n_features > max_features:
        return None
    if hasattr(mat, "todense"):
        return np.asarray(mat.todense(), dtype=np.float32)
    return np.asarray(mat, dtype=np.float32)


def _classifier_metrics(
    x_dense: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    cache_key: ClassifierCacheKey,
    reference_cache: dict[ClassifierCacheKey, ClassifierCacheValue],
    is_reference: bool,
) -> ClassifierCacheValue:
    if is_reference and cache_key in reference_cache:
        return reference_cache[cache_key]
    roc_m, roc_s, ap_m, ap_s = _ovr_roc_ap_cv(x_dense, y, seed=seed)
    null_v = _permutation_null_roc(x_dense, y, seed=seed)
    result = (roc_m, roc_s, ap_m, ap_s, null_v)
    if is_reference:
        reference_cache[cache_key] = result
    return result


def _append_metadata_rows(
    rows: list[dict[str, Any]],
    *,
    mat: Any,
    obs_df: pd.DataFrame,
    space_label: str,
    cell_type_col: str | None,
    batch_col: str | None,
    k_values: list[int],
    distance_metrics: list[str],
    leiden_resolution: float,
    seed: int,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    n_cells: int,
    reference_cache: dict[ClassifierCacheKey, ClassifierCacheValue],
    knn_cache: KnnIndexCacheLike | None = None,
    leiden_cache: LeidenCache | None = None,
    max_features_for_classifier: int = 2048,
) -> None:
    is_reference = space_label.endswith("_reference")
    run_leiden = space_label.startswith("embedding")
    leiden_cache_for_space = leiden_cache if is_reference else None
    k_sorted = sorted(int(k) for k in k_values)
    k_max = k_sorted[-1]

    x_dense = _as_dense_if_small(mat, max_features_for_classifier)
    x_leiden = x_dense if run_leiden else None

    metadata_specs: list[tuple[str, str, int, np.ndarray]] = []
    if _obs_col_present(obs_df, cell_type_col):
        y_ct = obs_df[cell_type_col].astype(str).to_numpy()
        metadata_specs.append(("cell_type", cell_type_col, seed, _encode_labels(y_ct)[0]))
    if _obs_col_present(obs_df, batch_col):
        y_b = obs_df[batch_col].astype(str).to_numpy()
        metadata_specs.append(("batch", batch_col, seed + 7, _encode_labels(y_b)[0]))

    for metadata_type, col, clf_seed, inverse_labels in metadata_specs:
        y = obs_df[col].astype(str).to_numpy()
        if x_dense is not None:
            roc_m, roc_s, ap_m, ap_s, null_v = _classifier_metrics(
                x_dense,
                y,
                seed=clf_seed,
                cache_key=(space_label, metadata_type),
                reference_cache=reference_cache,
                is_reference=is_reference,
            )
        else:
            roc_m, roc_s, ap_m, ap_s, null_v = (np.nan, np.nan, np.nan, np.nan, np.nan)

        clf_base = {
            "dataset_id": dataset_id,
            "model": model,
            "intervention_id": intervention_id,
            "intervention_name": intervention_name,
            "metric_category": "cell_type_and_batch_metrics",
            "space": space_label,
            "seed": seed,
            "n_cells": n_cells,
            "distance_metric": distance_metrics[0],
            "k": k_sorted[0],
            "leiden_resolution": leiden_resolution,
            "metadata_type": metadata_type,
        }
        roc_summary = DistributionSummary(
            mean=roc_m,
            median=roc_m,
            std=roc_s,
            min=roc_m,
            max=roc_m,
            q05=float("nan"),
            q25=float("nan"),
            q75=float("nan"),
            q95=float("nan"),
        )
        ap_summary = DistributionSummary(
            mean=ap_m,
            median=ap_m,
            std=ap_s,
            min=ap_m,
            max=ap_m,
            q05=float("nan"),
            q25=float("nan"),
            q75=float("nan"),
            q95=float("nan"),
        )
        rows.append(
            {
                **clf_base,
                "metric_name": "classifier_roc_auc_ovr_macro_cv_mean",
                **summary_to_row_fields(roc_summary),
                "null_value": null_v,
            }
        )
        rows.append(
            {
                **clf_base,
                "metric_name": "classifier_ap_macro_cv_mean",
                **summary_to_row_fields(ap_summary),
                "null_value": np.nan,
            }
        )

    if knn_cache is not None:
        neighbor_idx_by_metric = {
            metric: knn_cache.neighbors(mat, k_max, metric)[1] for metric in distance_metrics
        }
    else:
        neighbor_idx_by_metric = {
            metric: knn_neighbors(mat, k_max, metric)[1] for metric in distance_metrics
        }

    silhouette_cache: dict[tuple[str, str], DistributionSummary] = {}
    for metric in distance_metrics:
        for metadata_type, col, _, _ in metadata_specs:
            y = obs_df[col].astype(str).to_numpy()
            silhouette_cache[(metric, metadata_type)] = scalar_summary(
                safe_silhouette(mat, y, metric=metric, seed=seed)
            )

    for metric in distance_metrics:
        idx_max = neighbor_idx_by_metric[metric]
        for k in k_sorted:
            neighbor_idx = idx_max[:, :k]
            for metadata_type, col, _, inverse_labels in metadata_specs:
                base = {
                    "dataset_id": dataset_id,
                    "model": model,
                    "intervention_id": intervention_id,
                    "intervention_name": intervention_name,
                    "metric_category": "cell_type_and_batch_metrics",
                    "space": space_label,
                    "seed": seed,
                    "n_cells": n_cells,
                    "distance_metric": metric,
                    "k": k,
                    "leiden_resolution": leiden_resolution,
                    "metadata_type": metadata_type,
                }
                rows.append(
                    {
                        **base,
                        "metric_name": "silhouette",
                        **summary_to_row_fields(silhouette_cache[(metric, metadata_type)]),
                        "null_value": np.nan,
                    }
                )
                rows.append(
                    {
                        **base,
                        "metric_name": "neighbor_same_label_fraction",
                        **summary_to_row_fields(
                            distribution_summary(
                                _neighbor_same_label_fraction_per_cell(
                                    inverse_labels, neighbor_idx
                                )
                            )
                        ),
                        "null_value": np.nan,
                    }
                )
                if metadata_type == "cell_type":
                    rows.append(
                        {
                            **base,
                            "metric_name": "neighbor_label_entropy_norm",
                            **summary_to_row_fields(
                                distribution_summary(
                                    _neighbor_label_entropy_norm_per_cell(
                                        inverse_labels, neighbor_idx
                                    )
                                )
                            ),
                            "null_value": np.nan,
                        }
                    )
                    rows.append(
                        {
                            **base,
                            "metric_name": "ilisi_like_inverse_simpson",
                            **summary_to_row_fields(
                                distribution_summary(
                                    _ilisi_like_score_per_cell(inverse_labels, neighbor_idx)
                                )
                            ),
                            "null_value": np.nan,
                        }
                    )
                    if x_leiden is not None:
                        ari, nmi = label_cluster_agreement(
                            x_leiden,
                            y,
                            k=k,
                            metric=metric,
                            resolution=leiden_resolution,
                            seed=seed,
                            leiden_cache=leiden_cache_for_space,
                        )
                    else:
                        ari, nmi = (float("nan"), float("nan"))
                    rows.append(
                        {
                            **base,
                            "metric_name": "label_vs_leiden_ari",
                            **summary_to_row_fields(scalar_summary(ari)),
                            "null_value": np.nan,
                        }
                    )
                    rows.append(
                        {
                            **base,
                            "metric_name": "label_vs_leiden_nmi",
                            **summary_to_row_fields(scalar_summary(nmi)),
                            "null_value": np.nan,
                        }
                    )


def compute_cell_batch_static_rows(
    *,
    mat: Any,
    obs_df: pd.DataFrame,
    space_label: str,
    dataset_id: str,
    model: str,
    seed: int,
    cell_type_col: str | None,
    batch_col: str | None,
    k_values: list[int],
    distance_metrics: list[str],
    leiden_resolution: float,
    reference_cache: dict[ClassifierCacheKey, ClassifierCacheValue],
    knn_cache: KnnIndexCacheLike | None,
    leiden_cache: LeidenCache | None,
    n_cells: int,
) -> list[dict[str, Any]]:
    """Metrics for a reference-only matrix (constant across interventions)."""
    rows: list[dict[str, Any]] = []
    logger.info("cell_type_and_batch_metrics: precomputing static space=%s", space_label)
    _append_metadata_rows(
        rows,
        mat=mat,
        obs_df=obs_df,
        space_label=space_label,
        cell_type_col=cell_type_col,
        batch_col=batch_col,
        k_values=k_values,
        distance_metrics=distance_metrics,
        leiden_resolution=leiden_resolution,
        seed=seed,
        dataset_id=dataset_id,
        model=model,
        intervention_id="__static__",
        intervention_name="__static__",
        n_cells=n_cells,
        reference_cache=reference_cache,
        knn_cache=knn_cache,
        leiden_cache=leiden_cache,
    )
    return rows


def stamp_cell_batch_rows(
    rows: list[dict[str, Any]],
    *,
    intervention_id: str,
    intervention_name: str,
) -> list[dict[str, Any]]:
    stamped: list[dict[str, Any]] = []
    for row in rows:
        stamped.append(
            {
                **row,
                "intervention_id": intervention_id,
                "intervention_name": intervention_name,
            }
        )
    return stamped


def compute_cell_type_and_batch_metrics(
    *,
    bundle: Any,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    seed: int,
    cell_type_col: str | None,
    batch_col: str | None,
    k_values: list[int],
    distance_metrics: list[str],
    leiden_resolution: float,
    reference_cache: dict[ClassifierCacheKey, ClassifierCacheValue],
    knn_cache: KnnIndexCacheLike | None = None,
    static_row_templates: list[list[dict[str, Any]]] | None = None,
    leiden_cache: LeidenCache | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    obs = bundle.obs
    n_cells = bundle.emb_ref.shape[0]

    if static_row_templates:
        for template in static_row_templates:
            rows.extend(
                stamp_cell_batch_rows(
                    template,
                    intervention_id=intervention_id,
                    intervention_name=intervention_name,
                )
            )

    matrix_items = [
        ("raw_manipulated", bundle.raw_man),
        ("embedding_manipulated", bundle.emb_man),
    ]
    logger.info(
        "cell_type_and_batch_metrics: intervention=%s n_cells=%d (2 manipulated spaces)",
        intervention_id,
        n_cells,
    )

    for space_label, mat in matrix_items:
        logger.info("cell_type_and_batch_metrics: space=%s", space_label)
        _append_metadata_rows(
            rows,
            mat=mat,
            obs_df=obs,
            space_label=space_label,
            cell_type_col=cell_type_col,
            batch_col=batch_col,
            k_values=k_values,
            distance_metrics=distance_metrics,
            leiden_resolution=leiden_resolution,
            seed=seed,
            dataset_id=dataset_id,
            model=model,
            intervention_id=intervention_id,
            intervention_name=intervention_name,
            n_cells=n_cells,
            reference_cache=reference_cache,
            knn_cache=knn_cache,
            leiden_cache=leiden_cache,
        )

    if not rows:
        logger.info("No cell_type/batch columns matched; skipping cell_type_and_batch_metrics")
    return pd.DataFrame(rows)
