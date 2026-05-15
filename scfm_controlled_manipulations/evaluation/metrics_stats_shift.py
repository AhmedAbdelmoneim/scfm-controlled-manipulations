"""Light global summaries (embedding_stats) and paired geometric shift (embedding_shift).

embedding_shift (no diffusion here — see ``metrics_knn``):

- ``centroid_l2``: ||mean(man) - mean(ref)||_2 in that space (column means).
- ``mean_paired_l2`` / ``median_paired_l2`` / ``std_paired_l2``: distribution of per-cell
  ||man_i - ref_i||_2.
- ``coherence_mean_dot``: mean_i cos(delta_i, u) where delta_i = man_i - ref_i and
  u = (mean(man) - mean(ref)) / ||...|| (zero vector yields NaN coherence).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scfm_controlled_manipulations.evaluation.metrics_common import (
    distribution_summary,
    scalar_summary,
)


def _row_norms_csr(x: sp.csr_matrix) -> np.ndarray:
    return np.sqrt(x.multiply(x).sum(axis=1)).A1


def _col_means_csr(x: sp.csr_matrix) -> np.ndarray:
    return np.asarray(x.mean(axis=0)).ravel()


def _mean_col_variance_csr(x: sp.csr_matrix) -> float:
    """Mean per-column variance: E_j Var_i(X_ij) for sparse X (cells x genes)."""
    col_means = _col_means_csr(x)
    col_means_sq = np.asarray(x.power(2).mean(axis=0)).ravel()
    col_vars = col_means_sq - col_means**2
    return float(np.mean(col_vars))


def _centroid_l2_dense(ref: np.ndarray, man: np.ndarray) -> float:
    return float(np.linalg.norm(np.mean(man, axis=0) - np.mean(ref, axis=0)))


def _paired_shift_stats_dense(ref: np.ndarray, man: np.ndarray) -> tuple[float, float, float]:
    d = np.linalg.norm(man - ref, axis=1)
    return float(np.mean(d)), float(np.median(d)), float(np.std(d))


def _paired_shift_stats_sparse(
    ref: sp.csr_matrix, man: sp.csr_matrix
) -> tuple[float, float, float]:
    diff = man - ref
    d = _row_norms_csr(diff)
    return float(np.mean(d)), float(np.median(d)), float(np.std(d))


def _shift_coherence_cosines_dense(ref: np.ndarray, man: np.ndarray) -> np.ndarray:
    delta = man - ref
    u = np.mean(man, axis=0) - np.mean(ref, axis=0)
    norm_u = np.linalg.norm(u)
    if norm_u <= 0:
        return np.array([], dtype=np.float64)
    u = u / norm_u
    dots = delta @ u
    norms = np.linalg.norm(delta, axis=1)
    valid = norms > 1e-12
    if not np.any(valid):
        return np.array([], dtype=np.float64)
    return dots[valid] / norms[valid]


def _shift_coherence_cosines_sparse(ref: sp.csr_matrix, man: sp.csr_matrix) -> np.ndarray:
    delta = (man - ref).tocsr()
    u = np.asarray(man.mean(axis=0) - ref.mean(axis=0)).ravel()
    norm_u = np.linalg.norm(u)
    if norm_u <= 1e-12:
        return np.array([], dtype=np.float64)
    u = u / norm_u
    dots = np.asarray(delta @ u).ravel()
    norms = _row_norms_csr(delta)
    valid = norms > 1e-12
    if not np.any(valid):
        return np.array([], dtype=np.float64)
    return dots[valid] / norms[valid]


def _row_metric(
    *,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    category: str,
    metric_name: str,
    space: str,
    value_mean: float,
    value_median: float,
    value_std: float,
    n_cells: int,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset_id": dataset_id,
        "model": model,
        "intervention_id": intervention_id,
        "intervention_name": intervention_name,
        "metric_category": category,
        "metric_name": metric_name,
        "space": space,
        "value_mean": value_mean,
        "value_median": value_median,
        "value_std": value_std,
        "null_value": np.nan,
        "n_cells": n_cells,
        "seed": seed,
    }
    if extra:
        row.update(extra)
    return row


def compute_embedding_stats(
    *,
    bundle: Any,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = bundle.emb_ref.shape[0]

    for space, ref, man in (
        ("raw", bundle.raw_ref, bundle.raw_man),
        ("embedding", bundle.emb_ref, bundle.emb_man),
    ):
        if space == "raw":
            ref_norms = _row_norms_csr(ref)
            man_norms = _row_norms_csr(man)
            ref_var = _mean_col_variance_csr(ref)
            man_var = _mean_col_variance_csr(man)
        else:
            ref_norms = np.linalg.norm(ref, axis=1)
            man_norms = np.linalg.norm(man, axis=1)
            ref_var = float(np.mean(np.var(ref, axis=0)))
            man_var = float(np.mean(np.var(man, axis=0)))

        ref_norm_m, ref_norm_med, ref_norm_s = distribution_summary(ref_norms)
        man_norm_m, man_norm_med, man_norm_s = distribution_summary(man_norms)
        ref_var_m, ref_var_med, ref_var_s = scalar_summary(ref_var)
        man_var_m, man_var_med, man_var_s = scalar_summary(man_var)

        for metric_name, vm, vmed, vs in (
            ("mean_row_l2_norm_ref", ref_norm_m, ref_norm_med, ref_norm_s),
            ("mean_row_l2_norm_man", man_norm_m, man_norm_med, man_norm_s),
            ("mean_col_variance_ref", ref_var_m, ref_var_med, ref_var_s),
            ("mean_col_variance_man", man_var_m, man_var_med, man_var_s),
        ):
            rows.append(
                _row_metric(
                    dataset_id=dataset_id,
                    model=model,
                    intervention_id=intervention_id,
                    intervention_name=intervention_name,
                    category="embedding_stats",
                    metric_name=metric_name,
                    space=space,
                    value_mean=vm,
                    value_median=vmed,
                    value_std=vs,
                    n_cells=n,
                    seed=seed,
                )
            )

    return pd.DataFrame(rows)


def compute_embedding_shift(
    *,
    bundle: Any,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = bundle.emb_ref.shape[0]

    for space, ref, man in (
        ("raw", bundle.raw_ref, bundle.raw_man),
        ("embedding", bundle.emb_ref, bundle.emb_man),
    ):
        if space == "raw":
            centroid = float(np.linalg.norm(_col_means_csr(man) - _col_means_csr(ref)))
            m_pl2, med_pl2, s_pl2 = _paired_shift_stats_sparse(ref, man)
            coh_m, coh_med, coh_s = distribution_summary(_shift_coherence_cosines_sparse(ref, man))
        else:
            centroid = _centroid_l2_dense(ref, man)
            m_pl2, med_pl2, s_pl2 = _paired_shift_stats_dense(ref, man)
            coh_m, coh_med, coh_s = distribution_summary(_shift_coherence_cosines_dense(ref, man))

        cent_m, cent_med, cent_s = scalar_summary(centroid)
        for metric_name, vm, vmed, vs in (
            ("centroid_l2_shift", cent_m, cent_med, cent_s),
            ("paired_cell_l2_norm", m_pl2, med_pl2, s_pl2),
            ("shift_coherence_mean_cosine", coh_m, coh_med, coh_s),
        ):
            rows.append(
                _row_metric(
                    dataset_id=dataset_id,
                    model=model,
                    intervention_id=intervention_id,
                    intervention_name=intervention_name,
                    category="embedding_shift",
                    metric_name=metric_name,
                    space=space,
                    value_mean=vm,
                    value_median=vmed,
                    value_std=vs,
                    n_cells=n,
                    seed=seed,
                )
            )

    return pd.DataFrame(rows)
