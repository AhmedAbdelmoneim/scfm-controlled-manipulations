"""kNN neighborhood overlap (Phase 2) + diffusion KL/JS on kNN random walks (Phase 3).

Permutation nulls: single shuffle of ref/man row correspondence (broken pairing,
preserved manipulated geometry).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from scfm_controlled_manipulations.evaluation.disk_cache import load_or_build_pickle
from scfm_controlled_manipulations.evaluation.metrics_common import (
    DistributionSummary,
    distribution_summary,
    summary_to_row_fields,
)

logger = logging.getLogger(__name__)


def knn_neighbors(
    mat: Any,
    k: int,
    metric: str,
    *,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """kNN distances and indices (excluding self), shape ``(n_cells, k)``."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=n_jobs).fit(mat)
    dist, idx = nn.kneighbors(mat)
    return dist[:, 1:], idx[:, 1:]


def knn_indices(mat: Any, k: int, metric: str) -> tuple[np.ndarray, np.ndarray]:
    return knn_neighbors(mat, k, metric)


def knn_overlap_per_cell(
    ref_idx: np.ndarray,
    man_idx: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell recall and Jaccard from neighbor index arrays, shape ``(n_cells, k)``."""
    in_man = (ref_idx[:, :, None] == man_idx[:, None, :]).any(axis=2)
    intersection = in_man.sum(axis=1, dtype=np.float64)
    recall = intersection / k
    union_size = np.maximum(2 * k - intersection, 1.0)
    jaccard = intersection / union_size
    return recall, jaccard


def _knn_null_seed(base_seed: int, space: str, metric: str, k: int) -> int:
    digest = hashlib.sha256(f"{base_seed}|knn_recall|{space}|{metric}|{k}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _diffusion_null_seed(base_seed: int, space: str, metric: str, k: int, t: int) -> int:
    digest = hashlib.sha256(f"{base_seed}|diffusion|{space}|{metric}|{k}|{t}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def empirical_knn_recall_null(
    ref_mat: Any,
    man_mat: Any,
    *,
    k: int,
    metric: str,
    seed: int,
) -> float:
    """Mean recall under a single shuffle of ref/man row correspondence."""
    n_cells = ref_mat.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_cells)
    _, ref_idx = knn_indices(ref_mat, k=k, metric=metric)
    _, man_idx = knn_indices(man_mat, k=k, metric=metric)
    null_recall, _ = knn_overlap_per_cell(ref_idx, man_idx[perm], k)
    return float(np.mean(null_recall))


def build_weighted_knn_adjacency_from_knn(
    nn_distances: np.ndarray,
    nn_idx: np.ndarray,
) -> sp.csr_matrix:
    k = nn_idx.shape[1]
    n_cells = nn_idx.shape[0]
    row_idx = np.repeat(np.arange(n_cells), k)
    col_idx = nn_idx.ravel()
    sigma = float(np.median(nn_distances[:, -1]))
    if sigma <= 0:
        positive = nn_distances[nn_distances > 0]
        sigma = float(np.median(positive)) if positive.size > 0 else 1.0
    edge_weights = np.exp(-(nn_distances.ravel() ** 2) / (2 * sigma**2))
    adj = sp.csr_matrix((edge_weights, (row_idx, col_idx)), shape=(n_cells, n_cells))
    adj = adj.maximum(adj.T)
    adj.eliminate_zeros()
    return adj


def build_weighted_knn_adjacency(mat: Any, k: int, metric: str) -> sp.csr_matrix:
    nn_distances, nn_idx = knn_neighbors(mat, k, metric)
    return build_weighted_knn_adjacency_from_knn(nn_distances, nn_idx)


def row_normalize_sparse(adj: sp.csr_matrix) -> sp.csr_matrix:
    row_sums = np.asarray(adj.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    inv = sp.diags(1.0 / row_sums)
    return inv @ adj


def sparse_transition_power(adj: sp.csr_matrix, t: int) -> sp.csr_matrix:
    transition = row_normalize_sparse(adj)
    if t < 1:
        return transition
    out = transition.copy()
    for _ in range(1, t):
        out = out @ transition
    return out


def _chunk_column_indices(
    p_csr: sp.csr_matrix, q_csr: sp.csr_matrix, start: int, end: int
) -> np.ndarray:
    p_ind = p_csr.indices[p_csr.indptr[start] : p_csr.indptr[end]]
    q_ind = q_csr.indices[q_csr.indptr[start] : q_csr.indptr[end]]
    if p_ind.size == 0 and q_ind.size == 0:
        return np.empty(0, dtype=np.intp)
    if p_ind.size == 0:
        return np.unique(q_ind)
    if q_ind.size == 0:
        return np.unique(p_ind)
    return np.unique(np.concatenate([p_ind, q_ind]))


def sym_kl_js_per_cell(
    p_mat: sp.csr_matrix,
    q_mat: sp.csr_matrix,
    *,
    eps: float = 1e-12,
    row_chunk: int = 512,
    dense_max_cells: int = 8000,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric KL and JS for each row pair (vectorized)."""
    p_csr = p_mat.tocsr()
    q_csr = q_mat.tocsr()
    n_cells = p_csr.shape[0]

    if n_cells <= dense_max_cells and (p_csr.nnz + q_csr.nnz) <= n_cells * max(256, row_chunk):
        p_dense = np.asarray(p_csr.toarray(), dtype=np.float64) + eps
        q_dense = np.asarray(q_csr.toarray(), dtype=np.float64) + eps
        p_dense /= p_dense.sum(axis=1, keepdims=True)
        q_dense /= q_dense.sum(axis=1, keepdims=True)
        m_dense = 0.5 * (p_dense + q_dense)
        with np.errstate(divide="ignore", invalid="ignore"):
            kl_pq = np.sum(p_dense * np.log(p_dense / q_dense), axis=1)
            kl_qp = np.sum(q_dense * np.log(q_dense / p_dense), axis=1)
            js = 0.5 * np.sum(p_dense * np.log(p_dense / m_dense), axis=1) + 0.5 * np.sum(
                q_dense * np.log(q_dense / m_dense), axis=1
            )
        return 0.5 * (kl_pq + kl_qp), js

    sym_kl = np.zeros(n_cells, dtype=np.float64)
    js = np.zeros(n_cells, dtype=np.float64)
    for start in range(0, n_cells, row_chunk):
        end = min(start + row_chunk, n_cells)
        p_slice = p_csr[start:end]
        q_slice = q_csr[start:end]
        if p_slice.nnz == 0 and q_slice.nnz == 0:
            continue
        col_idx = _chunk_column_indices(p_csr, q_csr, start, end)
        p_block = p_slice[:, col_idx].toarray() + eps
        q_block = q_slice[:, col_idx].toarray() + eps
        p_block /= p_block.sum(axis=1, keepdims=True)
        q_block /= q_block.sum(axis=1, keepdims=True)
        m_block = 0.5 * (p_block + q_block)
        with np.errstate(divide="ignore", invalid="ignore"):
            kl_pq = np.sum(p_block * np.log(p_block / q_block), axis=1)
            kl_qp = np.sum(q_block * np.log(q_block / p_block), axis=1)
            js_block = 0.5 * np.sum(p_block * np.log(p_block / m_block), axis=1) + 0.5 * np.sum(
                q_block * np.log(q_block / m_block), axis=1
            )
        sym_kl[start:end] = 0.5 * (kl_pq + kl_qp)
        js[start:end] = js_block

    return sym_kl, js


def _diffusion_sym_kl_js_means(
    p_mat: sp.csr_matrix,
    q_mat: sp.csr_matrix,
    *,
    row_chunk: int,
) -> tuple[float, float]:
    sym_kl, js = sym_kl_js_per_cell(p_mat, q_mat, row_chunk=row_chunk)
    return float(np.mean(sym_kl)), float(np.mean(js))


def _cache_path(
    cache_dir: Path,
    *,
    dataset_id: str,
    model: str,
    space: str,
    metric: str,
    k: int,
    t: int,
    n_cells: int,
    side: str,
) -> Path:
    payload = f"{dataset_id}|{model}|{space}|{metric}|{k}|{t}|{n_cells}|{side}"
    h = hashlib.sha256(payload.encode()).hexdigest()[:20]
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"diffusion_{h}.pkl"


def _load_transition_t(
    cache_dir: Path,
    *,
    dataset_id: str,
    model: str,
    space: str,
    metric: str,
    k: int,
    t: int,
    n_cells: int,
    side: str,
    builder: Any,
) -> sp.csr_matrix:
    path = _cache_path(
        cache_dir,
        dataset_id=dataset_id,
        model=model,
        space=space,
        metric=metric,
        k=k,
        t=t,
        n_cells=n_cells,
        side=side,
    )
    label = f"diffusion space={space} metric={metric} k={k} t={t} side={side} ({n_cells} cells)"
    return load_or_build_pickle(path, builder, label=label)


def _row_metric_row(
    *,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    category: str,
    metric_name: str,
    space: str,
    summary: DistributionSummary,
    null_value: float,
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
        **summary_to_row_fields(summary),
        "null_value": null_value,
        "n_cells": n_cells,
        "seed": seed,
    }
    if extra:
        row.update(extra)
    return row


def compute_knn_metrics(
    *,
    bundle: Any,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    seed: int,
    distance_metrics: list[str],
    k_values: list[int],
    diffusion_t_values: list[int],
    row_chunk: int = 512,
    cache_dir: Path,
    knn_cache: Any | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_cells = bundle.emb_ref.shape[0]
    k_sorted = sorted(int(k) for k in k_values)
    k_max = k_sorted[-1]
    knn_spaces = ("raw", "embedding")
    diffusion_spaces = ("embedding",)
    n_diffusion_jobs = (
        len(diffusion_spaces) * len(distance_metrics) * len(k_sorted) * len(diffusion_t_values) * 2
    )

    logger.info(
        "knn_metrics: intervention=%s n_cells=%d k_max=%d (%d diffusion matrices to load/build)",
        intervention_id,
        n_cells,
        k_max,
        n_diffusion_jobs,
    )

    def mats_for(space: str) -> tuple[Any, Any]:
        if space == "raw":
            return bundle.raw_ref, bundle.raw_man
        return bundle.emb_ref, bundle.emb_man

    def _neighbors(mat: Any, metric: str) -> tuple[np.ndarray, np.ndarray]:
        if knn_cache is not None:
            return knn_cache.neighbors(mat, k_max, metric)
        return knn_neighbors(mat, k_max, metric)

    spaces_needed = tuple(dict.fromkeys((*knn_spaces, *diffusion_spaces)))
    knn_at_max: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def _build_space_metric(space: str, metric: str) -> tuple[str, str, tuple]:
        ref_mat, man_mat = mats_for(space)
        logger.info(
            "knn_metrics: space=%s metric=%s — kNN graph at k_max=%d",
            space,
            metric,
            k_max,
        )
        ref_dist_max, ref_idx_max = _neighbors(ref_mat, metric)
        man_dist_max, man_idx_max = _neighbors(man_mat, metric)
        return space, metric, (ref_dist_max, ref_idx_max, man_dist_max, man_idx_max)

    for space, metric in ((s, m) for s in spaces_needed for m in distance_metrics):
        s, m, payload = _build_space_metric(space, metric)
        knn_at_max[(s, m)] = payload

    for space in knn_spaces:
        for metric in distance_metrics:
            _, ref_idx_max, _, man_idx_max = knn_at_max[(space, metric)]
            for k in k_sorted:
                ref_idx = ref_idx_max[:, :k]
                man_idx = man_idx_max[:, :k]
                recall, jaccard = knn_overlap_per_cell(ref_idx, man_idx, k)
                null_rng = np.random.default_rng(_knn_null_seed(seed, space, metric, k))
                perm = null_rng.permutation(n_cells)
                null_per_cell, _ = knn_overlap_per_cell(ref_idx, man_idx[perm], k)
                null_recall = float(np.mean(null_per_cell))
                rows.append(
                    _row_metric_row(
                        dataset_id=dataset_id,
                        model=model,
                        intervention_id=intervention_id,
                        intervention_name=intervention_name,
                        category="knn_metrics",
                        metric_name="knn_recall",
                        space=space,
                        summary=distribution_summary(recall),
                        null_value=null_recall,
                        n_cells=n_cells,
                        seed=seed,
                        extra={"distance_metric": metric, "k": k, "diffusion_t": np.nan},
                    )
                )
                rows.append(
                    _row_metric_row(
                        dataset_id=dataset_id,
                        model=model,
                        intervention_id=intervention_id,
                        intervention_name=intervention_name,
                        category="knn_metrics",
                        metric_name="knn_jaccard",
                        space=space,
                        summary=distribution_summary(jaccard),
                        null_value=np.nan,
                        n_cells=n_cells,
                        seed=seed,
                        extra={"distance_metric": metric, "k": k, "diffusion_t": np.nan},
                    )
                )

    for space in diffusion_spaces:
        for metric in distance_metrics:
            ref_dist_max, ref_idx_max, man_dist_max, man_idx_max = knn_at_max[(space, metric)]
            for k in k_sorted:
                ref_adj = build_weighted_knn_adjacency_from_knn(
                    ref_dist_max[:, :k], ref_idx_max[:, :k]
                )
                man_adj = build_weighted_knn_adjacency_from_knn(
                    man_dist_max[:, :k], man_idx_max[:, :k]
                )
                for t in diffusion_t_values:

                    def build_ref(
                        ref_adj=ref_adj,
                        t=t,
                    ) -> sp.csr_matrix:
                        return sparse_transition_power(ref_adj, t)

                    def build_man(
                        man_adj=man_adj,
                        t=t,
                    ) -> sp.csr_matrix:
                        return sparse_transition_power(man_adj, t)

                    p_t = _load_transition_t(
                        cache_dir,
                        dataset_id=dataset_id,
                        model=model,
                        space=space,
                        metric=metric,
                        k=k,
                        t=t,
                        n_cells=n_cells,
                        side="ref",
                        builder=build_ref,
                    )
                    q_t = _load_transition_t(
                        cache_dir,
                        dataset_id=dataset_id,
                        model=model,
                        space=space,
                        metric=metric,
                        k=k,
                        t=t,
                        n_cells=n_cells,
                        side=f"man_{intervention_id}",
                        builder=build_man,
                    )

                    sym_kl, js = sym_kl_js_per_cell(p_t, q_t, row_chunk=row_chunk)

                    null_rng = np.random.default_rng(
                        _diffusion_null_seed(seed, space, metric, k, t)
                    )
                    perm = null_rng.permutation(n_cells)
                    null_sym_mean, null_js_mean = _diffusion_sym_kl_js_means(
                        p_t, q_t[perm], row_chunk=row_chunk
                    )

                    rows.append(
                        _row_metric_row(
                            dataset_id=dataset_id,
                            model=model,
                            intervention_id=intervention_id,
                            intervention_name=intervention_name,
                            category="knn_metrics",
                            metric_name="diffusion_sym_kl",
                            space=space,
                            summary=distribution_summary(sym_kl),
                            null_value=float(null_sym_mean),
                            n_cells=n_cells,
                            seed=seed,
                            extra={"distance_metric": metric, "k": k, "diffusion_t": t},
                        )
                    )
                    rows.append(
                        _row_metric_row(
                            dataset_id=dataset_id,
                            model=model,
                            intervention_id=intervention_id,
                            intervention_name=intervention_name,
                            category="knn_metrics",
                            metric_name="diffusion_js",
                            space=space,
                            summary=distribution_summary(js),
                            null_value=float(null_js_mean),
                            n_cells=n_cells,
                            seed=seed,
                            extra={"distance_metric": metric, "k": k, "diffusion_t": t},
                        )
                    )

    return pd.DataFrame(rows)
