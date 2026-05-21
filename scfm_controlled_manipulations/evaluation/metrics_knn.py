"""kNN neighborhood overlap (Phase 2) + diffusion KL/JS on kNN random walks (Phase 3).

Diffusion operator construction follows PHATE: per-cell adaptive bandwidth set
to the bandwidth_k-th neighbour distance, alpha-decay kernel (alpha=10 by
default; alpha=2 recovers the Gaussian), and additive symmetrization of the
two one-sided affinities.

Permutation nulls: row-permutation shuffles of ref/man correspondence with the
geometry of each space preserved. By default one shuffle is sampled and the
mean over cells is stored as the scalar null. Set ``n_null_permutations > 1``
to average across multiple shuffles.
"""

from __future__ import annotations

from collections.abc import Callable
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
    distribution_summary,
    make_metric_row,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# kNN construction and overlap
# ----------------------------------------------------------------------------


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


def knn_overlap_per_cell(
    ref_idx: np.ndarray,
    man_idx: np.ndarray,
    k: int,
) -> np.ndarray:
    """Per-cell kNN recall: fraction of ref neighbours also in man neighbours."""
    if ref_idx.shape != man_idx.shape:
        raise ValueError("ref_idx and man_idx must share shape")
    combined = np.concatenate([ref_idx, man_idx], axis=1)
    combined.sort(axis=1, kind="quicksort")
    duplicates = combined[:, 1:] == combined[:, :-1]
    intersection = duplicates.sum(axis=1, dtype=np.float64)
    return intersection / k


def _knn_null_seed(base_seed: int, space: str, metric: str, k: int) -> int:
    digest = hashlib.sha256(f"{base_seed}|knn_recall|{space}|{metric}|{k}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _diffusion_null_seed(base_seed: int, space: str, metric: str, k: int, t: int) -> int:
    digest = hashlib.sha256(f"{base_seed}|diffusion|{space}|{metric}|{k}|{t}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


# ----------------------------------------------------------------------------
# PHATE-style diffusion adjacency
# ----------------------------------------------------------------------------


def build_weighted_knn_adjacency_from_knn(
    nn_distances: np.ndarray,
    nn_idx: np.ndarray,
    *,
    alpha: float = 10.0,
    bandwidth_k: int | None = None,
) -> sp.csr_matrix:
    """PHATE-style symmetric affinity matrix from a kNN graph."""
    n_cells, k_graph = nn_idx.shape
    if bandwidth_k is None:
        bandwidth_k = k_graph
    if not (1 <= bandwidth_k <= k_graph):
        raise ValueError(f"bandwidth_k={bandwidth_k} must be in [1, {k_graph}]")

    sigma = nn_distances[:, bandwidth_k - 1].astype(np.float64)
    if not np.all(sigma > 0):
        positive = sigma[sigma > 0]
        fallback = float(np.median(positive)) if positive.size > 0 else 1.0
        sigma = np.where(sigma > 0, sigma, fallback)

    scaled = nn_distances.astype(np.float64) / sigma[:, None]
    one_sided = np.exp(-(scaled**alpha))

    row_idx = np.repeat(np.arange(n_cells), k_graph)
    col_idx = nn_idx.ravel()
    k_out = sp.csr_matrix(
        (one_sided.ravel(), (row_idx, col_idx)),
        shape=(n_cells, n_cells),
    )
    k_sym = (k_out + k_out.T) * 0.5
    k_sym.eliminate_zeros()
    return k_sym


def build_weighted_knn_adjacency(
    mat: Any,
    k: int,
    metric: str,
    *,
    alpha: float = 10.0,
    bandwidth_k: int | None = None,
) -> sp.csr_matrix:
    nn_distances, nn_idx = knn_neighbors(mat, k, metric)
    return build_weighted_knn_adjacency_from_knn(
        nn_distances, nn_idx, alpha=alpha, bandwidth_k=bandwidth_k
    )


def row_normalize_sparse(adj: sp.csr_matrix) -> sp.csr_matrix:
    row_sums = np.asarray(adj.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    inv = sp.diags(1.0 / row_sums)
    return inv @ adj


def transition_powers(adj: sp.csr_matrix, t_values: list[int]) -> dict[int, sp.csr_matrix]:
    """T^t for many t values, sharing multiplicative work via squaring when possible."""
    transition = row_normalize_sparse(adj)
    t_sorted = sorted({int(t) for t in t_values if int(t) >= 1})
    if not t_sorted:
        return {}

    out: dict[int, sp.csr_matrix] = {}
    all_powers_of_two = all((t & (t - 1)) == 0 for t in t_sorted)

    if all_powers_of_two:
        current = transition
        current_t = 1
        if 1 in t_sorted:
            out[1] = current
        max_t = t_sorted[-1]
        while current_t < max_t:
            current = current @ current
            current_t *= 2
            if current_t in t_sorted:
                out[current_t] = current
    else:
        logger.warning(
            "transition_powers: t_values %s contains non-power-of-two entries; "
            "falling back to sequential multiplication",
            t_sorted,
        )
        current = transition
        current_t = 1
        if 1 in t_sorted:
            out[1] = current
        for target in t_sorted:
            while current_t < target:
                current = current @ transition
                current_t += 1
            if target not in out:
                out[target] = current

    return out


# ----------------------------------------------------------------------------
# KL / JS on rows
# ----------------------------------------------------------------------------


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


def _kl_js_block(
    p_csr: sp.csr_matrix,
    q_csr: sp.csr_matrix,
    start: int,
    end: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    col_idx = _chunk_column_indices(p_csr, q_csr, start, end)
    n_chunk = end - start
    if col_idx.size == 0:
        return np.zeros(n_chunk, dtype=np.float64), np.zeros(n_chunk, dtype=np.float64)

    p_block = p_csr[start:end][:, col_idx].toarray().astype(np.float64) + eps
    q_block = q_csr[start:end][:, col_idx].toarray().astype(np.float64) + eps
    p_block /= p_block.sum(axis=1, keepdims=True)
    q_block /= q_block.sum(axis=1, keepdims=True)
    m_block = 0.5 * (p_block + q_block)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.log(p_block)
        log_q = np.log(q_block)
        log_m = np.log(m_block)
        kl_pq = np.sum(p_block * (log_p - log_q), axis=1)
        kl_qp = np.sum(q_block * (log_q - log_p), axis=1)
        js = 0.5 * np.sum(p_block * (log_p - log_m), axis=1) + 0.5 * np.sum(
            q_block * (log_q - log_m), axis=1
        )

    return 0.5 * (kl_pq + kl_qp), js


def sym_kl_js_per_cell(
    p_mat: sp.csr_matrix,
    q_mat: sp.csr_matrix,
    *,
    eps: float = 1e-12,
    row_chunk: int = 512,
    dense_max_cells: int = 8000,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric KL and JS for each row pair."""
    p_csr = p_mat.tocsr()
    q_csr = q_mat.tocsr()
    n_cells = p_csr.shape[0]

    if n_cells <= dense_max_cells and (p_csr.nnz + q_csr.nnz) <= n_cells * max(256, row_chunk):
        return _kl_js_block(p_csr, q_csr, 0, n_cells, eps)

    sym_kl = np.zeros(n_cells, dtype=np.float64)
    js = np.zeros(n_cells, dtype=np.float64)
    for start in range(0, n_cells, row_chunk):
        end = min(start + row_chunk, n_cells)
        chunk_kl, chunk_js = _kl_js_block(p_csr, q_csr, start, end, eps)
        sym_kl[start:end] = chunk_kl
        js[start:end] = chunk_js
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


def _knn_cache_path(
    cache_dir: Path,
    *,
    dataset_id: str,
    model: str,
    space: str,
    metric: str,
    k: int,
    n_cells: int,
    side: str,
) -> Path:
    payload = f"{dataset_id}|{model}|{space}|{metric}|{k}|{n_cells}|{side}"
    h = hashlib.sha256(payload.encode()).hexdigest()[:20]
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"knn_{h}.pkl"


def _load_or_compute_transition_powers(
    cache_dir: Path,
    *,
    dataset_id: str,
    model: str,
    space: str,
    metric: str,
    k: int,
    n_cells: int,
    side: str,
    adj_builder: Callable[[], sp.csr_matrix],
    t_values: list[int],
) -> dict[int, sp.csr_matrix]:
    """Return T^t for each requested t, reading per-t pickles where present."""
    paths = {
        t: _cache_path(
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
        for t in t_values
    }
    missing = [t for t in t_values if not paths[t].exists()]
    if missing:
        logger.info(
            "transitions: side=%s k=%d computing %d/%d missing via squaring",
            side,
            k,
            len(missing),
            len(t_values),
        )
        adj = adj_builder()
        precomputed = transition_powers(adj, missing)
    else:
        precomputed = {}

    out: dict[int, sp.csr_matrix] = {}
    for t in t_values:
        label = (
            f"diffusion space={space} metric={metric} k={k} t={t} side={side} ({n_cells} cells)"
        )

        def builder(t=t):
            return precomputed[t]

        out[t] = load_or_build_pickle(paths[t], builder, label=label)
    return out


def _compute_knn_recall_rows(
    *,
    knn_at_max: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    knn_spaces: tuple[str, ...],
    distance_metrics: list[str],
    k_sorted: list[int],
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    seed: int,
    n_cells: int,
    n_null: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for space in knn_spaces:
        for metric in distance_metrics:
            _, ref_idx_max, _, man_idx_max = knn_at_max[(space, metric)]
            for k in k_sorted:
                ref_idx = ref_idx_max[:, :k]
                man_idx = man_idx_max[:, :k]
                recall = knn_overlap_per_cell(ref_idx, man_idx, k)

                null_rng = np.random.default_rng(_knn_null_seed(seed, space, metric, k))
                null_recall_sum = 0.0
                for _ in range(n_null):
                    perm = null_rng.permutation(n_cells)
                    null_recall_cells = knn_overlap_per_cell(ref_idx, man_idx[perm], k)
                    null_recall_sum += float(np.mean(null_recall_cells))
                null_recall = null_recall_sum / n_null

                rows.append(
                    make_metric_row(
                        dataset_id=dataset_id,
                        model=model,
                        intervention_id=intervention_id,
                        intervention_name=intervention_name,
                        metric_category="knn_metrics",
                        metric_name="knn_recall",
                        space=space,
                        summary=distribution_summary(recall),
                        null_value=null_recall,
                        n_cells=n_cells,
                        seed=seed,
                        extra={"distance_metric": metric, "k": k, "diffusion_t": np.nan},
                    )
                )
    return rows


def _compute_diffusion_rows(
    *,
    knn_at_max: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    diffusion_spaces: tuple[str, ...],
    distance_metrics: list[str],
    k_sorted: list[int],
    t_list: list[int],
    cache_dir: Path,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    seed: int,
    n_cells: int,
    n_null: int,
    alpha: float,
    bandwidth_k: int | None,
    row_chunk: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for space in diffusion_spaces:
        for metric in distance_metrics:
            ref_dist_max, ref_idx_max, man_dist_max, man_idx_max = knn_at_max[(space, metric)]
            for k in k_sorted:

                def _build_ref_adj() -> sp.csr_matrix:
                    return build_weighted_knn_adjacency_from_knn(
                        ref_dist_max[:, :k],
                        ref_idx_max[:, :k],
                        alpha=alpha,
                        bandwidth_k=bandwidth_k,
                    )

                def _build_man_adj() -> sp.csr_matrix:
                    return build_weighted_knn_adjacency_from_knn(
                        man_dist_max[:, :k],
                        man_idx_max[:, :k],
                        alpha=alpha,
                        bandwidth_k=bandwidth_k,
                    )

                ref_powers = _load_or_compute_transition_powers(
                    cache_dir,
                    dataset_id=dataset_id,
                    model=model,
                    space=space,
                    metric=metric,
                    k=k,
                    n_cells=n_cells,
                    side="ref",
                    adj_builder=_build_ref_adj,
                    t_values=t_list,
                )
                man_powers = _load_or_compute_transition_powers(
                    cache_dir,
                    dataset_id=dataset_id,
                    model=model,
                    space=space,
                    metric=metric,
                    k=k,
                    n_cells=n_cells,
                    side=f"man_{intervention_id}",
                    adj_builder=_build_man_adj,
                    t_values=t_list,
                )

                for t in t_list:
                    p_t = ref_powers[t]
                    q_t = man_powers[t]
                    sym_kl, js = sym_kl_js_per_cell(p_t, q_t, row_chunk=row_chunk)

                    null_rng = np.random.default_rng(
                        _diffusion_null_seed(seed, space, metric, k, t)
                    )
                    null_sym_sum = 0.0
                    null_js_sum = 0.0
                    for _ in range(n_null):
                        perm = null_rng.permutation(n_cells)
                        null_sym_mean, null_js_mean = _diffusion_sym_kl_js_means(
                            p_t, q_t[perm], row_chunk=row_chunk
                        )
                        null_sym_sum += null_sym_mean
                        null_js_sum += null_js_mean
                    null_sym = null_sym_sum / n_null
                    null_js_val = null_js_sum / n_null

                    for metric_name, summary, null_val in (
                        ("diffusion_sym_kl", distribution_summary(sym_kl), null_sym),
                        ("diffusion_js", distribution_summary(js), null_js_val),
                    ):
                        rows.append(
                            make_metric_row(
                                dataset_id=dataset_id,
                                model=model,
                                intervention_id=intervention_id,
                                intervention_name=intervention_name,
                                metric_category="knn_metrics",
                                metric_name=metric_name,
                                space=space,
                                summary=summary,
                                null_value=null_val,
                                n_cells=n_cells,
                                seed=seed,
                                extra={
                                    "distance_metric": metric,
                                    "k": k,
                                    "diffusion_t": t,
                                },
                            )
                        )
    return rows


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


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
    alpha: float = 10.0,
    bandwidth_k: int | None = None,
    n_null_permutations: int = 1,
) -> pd.DataFrame:
    """Compute kNN overlap and diffusion KL/JS metrics for one intervention."""
    n_cells = bundle.emb_ref.shape[0]
    k_sorted = sorted(int(k) for k in k_values)
    k_max = k_sorted[-1]
    knn_spaces = ("raw", "embedding")
    diffusion_spaces = ("embedding",)
    n_null = max(1, int(n_null_permutations))
    t_list = list(diffusion_t_values)

    n_diffusion_jobs = (
        len(diffusion_spaces) * len(distance_metrics) * len(k_sorted) * len(t_list) * 2
    )
    logger.info(
        "knn_metrics: intervention=%s n_cells=%d k_max=%d "
        "(%d diffusion matrices, alpha=%.2f, bandwidth_k=%s, null_perms=%d)",
        intervention_id,
        n_cells,
        k_max,
        n_diffusion_jobs,
        alpha,
        bandwidth_k,
        n_null,
    )

    def mats_for(space: str) -> tuple[Any, Any]:
        if space == "raw":
            return bundle.raw_ref, bundle.raw_man
        return bundle.emb_ref, bundle.emb_man

    def _neighbors(mat: Any, metric: str) -> tuple[np.ndarray, np.ndarray]:
        if knn_cache is not None:
            return knn_cache.neighbors(mat, k_max, metric)
        return knn_neighbors(mat, k_max, metric)

    def _neighbors_with_disk_cache(
        mat: Any,
        *,
        metric: str,
        space: str,
        side: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        path = _knn_cache_path(
            cache_dir,
            dataset_id=dataset_id,
            model=model,
            space=space,
            metric=metric,
            k=k_max,
            n_cells=n_cells,
            side=side,
        )
        label = f"knn side={side} space={space} metric={metric} k={k_max} ({n_cells} cells)"
        return load_or_build_pickle(path, lambda: _neighbors(mat, metric), label=label)

    spaces_needed = tuple(dict.fromkeys((*knn_spaces, *diffusion_spaces)))
    knn_at_max: dict[
        tuple[str, str],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}

    for space in spaces_needed:
        for metric in distance_metrics:
            ref_mat, man_mat = mats_for(space)
            logger.info(
                "knn_metrics: space=%s metric=%s — kNN graph at k_max=%d",
                space,
                metric,
                k_max,
            )
            ref_dist_max, ref_idx_max = _neighbors_with_disk_cache(
                ref_mat, metric=metric, space=space, side="ref"
            )
            man_dist_max, man_idx_max = _neighbors_with_disk_cache(
                man_mat,
                metric=metric,
                space=space,
                side=f"man_{intervention_id}",
            )
            knn_at_max[(space, metric)] = (
                ref_dist_max,
                ref_idx_max,
                man_dist_max,
                man_idx_max,
            )

    rows = _compute_knn_recall_rows(
        knn_at_max=knn_at_max,
        knn_spaces=knn_spaces,
        distance_metrics=distance_metrics,
        k_sorted=k_sorted,
        dataset_id=dataset_id,
        model=model,
        intervention_id=intervention_id,
        intervention_name=intervention_name,
        seed=seed,
        n_cells=n_cells,
        n_null=n_null,
    )
    rows.extend(
        _compute_diffusion_rows(
            knn_at_max=knn_at_max,
            diffusion_spaces=diffusion_spaces,
            distance_metrics=distance_metrics,
            k_sorted=k_sorted,
            t_list=t_list,
            cache_dir=cache_dir,
            dataset_id=dataset_id,
            model=model,
            intervention_id=intervention_id,
            intervention_name=intervention_name,
            seed=seed,
            n_cells=n_cells,
            n_null=n_null,
            alpha=alpha,
            bandwidth_k=bandwidth_k,
            row_chunk=row_chunk,
        )
    )

    return pd.DataFrame(rows)
