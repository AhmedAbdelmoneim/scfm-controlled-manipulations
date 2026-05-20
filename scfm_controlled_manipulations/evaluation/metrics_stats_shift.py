"""Light global summaries (embedding_stats) and paired geometric shift (embedding_shift)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scfm_controlled_manipulations.evaluation.metrics_common import (
    DistributionSummary,
    distribution_summary,
    summary_to_row_fields,
)


def _row_norms_csr(x: sp.csr_matrix) -> np.ndarray:
    return np.sqrt(x.multiply(x).sum(axis=1)).A1


def _col_variances_csr(x: sp.csr_matrix) -> np.ndarray:
    col_means = np.asarray(x.mean(axis=0)).ravel()
    col_means_sq = np.asarray(x.power(2).mean(axis=0)).ravel()
    return col_means_sq - col_means**2


def _col_variances_dense(x: np.ndarray) -> np.ndarray:
    return np.var(x, axis=0)


def _sample_pairwise_cell_indices(n_cells: int, n_sub: int, *, seed: int) -> np.ndarray:
    n_sub = min(int(n_sub), int(n_cells))
    if n_sub <= 0:
        return np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_cells, size=n_sub, replace=False))


def _cap_pair_indices(
    i_idx: np.ndarray, j_idx: np.ndarray, max_pairs: int | None, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    n_pairs = i_idx.size
    if max_pairs is None or n_pairs <= max_pairs:
        return i_idx, j_idx
    rng = np.random.default_rng(seed)
    pick = rng.choice(n_pairs, size=int(max_pairs), replace=False)
    return i_idx[pick], j_idx[pick]


def _pairwise_l2_dense(
    sub: np.ndarray, *, max_pairs: int | None = None, seed: int = 0
) -> np.ndarray:
    n = sub.shape[0]
    if n < 2:
        return np.array([], dtype=np.float64)
    i_idx, j_idx = np.triu_indices(n, k=1)
    i_idx, j_idx = _cap_pair_indices(i_idx, j_idx, max_pairs, seed=seed)
    return np.linalg.norm(sub[i_idx] - sub[j_idx], axis=1)


def _pairwise_l2_sparse(
    sub: sp.csr_matrix, *, max_pairs: int | None = None, seed: int = 0
) -> np.ndarray:
    return _pairwise_l2_dense(
        np.asarray(sub.toarray(), dtype=np.float64), max_pairs=max_pairs, seed=seed
    )


def _paired_cell_l2_norms_dense(ref: np.ndarray, man: np.ndarray) -> np.ndarray:
    return np.linalg.norm(man - ref, axis=1)


def _paired_cell_l2_norms_sparse(ref: sp.csr_matrix, man: sp.csr_matrix) -> np.ndarray:
    return _row_norms_csr((man - ref).tocsr())


def _shift_dots_dense(ref: np.ndarray, man: np.ndarray) -> np.ndarray:
    delta = man - ref
    u = np.mean(man, axis=0) - np.mean(ref, axis=0)
    return np.asarray(delta @ u, dtype=np.float64).ravel()


def _shift_dots_sparse(ref: sp.csr_matrix, man: sp.csr_matrix) -> np.ndarray:
    delta = (man - ref).tocsr()
    u = np.asarray(man.mean(axis=0) - ref.mean(axis=0)).ravel()
    return np.asarray(delta @ u, dtype=np.float64).ravel()


def _row_metric(
    *,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    category: str,
    metric_name: str,
    space: str,
    summary: DistributionSummary,
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
        "null_value": np.nan,
        "n_cells": n_cells,
        "seed": seed,
    }
    if extra:
        row.update(extra)
    return row


def _append_stats_rows(
    rows: list[dict[str, Any]],
    *,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    space: str,
    n_cells: int,
    seed: int,
    ref_row_norms: np.ndarray,
    man_row_norms: np.ndarray,
    ref_col_variances: np.ndarray,
    man_col_variances: np.ndarray,
) -> None:
    base = dict(
        dataset_id=dataset_id,
        model=model,
        intervention_id=intervention_id,
        intervention_name=intervention_name,
        category="embedding_stats",
        space=space,
        n_cells=n_cells,
        seed=seed,
    )
    for metric_name, values in (
        ("mean_row_l2_norm_ref", ref_row_norms),
        ("mean_row_l2_norm_man", man_row_norms),
        ("col_variance_ref", ref_col_variances),
        ("col_variance_man", man_col_variances),
    ):
        rows.append(
            _row_metric(
                **base,
                metric_name=metric_name,
                summary=distribution_summary(values),
            )
        )


def compute_embedding_stats(
    *,
    bundle: Any,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    seed: int,
    ref_cache: Any | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = bundle.emb_ref.shape[0]

    for space, ref, man in (
        ("raw", bundle.raw_ref, bundle.raw_man),
        ("embedding", bundle.emb_ref, bundle.emb_man),
    ):
        if ref_cache is not None:
            if space == "raw":
                ref_row_norms = ref_cache.raw_ref_row_norms
                ref_col_variances = ref_cache.raw_col_variances
            else:
                ref_row_norms = ref_cache.emb_ref_row_norms
                ref_col_variances = ref_cache.emb_col_variances
        elif space == "raw":
            ref_row_norms = _row_norms_csr(ref)
            ref_col_variances = _col_variances_csr(ref)
        else:
            ref_row_norms = np.linalg.norm(ref, axis=1)
            ref_col_variances = _col_variances_dense(ref)

        if space == "raw":
            man_row_norms = _row_norms_csr(man)
            man_col_variances = _col_variances_csr(man)
        else:
            man_row_norms = np.linalg.norm(man, axis=1)
            man_col_variances = _col_variances_dense(man)

        _append_stats_rows(
            rows,
            dataset_id=dataset_id,
            model=model,
            intervention_id=intervention_id,
            intervention_name=intervention_name,
            space=space,
            n_cells=n,
            seed=seed,
            ref_row_norms=ref_row_norms,
            man_row_norms=man_row_norms,
            ref_col_variances=ref_col_variances,
            man_col_variances=man_col_variances,
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
    ref_cache: Any | None = None,
    pairwise_max_pairs: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = bundle.emb_ref.shape[0]
    pair_seed = int(seed) + 17_001

    for space, ref, man in (
        ("raw", bundle.raw_ref, bundle.raw_man),
        ("embedding", bundle.emb_ref, bundle.emb_man),
    ):
        if space == "raw":
            paired_norms = _paired_cell_l2_norms_sparse(ref, man)
            shift_dots = _shift_dots_sparse(ref, man)
        else:
            paired_norms = _paired_cell_l2_norms_dense(ref, man)
            shift_dots = _shift_dots_dense(ref, man)

        base = dict(
            dataset_id=dataset_id,
            model=model,
            intervention_id=intervention_id,
            intervention_name=intervention_name,
            category="embedding_shift",
            space=space,
            n_cells=n,
            seed=seed,
        )
        rows.append(
            _row_metric(
                **base,
                metric_name="paired_cell_l2_norm",
                summary=distribution_summary(paired_norms),
            )
        )
        rows.append(
            _row_metric(
                **base,
                metric_name="shift_dot_with_mean",
                summary=distribution_summary(shift_dots),
            )
        )

        if ref_cache is not None:
            if space == "raw":
                ref_within = ref_cache.raw_within_pairwise_l2
            else:
                ref_within = ref_cache.emb_within_pairwise_l2
            indices = ref_cache.pairwise_cell_indices
        else:
            indices = _sample_pairwise_cell_indices(n, min(n, 500), seed=seed)
            if space == "raw":
                ref_sub = ref[indices]
                ref_within = _pairwise_l2_sparse(
                    ref_sub, max_pairs=pairwise_max_pairs, seed=pair_seed
                )
            else:
                ref_sub = ref[indices]
                ref_within = _pairwise_l2_dense(
                    ref_sub, max_pairs=pairwise_max_pairs, seed=pair_seed
                )

        rows.append(
            _row_metric(
                **base,
                metric_name="within_ref_pairwise_l2",
                summary=distribution_summary(ref_within),
            )
        )

        if space == "raw":
            man_sub = man[indices]
            man_within = _pairwise_l2_sparse(
                man_sub, max_pairs=pairwise_max_pairs, seed=pair_seed + 2
            )
        else:
            man_sub = man[indices]
            man_within = _pairwise_l2_dense(
                man_sub, max_pairs=pairwise_max_pairs, seed=pair_seed + 2
            )

        rows.append(
            _row_metric(
                **base,
                metric_name="within_man_pairwise_l2",
                summary=distribution_summary(man_within),
            )
        )

    return pd.DataFrame(rows)
