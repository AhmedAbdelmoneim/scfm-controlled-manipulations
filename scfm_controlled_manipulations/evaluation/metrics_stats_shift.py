"""Light global summaries (embedding_stats) and paired geometric shift (embedding_shift)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

from scfm_controlled_manipulations.evaluation.metrics_common import (
    distribution_summary,
    make_metric_row,
    scalar_summary,
)


def _col_means_dense(x: np.ndarray) -> np.ndarray:
    return np.asarray(np.mean(x, axis=0)).ravel()


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


def _paired_cell_l2_norms_dense(ref: np.ndarray, man: np.ndarray) -> np.ndarray:
    return np.linalg.norm(man - ref, axis=1)


def _shifts_dense_sub(ref_sub: np.ndarray, man_sub: np.ndarray) -> np.ndarray:
    return np.asarray(man_sub - ref_sub, dtype=np.float64)


def _pairwise_cosine_dense(
    shifts: np.ndarray, *, max_pairs: int | None = None, seed: int = 0
) -> np.ndarray:
    """Cosine similarity cos(shift_i, shift_j) for upper-triangle cell pairs."""
    n = shifts.shape[0]
    if n < 2:
        return np.array([], dtype=np.float64)
    norms = np.linalg.norm(shifts, axis=1)
    i_idx, j_idx = np.triu_indices(n, k=1)
    i_idx, j_idx = _cap_pair_indices(i_idx, j_idx, max_pairs, seed=seed)
    dots = np.sum(shifts[i_idx] * shifts[j_idx], axis=1)
    denom = norms[i_idx] * norms[j_idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, dots / denom, np.nan)


def _append_stats_rows(
    rows: list[dict[str, Any]],
    *,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    n_cells: int,
    seed: int,
    ref_row_norms: np.ndarray,
    man_row_norms: np.ndarray,
    ref_col_means: np.ndarray,
    man_col_means: np.ndarray,
    ref_col_variances: np.ndarray,
    man_col_variances: np.ndarray,
) -> None:
    base = dict(
        dataset_id=dataset_id,
        model=model,
        intervention_id=intervention_id,
        intervention_name=intervention_name,
        category="embedding_stats",
        space="embedding",
        n_cells=n_cells,
        seed=seed,
    )
    for metric_name, values in (
        ("mean_row_l2_norm_ref", ref_row_norms),
        ("mean_row_l2_norm_man", man_row_norms),
        ("col_mean_ref", ref_col_means),
        ("col_mean_man", man_col_means),
        ("col_variance_ref", ref_col_variances),
        ("col_variance_man", man_col_variances),
    ):
        rows.append(
            make_metric_row(
                dataset_id=base["dataset_id"],
                model=base["model"],
                intervention_id=base["intervention_id"],
                intervention_name=base["intervention_name"],
                metric_category=base["category"],
                metric_name=metric_name,
                space=base["space"],
                summary=distribution_summary(values),
                n_cells=base["n_cells"],
                seed=base["seed"],
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
    ref = bundle.emb_ref
    man = bundle.emb_man
    n = ref.shape[0]

    if ref_cache is not None:
        ref_row_norms = ref_cache.emb_ref_row_norms
        ref_col_means = ref_cache.emb_col_means
        ref_col_variances = ref_cache.emb_col_variances
    else:
        ref_row_norms = np.linalg.norm(ref, axis=1)
        ref_col_means = _col_means_dense(ref)
        ref_col_variances = _col_variances_dense(ref)

    man_row_norms = np.linalg.norm(man, axis=1)
    man_col_means = _col_means_dense(man)
    man_col_variances = _col_variances_dense(man)

    _append_stats_rows(
        rows,
        dataset_id=dataset_id,
        model=model,
        intervention_id=intervention_id,
        intervention_name=intervention_name,
        n_cells=n,
        seed=seed,
        ref_row_norms=ref_row_norms,
        man_row_norms=man_row_norms,
        ref_col_means=ref_col_means,
        man_col_means=man_col_means,
        ref_col_variances=ref_col_variances,
        man_col_variances=man_col_variances,
    )

    return pd.DataFrame(rows)


def global_distance_correlation(ref: np.ndarray, man: np.ndarray, *, metric: str) -> float:
    """Pearson r between upper-triangle pdist vectors for ref vs man (aligned rows)."""
    if ref.shape[0] < 2:
        return float("nan")
    d_ref = pdist(ref, metric=metric)
    d_man = pdist(man, metric=metric)
    return float(pearsonr(d_ref, d_man).statistic)


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
    distance_correlation_subsample_n: int | None = None,
    distance_metrics: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ref = bundle.emb_ref
    man = bundle.emb_man
    n = ref.shape[0]
    pair_seed = int(seed) + 17_001
    space = "embedding"

    paired_norms = _paired_cell_l2_norms_dense(ref, man)

    if ref_cache is not None:
        ref_within = ref_cache.emb_within_pairwise_l2
        indices = ref_cache.pairwise_cell_indices
    else:
        indices = _sample_pairwise_cell_indices(n, min(n, 500), seed=seed)
        ref_sub = ref[indices]
        ref_within = _pairwise_l2_dense(ref_sub, max_pairs=pairwise_max_pairs, seed=pair_seed)

    ref_sub = ref[indices]
    man_sub = man[indices]
    man_within = _pairwise_l2_dense(man_sub, max_pairs=pairwise_max_pairs, seed=pair_seed + 2)
    shifts_sub = _shifts_dense_sub(ref_sub, man_sub)
    shift_pairwise_cosine = _pairwise_cosine_dense(
        shifts_sub, max_pairs=pairwise_max_pairs, seed=pair_seed + 4
    )

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
        make_metric_row(
            dataset_id=base["dataset_id"],
            model=base["model"],
            intervention_id=base["intervention_id"],
            intervention_name=base["intervention_name"],
            metric_category=base["category"],
            metric_name="paired_cell_l2_norm",
            space=base["space"],
            summary=distribution_summary(paired_norms),
            n_cells=base["n_cells"],
            seed=base["seed"],
        )
    )
    rows.append(
        make_metric_row(
            dataset_id=base["dataset_id"],
            model=base["model"],
            intervention_id=base["intervention_id"],
            intervention_name=base["intervention_name"],
            metric_category=base["category"],
            metric_name="within_ref_pairwise_l2",
            space=base["space"],
            summary=distribution_summary(ref_within),
            n_cells=base["n_cells"],
            seed=base["seed"],
        )
    )
    rows.append(
        make_metric_row(
            dataset_id=base["dataset_id"],
            model=base["model"],
            intervention_id=base["intervention_id"],
            intervention_name=base["intervention_name"],
            metric_category=base["category"],
            metric_name="within_man_pairwise_l2",
            space=base["space"],
            summary=distribution_summary(man_within),
            n_cells=base["n_cells"],
            seed=base["seed"],
        )
    )
    rows.append(
        make_metric_row(
            dataset_id=base["dataset_id"],
            model=base["model"],
            intervention_id=base["intervention_id"],
            intervention_name=base["intervention_name"],
            metric_category=base["category"],
            metric_name="shift_pairwise_cosine",
            space=base["space"],
            summary=distribution_summary(shift_pairwise_cosine),
            n_cells=base["n_cells"],
            seed=base["seed"],
        )
    )

    if distance_correlation_subsample_n is not None and distance_metrics:
        if ref_cache is not None:
            dist_indices = ref_cache.pairwise_cell_indices
        else:
            dist_indices = _sample_pairwise_cell_indices(
                n, distance_correlation_subsample_n, seed=seed
            )
        for dist_metric in distance_metrics:
            dist_corr = global_distance_correlation(
                ref[dist_indices], man[dist_indices], metric=dist_metric
            )
            rows.append(
                make_metric_row(
                    dataset_id=base["dataset_id"],
                    model=base["model"],
                    intervention_id=base["intervention_id"],
                    intervention_name=base["intervention_name"],
                    metric_category=base["category"],
                    metric_name="global_distance_correlation",
                    space=base["space"],
                    summary=scalar_summary(dist_corr),
                    n_cells=base["n_cells"],
                    seed=base["seed"],
                    extra={"distance_metric": dist_metric},
                )
            )

    return pd.DataFrame(rows)
