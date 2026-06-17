"""One-time reference precomputation for embedding_stats / embedding_shift metrics."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time

import numpy as np

from scfm_controlled_manipulations.evaluation.context import (
    DatasetEvaluateContext,
    ModelEvaluateContext,
)
from scfm_controlled_manipulations.evaluation.metrics_stats_shift import (
    _col_means_dense,
    _col_variances_dense,
    _pairwise_l2_dense,
    _sample_pairwise_cell_indices,
)

logger = logging.getLogger(__name__)


@dataclass
class ReferenceStatsShiftCache:
    """Reference-only quantities reused across interventions."""

    emb_ref_row_norms: np.ndarray
    emb_col_means: np.ndarray
    emb_col_variances: np.ndarray
    pairwise_cell_indices: np.ndarray
    emb_within_pairwise_l2: np.ndarray


def precompute_reference_stats_shift(
    model_ctx: ModelEvaluateContext,
    dataset_ctx: DatasetEvaluateContext,
    *,
    seed: int,
    pairwise_cell_subsample_n: int,
    pairwise_max_pairs: int | None = None,
) -> ReferenceStatsShiftCache:
    """Precompute full-reference norms/variances and subsampled within-ref pairwise spread."""
    t0 = time.perf_counter()
    emb_ref = model_ctx.emb_ref
    n_cells = dataset_ctx.n_cells

    emb_ref_row_norms = np.linalg.norm(emb_ref, axis=1)
    emb_col_means = _col_means_dense(emb_ref)
    emb_col_variances = _col_variances_dense(emb_ref)

    pairwise_cell_indices = _sample_pairwise_cell_indices(
        n_cells, pairwise_cell_subsample_n, seed=seed
    )
    emb_sub = emb_ref[pairwise_cell_indices]
    pair_seed = int(seed) + 17_001
    emb_within_pairwise_l2 = _pairwise_l2_dense(
        emb_sub, max_pairs=pairwise_max_pairs, seed=pair_seed + 1
    )

    cache = ReferenceStatsShiftCache(
        emb_ref_row_norms=emb_ref_row_norms,
        emb_col_means=emb_col_means,
        emb_col_variances=emb_col_variances,
        pairwise_cell_indices=pairwise_cell_indices,
        emb_within_pairwise_l2=emb_within_pairwise_l2,
    )
    logger.info(
        "Reference stats/shift precompute done (%.1fs; n_cells=%d n_sub=%d n_emb_pairs=%d)",
        time.perf_counter() - t0,
        n_cells,
        pairwise_cell_indices.size,
        emb_within_pairwise_l2.size,
    )
    return cache
