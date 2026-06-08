"""QC helpers for count-based cell filtering."""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def count_matrix(adata: ad.AnnData):
    """Return the matrix used for total-count QC (prefers ``adata.raw.X`` counts)."""
    raw = adata.raw
    if raw is not None and raw.shape[0] == adata.n_obs and raw.n_vars == adata.n_vars:
        return raw.X
    return adata.X


def _is_csr_dataset(X: object) -> bool:
    try:
        from anndata._core.sparse_dataset import CSRDataset

        return isinstance(X, CSRDataset)
    except ImportError:
        return type(X).__name__ == "_CSRDataset"


def _backed_csr_row_sums(adata: ad.AnnData, X: object) -> np.ndarray:
    import h5py

    file_path = adata.filename
    group_path = getattr(getattr(X, "group", None), "name", None)
    if not file_path or not group_path:
        raise ValueError("Backed CSR matrix is missing filename or HDF5 group path")

    with h5py.File(file_path, "r") as handle:
        group = handle
        for part in group_path.strip("/").split("/"):
            group = group[part]
        indptr = group["indptr"][:]
        data = group["data"][:]
    return np.add.reduceat(data, indptr[:-1])


def observation_total_counts(adata: ad.AnnData) -> np.ndarray:
    """Per-observation total counts from the count matrix."""
    X = count_matrix(adata)
    if _is_csr_dataset(X):
        return _backed_csr_row_sums(adata, X)
    if sp.issparse(X):
        return np.asarray(X.sum(axis=1)).ravel()
    return np.asarray(X.sum(axis=1), dtype=np.float64).ravel()


def nonzero_count_indices(adata: ad.AnnData) -> pd.Index:
    """Observation names with strictly positive total counts."""
    totals = observation_total_counts(adata)
    return adata.obs_names[totals > 0]


def filter_zero_count_cells(adata: ad.AnnData) -> int:
    """Drop observations whose total count is zero. Returns remaining cell count."""
    n_input = adata.n_obs
    if n_input == 0:
        raise ValueError("Cannot filter zero-count cells on an empty AnnData object")

    totals = observation_total_counts(adata)
    keep_mask = totals > 0
    n_nonzero = int(keep_mask.sum())
    n_zero = n_input - n_nonzero

    if n_zero == 0:
        logger.info(
            "Zero-count cell filter: %d cells with non-zero counts (none removed)",
            n_nonzero,
        )
        return n_nonzero

    adata._inplace_subset_obs(keep_mask)
    logger.info(
        "Zero-count cell filter: %d cells with non-zero counts (%d removed of %d input cells)",
        adata.n_obs,
        n_zero,
        n_input,
    )
    if adata.n_obs < 1:
        raise ValueError("No cells remain after removing zero-count observations")
    return adata.n_obs
