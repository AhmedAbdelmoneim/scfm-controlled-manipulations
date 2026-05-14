from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from scfm_controlled_manipulations.base import Intervention

logger = logging.getLogger(__name__)


class LocalSmoothing(Intervention):
    name = "local_smoothing"

    def __init__(self, k: int = 15, n_pcs: int = 50):
        if k < 2:
            raise ValueError("k must be at least 2")
        if n_pcs < 1:
            raise ValueError("n_pcs must be at least 1")
        self.k = k
        self.n_pcs = n_pcs

    @staticmethod
    def _log_normalize(adata: ad.AnnData) -> ad.AnnData:
        normalized = adata.copy()
        sc.pp.normalize_total(normalized, target_sum=10_000)
        sc.pp.log1p(normalized)
        return normalized

    def apply(self, adata: ad.AnnData, seed: int | None = None) -> ad.AnnData:
        if adata.n_obs < 2 or adata.n_vars < 2:
            raise ValueError("local_smoothing requires at least 2 observations and 2 variables")

        # Clamp k and n_pcs to dataset size
        k = min(self.k, adata.n_obs)
        n_components = min(self.n_pcs, adata.n_obs - 1, adata.n_vars - 1)
        n = adata.n_obs

        logger.debug(
            "Running local smoothing with k=%d n_pcs=%d on %d cells",
            k,
            n_components,
            n,
        )

        # PCA on log-normalized data, used only for neighbour finding
        normalized = self._log_normalize(adata)
        sc.pp.pca(normalized, n_comps=n_components, random_state=seed, svd_solver="arpack")
        pcs = normalized.obsm["X_pca"]

        # Uniform-weight kNN graph in PCA space.
        # n_neighbors=k includes the cell itself as its own nearest neighbor,
        # so each row averages over self + (k-1) neighbors with weight 1/k.
        nn = NearestNeighbors(n_neighbors=k).fit(pcs)
        _, knn_idx = nn.kneighbors(pcs)

        rows = np.repeat(np.arange(n), k)
        cols = knn_idx.flatten()
        data = np.full(n * k, 1.0 / k)
        S = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

        # Apply S to RAW counts (not log-normalized).
        # Counts get averaged uniformly across the kNN neighborhood.
        X_raw = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
        smoothed_counts = (S @ X_raw).toarray()
        # Round to non-negative integers
        smoothed_counts = np.rint(np.clip(smoothed_counts, 0, None)).astype(np.int32)
        logger.debug("Finished local smoothing")

        out = adata.copy()
        out.X = sp.csr_matrix(smoothed_counts)
        # Save operator in .uns for the equivariance metric
        out.uns["scfm_intervention"] = {
            self.name: {
                "k": k,
                "requested_k": self.k,
                "n_pcs": n_components,
                "requested_n_pcs": self.n_pcs,
                "seed": seed,
                "operator_indptr": S.indptr.tolist(),
                "operator_indices": S.indices.tolist(),
                "operator_data": S.data.tolist(),
                "operator_shape": list(S.shape),
            }
        }
        return out
