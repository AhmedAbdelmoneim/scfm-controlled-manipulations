from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

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

    @staticmethod
    def _row_normalize_graph(graph: sp.spmatrix) -> sp.csr_matrix:
        graph = graph.tocsr()
        row_sums = np.asarray(graph.sum(axis=1)).ravel()
        row_scale = np.divide(
            1.0,
            row_sums,
            out=np.zeros_like(row_sums, dtype=np.float64),
            where=row_sums > 0,
        )
        return graph.multiply(row_scale[:, None]).tocsr()

    def apply(self, adata: ad.AnnData, seed: int | None = None) -> ad.AnnData:
        if adata.n_obs < 2 or adata.n_vars < 2:
            raise ValueError("local_smoothing requires at least 2 observations and 2 variables")

        # PCA on log-normalized data just for neighbour finding
        k = min(self.k, adata.n_obs)
        n_components = min(self.n_pcs, adata.n_obs - 1, adata.n_vars - 1)
        normalized = self._log_normalize(adata)

        n = adata.n_obs
        logger.info(
            "Running local smoothing with k=%d n_pcs=%d on %d cells",
            k,
            n_components,
            n,
        )
        sc.pp.pca(normalized, n_comps=n_components, random_state=seed, svd_solver="arpack")
        sc.pp.neighbors(
            normalized,
            n_neighbors=k,
            n_pcs=n_components,
            random_state=seed,
            key_added=self.name,
        )

        # Scanpy connectivities are weighted; include self counts, then row-normalize for smoothing.
        connectivities = normalized.obsp[f"{self.name}_connectivities"]
        S = self._row_normalize_graph(connectivities + sp.eye(n, format="csr"))

        # Apply S to RAW counts (not log-normalized)
        # Counts get averaged across neighbours
        X_raw = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
        smoothed_counts = (S @ X_raw).toarray()
        # Round to non-negative integers
        smoothed_counts = np.rint(np.clip(smoothed_counts, 0, None)).astype(np.int32)
        logger.info("Finished local smoothing")

        out = adata.copy()
        out.X = sp.csr_matrix(smoothed_counts)
        # Save operator for the equivariance metric (sidecar file pattern)
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
