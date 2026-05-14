from __future__ import annotations

import anndata as ad
import numpy as np
import scipy.sparse as sp

from scfm_controlled_manipulations.base import Intervention


class PoissonResampling(Intervention):
    name = "poisson_resampling"

    def __init__(self, iterations: int = 1):
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        self.iterations = iterations

    def apply(self, adata: ad.AnnData, seed: int | None = None) -> ad.AnnData:
        rng = np.random.default_rng(seed)
        X = adata.X
        if not sp.issparse(X):
            X = sp.csr_matrix(X)

        current = X.copy()
        total_counts_by_iteration = [float(np.asarray(current.sum()).item())]
        for _ in range(self.iterations):
            rates = current.data.astype(np.float64)
            new_data = rng.poisson(rates).astype(X.dtype)
            current = sp.csr_matrix(
                (new_data, current.indices.copy(), current.indptr.copy()),
                shape=current.shape,
            )
            current.eliminate_zeros()
            total_counts_by_iteration.append(float(np.asarray(current.sum()).item()))

        out = adata.copy()
        out.X = current

        out.uns["scfm_intervention"] = {
            self.name: {
                "iterations": self.iterations,
                "seed": seed,
                "total_counts_by_iteration": total_counts_by_iteration,
            }
        }
        return out
