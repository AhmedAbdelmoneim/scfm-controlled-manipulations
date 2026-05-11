from __future__ import annotations

import anndata as ad
import numpy as np
import scipy.sparse as sp

from scfm_controlled_manipulations.base import Intervention


class Downsample(Intervention):
    name = "downsample"

    def __init__(self, fraction: float):
        if not 0 < fraction <= 1:
            raise ValueError("fraction must satisfy 0 < fraction <= 1")
        self.fraction = fraction

    def apply(self, adata: ad.AnnData, seed: int | None = None) -> ad.AnnData:
        rng = np.random.default_rng(seed)
        X = adata.X

        if not sp.issparse(X):
            X = sp.csr_matrix(X)

        # Binomial subsampling on the non-zero entries
        new_data = rng.binomial(n=X.data.astype(np.int64), p=self.fraction).astype(X.dtype)

        out = adata.copy()
        out.X = sp.csr_matrix((new_data, X.indices.copy(), X.indptr.copy()), shape=X.shape)
        out.X.eliminate_zeros()

        out.uns["scfm_intervention"] = {
            self.name: {
                "fraction": self.fraction,
                "seed": seed,
                "median_counts_before": float(np.median(np.asarray(X.sum(axis=1)).flatten())),
                "median_counts_after": float(np.median(np.asarray(out.X.sum(axis=1)).flatten())),
            }
        }
        return out
