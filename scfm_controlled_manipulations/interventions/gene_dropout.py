from __future__ import annotations

import anndata as ad
import numpy as np
import scipy.sparse as sp

from scfm_controlled_manipulations.base import Intervention


class GeneDropout(Intervention):
    name = "gene_dropout"

    def __init__(self, dropout_rate: float = 0.3):
        if not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate must satisfy 0 <= dropout_rate < 1")
        self.dropout_rate = dropout_rate

    def apply(self, adata: ad.AnnData, seed: int | None = None) -> ad.AnnData:
        rng = np.random.default_rng(seed)
        X = adata.X
        if not sp.issparse(X):
            X = sp.csr_matrix(X)

        # Bernoulli dropout on non-zero entries
        keep_mask = rng.random(len(X.data)) >= self.dropout_rate
        new_data = X.data.copy()
        new_data[~keep_mask] = 0

        out = adata.copy()
        out.X = sp.csr_matrix((new_data, X.indices.copy(), X.indptr.copy()), shape=X.shape)
        out.X.eliminate_zeros()

        actual_dropped = (~keep_mask).sum() / len(keep_mask) if len(keep_mask) else 0
        out.uns["scfm_intervention"] = {
            self.name: {
                "dropout_rate": self.dropout_rate,
                "actual_fraction_dropped": float(actual_dropped),
                "seed": seed,
            }
        }
        return out
