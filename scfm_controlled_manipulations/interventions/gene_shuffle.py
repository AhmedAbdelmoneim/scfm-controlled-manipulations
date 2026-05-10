from __future__ import annotations

import anndata as ad
import numpy as np

from scfm_controlled_manipulations.base import Intervention


class GeneShuffle(Intervention):
    """Shuffle gene (column) order together with ``var``; ``X`` and ``layers`` column order follows.

    Uses the same permutation for ``var`` and matrix columns so the object stays valid AnnData.
    """

    name = "gene_shuffle"

    def __init__(self, variant: str = "random", n_strata: int | None = None) -> None:
        self.variant = variant
        self.n_strata = n_strata

    def apply(self, adata: ad.AnnData, seed: int | None = None) -> ad.AnnData:
        rng = np.random.default_rng(seed)
        n_vars = int(adata.n_vars)

        if self.variant == "random":
            perm = rng.permutation(n_vars)
        elif self.variant == "stratified":
            strata = max(1, int(self.n_strata or 1))
            perm = np.arange(n_vars)
            stride = int(np.ceil(n_vars / strata))
            for start in range(0, n_vars, stride):
                block = perm[start : start + stride]
                perm[start : start + stride] = rng.permutation(block)
        else:
            raise ValueError(f"Unknown variant: {self.variant!r}")

        out = adata[:, perm].copy()
        out.uns.setdefault("scfm_intervention", {})
        out.uns["scfm_intervention"][self.name] = {
            "variant": self.variant,
            "n_strata": self.n_strata,
            "seed": seed,
        }
        return out
