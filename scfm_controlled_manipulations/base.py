from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import anndata as ad


class Intervention(ABC):
    """Base class for AnnData interventions.

    Metadata convention: store provenance and any quantities downstream metrics need in
    ``adata.uns["scfm_intervention"][self.name]`` (a mapping). Do not use other keys under
    ``adata.uns["scfm_intervention"]`` for the same intervention without namespacing.
    """

    name: str

    @abstractmethod
    def apply(self, adata: ad.AnnData, seed: int | None = None) -> ad.AnnData:
        """Return modified AnnData (may mutate in place; document per subclass)."""


class Metric(ABC):
    name: str

    @abstractmethod
    def compute(self, emb_ref: Any, emb_pert: Any) -> dict:
        """Return a plain dict of scalar or small-array results."""
