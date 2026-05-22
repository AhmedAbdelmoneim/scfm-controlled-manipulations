"""Load paired reference/manipulation matrices for structure evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass(frozen=True)
class AlignedBundle:
    """Paired matrices with identical ``obs`` ordering (see ``assert_obs_aligned``)."""

    raw_ref: Any
    raw_man: Any
    emb_ref: np.ndarray
    emb_man: np.ndarray
    obs: pd.DataFrame


def assert_obs_aligned(
    names_a: pd.Index,
    names_b: pd.Index,
    label_a: str,
    label_b: str,
) -> None:
    if not names_a.equals(names_b):
        first = None
        m = min(len(names_a), len(names_b))
        for i in range(m):
            if names_a[i] != names_b[i]:
                first = (i, names_a[i], names_b[i])
                break
        raise ValueError(
            f"obs_names are not aligned between {label_a} and {label_b}. First mismatch: {first}"
        )


def assert_obs_same_set(
    names_a: pd.Index,
    names_b: pd.Index,
    label_a: str,
    label_b: str,
) -> None:
    if names_a.equals(names_b):
        return
    missing = names_a.difference(names_b)
    extra = names_b.difference(names_a)
    if len(missing) or len(extra):
        raise ValueError(
            f"obs_names differ between {label_a} and {label_b}: "
            f"missing_in_{label_b}={len(missing)} extra_in_{label_b}={len(extra)}"
        )


def dense_embedding_aligned_to_obs(
    adata: ad.AnnData,
    target_obs: pd.Index | list[str],
    *,
    label: str,
) -> np.ndarray:
    """Dense float32 embedding rows ordered like ``target_obs`` (reindex if needed)."""
    target_index = pd.Index(target_obs)
    assert_obs_same_set(adata.obs_names, target_index, label, "target_obs")
    if adata.obs_names.equals(target_index):
        return _as_dense_embedding(adata.X)
    return _as_dense_embedding(adata[target_index].X)


def _as_dense_embedding(x: Any) -> np.ndarray:
    if sp.issparse(x):
        return np.asarray(x.todense(), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def read_h5ad_for_eval(path: Path | str) -> ad.AnnData:
    """Load h5ad for evaluation; deduplicate ``var_names`` in memory if needed."""
    adata = ad.read_h5ad(path)
    if not adata.var_names.is_unique:
        # Avoid "-" suffix collisions with gene names like SNORD115-1.
        adata.var_names_make_unique(join="_dup_")
    return adata


def _as_float_csr(x: Any) -> sp.csr_matrix:
    if sp.issparse(x):
        return x.tocsr().astype(np.float64)
    return sp.csr_matrix(np.asarray(x, dtype=np.float64))
