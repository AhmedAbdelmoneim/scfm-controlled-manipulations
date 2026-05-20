"""Load paired reference/manipulation matrices for structure evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scfm_controlled_manipulations.io import (
    embedding_path,
)


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


def load_aligned_bundle(
    *,
    results_dir: Path,
    embeddings_root: Path,
    model: str,
    intervention_id: str,
    reference_intervention_id: str,
) -> AlignedBundle:
    """Load raw + embedding reference/manipulation matrices with alignment checks."""
    manip_dir = results_dir / "manipulations"
    raw_ref_path = manip_dir / "reference.h5ad"
    raw_man_path = manip_dir / f"{intervention_id}.h5ad"
    emb_ref_path = embedding_path(embeddings_root, model, reference_intervention_id)
    emb_man_path = embedding_path(embeddings_root, model, intervention_id)

    ad_raw_ref = read_h5ad_for_eval(raw_ref_path)
    ad_raw_man = read_h5ad_for_eval(raw_man_path)
    ad_emb_ref = read_h5ad_for_eval(emb_ref_path)
    ad_emb_man = read_h5ad_for_eval(emb_man_path)

    target_obs = ad_raw_ref.obs_names
    assert_obs_aligned(target_obs, ad_raw_man.obs_names, "raw_ref", "raw_man")

    raw_ref = _as_float_csr(ad_raw_ref.X)
    raw_man = _as_float_csr(ad_raw_man.X)
    emb_ref = dense_embedding_aligned_to_obs(ad_emb_ref, target_obs, label="emb_ref")
    emb_man = dense_embedding_aligned_to_obs(ad_emb_man, target_obs, label="emb_man")

    return AlignedBundle(
        raw_ref=raw_ref,
        raw_man=raw_man,
        emb_ref=emb_ref,
        emb_man=emb_man,
        obs=ad_raw_ref.obs.copy(),
    )
