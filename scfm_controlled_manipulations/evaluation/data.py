"""Load paired reference/manipulation embeddings for structure evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scfm_controlled_manipulations.io import manipulations_dir


@dataclass(frozen=True)
class AlignedBundle:
    """Paired embedding matrices with identical ``obs`` ordering (see ``assert_obs_aligned``)."""

    emb_ref: np.ndarray
    emb_man: np.ndarray
    obs: pd.DataFrame
    uses_full_reference: bool = True


@dataclass(frozen=True)
class ObsAlignment:
    """Cell-id overlap diagnostics for aligning two AnnData-like obs indexes."""

    reference_obs: pd.Index
    candidate_obs: pd.Index
    shared_obs: pd.Index
    missing_in_candidate: pd.Index
    extra_in_candidate: pd.Index

    @property
    def is_full_reference(self) -> bool:
        return self.reference_obs.equals(self.shared_obs) and len(self.extra_in_candidate) == 0


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


def common_obs_alignment(
    reference_obs: pd.Index | list[str],
    candidate_obs: pd.Index | list[str],
    *,
    reference_label: str,
    candidate_label: str,
    allow_extra_candidate: bool = False,
) -> ObsAlignment:
    """Return shared cell IDs in reference order, with clear diagnostics for dropped cells."""
    reference_index = pd.Index(reference_obs)
    candidate_index = pd.Index(candidate_obs)
    if not reference_index.is_unique:
        raise ValueError(f"{reference_label} obs_names are not unique")
    if not candidate_index.is_unique:
        raise ValueError(f"{candidate_label} obs_names are not unique")

    shared = reference_index.intersection(candidate_index, sort=False)
    missing = reference_index.difference(candidate_index)
    extra = candidate_index.difference(reference_index)
    if len(shared) == 0:
        raise ValueError(
            f"obs_names have no overlap between {reference_label} and {candidate_label}: "
            f"missing_in_{candidate_label}={len(missing)} extra_in_{candidate_label}={len(extra)}"
        )
    if len(extra) and not allow_extra_candidate:
        raise ValueError(
            f"obs_names contain unexpected cells in {candidate_label}: "
            f"missing_in_{candidate_label}={len(missing)} extra_in_{candidate_label}={len(extra)}"
        )

    return ObsAlignment(
        reference_obs=reference_index,
        candidate_obs=candidate_index,
        shared_obs=shared,
        missing_in_candidate=missing,
        extra_in_candidate=extra,
    )


def obs_position_indexer(
    source_obs: pd.Index | list[str], target_obs: pd.Index | list[str]
) -> np.ndarray:
    """Integer positions that order ``source_obs`` rows like ``target_obs``."""
    source_index = pd.Index(source_obs)
    target_index = pd.Index(target_obs)
    indexer = source_index.get_indexer(target_index)
    if np.any(indexer < 0):
        missing = target_index[indexer < 0]
        raise ValueError(f"target_obs contains {len(missing)} cells not present in source_obs")
    return indexer


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


def load_manipulation_counts(
    results_dir: Path,
    intervention_id: str,
    manipulations_dir_path: Path | None = None,
) -> sp.csr_matrix:
    """Load count matrix from a manipulation h5ad (used by scib-metrics only)."""
    path = manipulations_dir(results_dir, manipulations_dir_path) / f"{intervention_id}.h5ad"
    adata = read_h5ad_for_eval(path)
    return _as_float_csr(adata.X)
