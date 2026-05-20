"""Shared evaluation context: load reference data once, reuse across interventions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scfm_controlled_manipulations.evaluation.reference_stats_shift import (
        ReferenceStatsShiftCache,
    )
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scfm_controlled_manipulations.evaluation.data import (
    AlignedBundle,
    _as_float_csr,
    assert_obs_aligned,
    dense_embedding_aligned_to_obs,
    read_h5ad_for_eval,
)
from scfm_controlled_manipulations.evaluation.knn_cache import KnnIndexCache
from scfm_controlled_manipulations.evaluation.leiden_cache import LeidenCache
from scfm_controlled_manipulations.io import embedding_path


@dataclass
class DatasetEvaluateContext:
    """Reference raw matrix and obs shared by all models for one dataset."""

    raw_ref: sp.csr_matrix
    obs: pd.DataFrame
    n_cells: int
    knn_cache: KnnIndexCache = field(default_factory=KnnIndexCache)


@dataclass
class ModelEvaluateContext:
    """Per-model reference embedding and Leiden-on-reference cache."""

    emb_ref: np.ndarray
    leiden_cache: LeidenCache = field(default_factory=LeidenCache)
    ref_stats_cache: ReferenceStatsShiftCache | None = None


def load_dataset_context(results_dir: Path) -> DatasetEvaluateContext:
    raw_ref_path = results_dir / "manipulations" / "reference.h5ad"
    ad_raw_ref = read_h5ad_for_eval(raw_ref_path)
    raw_ref = _as_float_csr(ad_raw_ref.X)
    return DatasetEvaluateContext(
        raw_ref=raw_ref,
        obs=ad_raw_ref.obs.copy(),
        n_cells=int(raw_ref.shape[0]),
    )


def load_model_context(
    embeddings_root: Path,
    model: str,
    reference_intervention_id: str,
    target_obs: pd.Index,
) -> ModelEvaluateContext:
    ad_emb_ref = read_h5ad_for_eval(
        embedding_path(embeddings_root, model, reference_intervention_id)
    )
    emb_ref = dense_embedding_aligned_to_obs(ad_emb_ref, target_obs, label="emb_ref")
    return ModelEvaluateContext(emb_ref=emb_ref)


def load_intervention_bundle(
    *,
    dataset_ctx: DatasetEvaluateContext,
    model_ctx: ModelEvaluateContext,
    results_dir: Path,
    embeddings_root: Path,
    model: str,
    intervention_id: str,
) -> AlignedBundle:
    """Load only manipulation matrices; reuse cached reference matrices from context."""
    raw_man_path = results_dir / "manipulations" / f"{intervention_id}.h5ad"
    ad_raw_man = read_h5ad_for_eval(raw_man_path)
    ad_emb_man = read_h5ad_for_eval(embedding_path(embeddings_root, model, intervention_id))

    target_obs = dataset_ctx.obs.index
    assert_obs_aligned(target_obs, ad_raw_man.obs_names, "raw_ref", "raw_man")
    emb_man = dense_embedding_aligned_to_obs(ad_emb_man, target_obs, label="emb_man")

    return AlignedBundle(
        raw_ref=dataset_ctx.raw_ref,
        raw_man=_as_float_csr(ad_raw_man.X),
        emb_ref=model_ctx.emb_ref,
        emb_man=emb_man,
        obs=dataset_ctx.obs,
    )
