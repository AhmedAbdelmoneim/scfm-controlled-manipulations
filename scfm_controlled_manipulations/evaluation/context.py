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

from scfm_controlled_manipulations.evaluation.data import (
    AlignedBundle,
    assert_obs_same_set,
    dense_embedding_aligned_to_obs,
    read_h5ad_for_eval,
)
from scfm_controlled_manipulations.evaluation.leiden_cache import LeidenCache
from scfm_controlled_manipulations.io import embedding_path


@dataclass
class DatasetEvaluateContext:
    """Reference obs shared by all models for one dataset."""

    obs: pd.DataFrame
    n_cells: int


@dataclass
class ModelEvaluateContext:
    """Per-model reference embedding and Leiden-on-reference cache."""

    emb_ref: np.ndarray
    leiden_cache: LeidenCache = field(default_factory=LeidenCache)
    ref_stats_cache: ReferenceStatsShiftCache | None = None


def load_dataset_context(results_dir: Path) -> DatasetEvaluateContext:
    ref_path = results_dir / "manipulations" / "reference.h5ad"
    ad_ref = read_h5ad_for_eval(ref_path)
    return DatasetEvaluateContext(
        obs=ad_ref.obs.copy(),
        n_cells=int(ad_ref.n_obs),
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
    """Load manipulation embedding; reuse cached reference embedding from context."""
    ad_emb_man = read_h5ad_for_eval(embedding_path(embeddings_root, model, intervention_id))

    target_obs = dataset_ctx.obs.index
    assert_obs_same_set(target_obs, ad_emb_man.obs_names, "emb_ref", "emb_man")
    emb_man = dense_embedding_aligned_to_obs(ad_emb_man, target_obs, label="emb_man")

    return AlignedBundle(
        emb_ref=model_ctx.emb_ref,
        emb_man=emb_man,
        obs=dataset_ctx.obs,
    )
