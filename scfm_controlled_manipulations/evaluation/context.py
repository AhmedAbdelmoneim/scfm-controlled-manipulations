"""Shared evaluation context: load reference data once, reuse across interventions."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
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
    common_obs_alignment,
    dense_embedding_aligned_to_obs,
    obs_position_indexer,
    read_h5ad_for_eval,
)
from scfm_controlled_manipulations.evaluation.leiden_cache import LeidenCache
from scfm_controlled_manipulations.io import embedding_path, manipulations_dir

logger = logging.getLogger(__name__)


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


def load_dataset_context(
    results_dir: Path,
    manipulations_dir_path: Path | None = None,
) -> DatasetEvaluateContext:
    ref_path = manipulations_dir(results_dir, manipulations_dir_path) / "reference.h5ad"
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
    alignment = common_obs_alignment(
        target_obs,
        ad_emb_man.obs_names,
        reference_label="emb_ref",
        candidate_label="emb_man",
    )
    if not alignment.is_full_reference:
        logger.warning(
            "Using shared cell subset for %s/%s: reference=%d embedding=%d shared=%d "
            "missing_in_embedding=%d extra_in_embedding=%d",
            model,
            intervention_id,
            len(alignment.reference_obs),
            len(alignment.candidate_obs),
            len(alignment.shared_obs),
            len(alignment.missing_in_candidate),
            len(alignment.extra_in_candidate),
        )

    ref_indexer = obs_position_indexer(target_obs, alignment.shared_obs)
    emb_ref = model_ctx.emb_ref[ref_indexer]
    emb_man = dense_embedding_aligned_to_obs(
        ad_emb_man, alignment.shared_obs, label="emb_man"
    )
    obs = dataset_ctx.obs.loc[alignment.shared_obs].copy()

    return AlignedBundle(
        emb_ref=emb_ref,
        emb_man=emb_man,
        obs=obs,
        uses_full_reference=alignment.is_full_reference,
    )
