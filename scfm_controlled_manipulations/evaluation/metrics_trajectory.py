"""Trajectory inference metrics computed from reference embeddings."""

from __future__ import annotations

import logging
from typing import Any
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr

from scfm_controlled_manipulations.evaluation.data import _as_dense_embedding
from scfm_controlled_manipulations.evaluation.metrics_common import make_metric_row, scalar_summary

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

logger = logging.getLogger(__name__)

TRAJECTORY_CATEGORY = "trajectory_metrics"


def dpt_pseudotime(
    embedding: np.ndarray,
    reference_order: np.ndarray,
    milestones: np.ndarray,
    *,
    n_neighbors: int,
    n_dcs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer pseudotime from an embedding with Scanpy DPT.

    The DPT root is the cell nearest to the centroid of the earliest reference
    milestone, matching the trajectory notebook.
    """
    emb = np.asarray(embedding, dtype=np.float32)
    if emb.shape[0] < 3:
        pseudotime = np.full(emb.shape[0], np.nan, dtype=np.float64)
        return pseudotime, np.zeros(emb.shape[0], dtype=bool)

    k = min(max(2, int(n_neighbors)), emb.shape[0] - 1)
    n_dcs_eff = max(1, min(int(n_dcs), emb.shape[0] - 1))
    diffmap_components = max(n_dcs_eff, min(10, emb.shape[0] - 1))

    adata = ad.AnnData(X=emb)
    sc.pp.neighbors(adata, n_neighbors=k, use_rep="X")
    sc.tl.diffmap(adata, n_comps=diffmap_components)

    first = milestones.min()
    first_cells = np.where(reference_order == first)[0]
    centroid = emb[first_cells].mean(axis=0)
    root = int(first_cells[np.argmin(np.linalg.norm(emb[first_cells] - centroid, axis=1))])
    adata.uns["iroot"] = root

    sc.tl.dpt(adata, n_dcs=n_dcs_eff, n_branchings=0)
    pseudotime = adata.obs["dpt_pseudotime"].to_numpy(dtype=np.float64)
    finite = np.isfinite(pseudotime)
    return pseudotime, finite


def ordering_correlation(
    pseudotime: np.ndarray,
    reference_order: np.ndarray,
    finite: np.ndarray,
) -> dict[str, float]:
    """Dyneval-style monotone agreement between pseudotime and reference order."""
    pt = pseudotime[finite]
    ref = reference_order[finite]
    frac_connected = float(np.mean(finite)) if finite.size else float("nan")
    if pt.size < 3 or np.unique(pt).size < 2 or np.unique(ref).size < 2:
        return {"spearman": float("nan"), "frac_connected": frac_connected}

    rho = spearmanr(pt, ref).correlation
    return {
        "spearman": float(abs(rho)) if np.isfinite(rho) else float("nan"),
        "frac_connected": frac_connected,
    }


def compute_ordering_correlation(
    embedding: np.ndarray,
    reference_order: np.ndarray,
    *,
    n_neighbors: int,
    n_dcs: int,
) -> dict[str, float]:
    milestones = np.unique(reference_order)
    pseudotime, finite = dpt_pseudotime(
        embedding,
        reference_order,
        milestones,
        n_neighbors=n_neighbors,
        n_dcs=n_dcs,
    )
    return ordering_correlation(pseudotime, reference_order, finite)


def ordering_correlation_null_summary(
    embedding: np.ndarray,
    reference_order: np.ndarray,
    *,
    n_neighbors: int,
    n_dcs: int,
    n_permutations: int,
    seed: int,
    observed: float,
) -> dict[str, float]:
    """Permutation null for ordering correlation, including empirical p-value."""
    if n_permutations <= 0:
        return {
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "null_z": float("nan"),
            "null_p_value": float("nan"),
        }

    rng = np.random.default_rng(seed)
    null_values = np.empty(n_permutations, dtype=np.float64)
    for idx in range(n_permutations):
        shuffled = rng.permutation(reference_order)
        null_values[idx] = compute_ordering_correlation(
            embedding,
            shuffled,
            n_neighbors=n_neighbors,
            n_dcs=n_dcs,
        )["spearman"]

    finite_null = null_values[np.isfinite(null_values)]
    if finite_null.size == 0 or not np.isfinite(observed):
        p_value = float("nan")
    else:
        p_value = float((np.sum(finite_null >= observed) + 1) / (finite_null.size + 1))

    null_mean = float(np.nanmean(null_values))
    null_std = float(np.nanstd(null_values))
    z = (
        float((observed - null_mean) / (null_std + 1e-12))
        if np.isfinite(observed)
        else float("nan")
    )
    return {
        "null_mean": null_mean,
        "null_std": null_std,
        "null_z": z,
        "null_p_value": p_value,
    }


def _valid_trajectory_inputs(
    *,
    embedding: Any,
    obs_df: pd.DataFrame,
    trajectory_key: str,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    emb = _as_dense_embedding(embedding).astype(np.float64, copy=False)
    trajectory = pd.to_numeric(obs_df[trajectory_key], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(trajectory) & np.all(np.isfinite(emb), axis=1)
    emb = emb[valid]
    ref = trajectory[valid].astype(int)
    obs_names = obs_df.index[valid]
    return emb, ref, obs_names


def compute_trajectory_reference_rows(
    *,
    mat: Any,
    obs_df: pd.DataFrame,
    trajectory_key: str,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    space_label: str,
    seed: int,
    n_neighbors: int,
    n_dcs: int,
    n_permutations: int,
) -> list[dict[str, Any]]:
    """Compute reference-only trajectory metric rows for one model embedding."""
    if trajectory_key not in obs_df.columns:
        logger.warning("trajectory column %r not found in reference obs; skipping", trajectory_key)
        return []

    emb, ref, _obs_names = _valid_trajectory_inputs(
        embedding=mat,
        obs_df=obs_df,
        trajectory_key=trajectory_key,
    )
    if emb.shape[0] < 3 or np.unique(ref).size < 2:
        logger.warning(
            "trajectory column %r has insufficient valid ordered cells for model %s "
            "(valid_cells=%d milestones=%d); skipping",
            trajectory_key,
            model,
            emb.shape[0],
            np.unique(ref).size,
        )
        return []

    observed = compute_ordering_correlation(
        emb,
        ref,
        n_neighbors=n_neighbors,
        n_dcs=n_dcs,
    )
    null = ordering_correlation_null_summary(
        emb,
        ref,
        n_neighbors=n_neighbors,
        n_dcs=n_dcs,
        n_permutations=n_permutations,
        seed=seed,
        observed=observed["spearman"],
    )

    common_extra = {
        "trajectory_key": trajectory_key,
        "n_neighbors": int(n_neighbors),
        "n_dcs": int(n_dcs),
        "n_permutations": int(n_permutations),
        "n_cells_total": int(obs_df.shape[0]),
    }
    return [
        make_metric_row(
            dataset_id=dataset_id,
            model=model,
            intervention_id=intervention_id,
            intervention_name=intervention_name,
            metric_category=TRAJECTORY_CATEGORY,
            metric_name="ordering_correlation_spearman",
            space=space_label,
            summary=scalar_summary(observed["spearman"]),
            null_value=null["null_mean"],
            n_cells=int(emb.shape[0]),
            seed=seed,
            extra={**common_extra, **null},
        ),
        make_metric_row(
            dataset_id=dataset_id,
            model=model,
            intervention_id=intervention_id,
            intervention_name=intervention_name,
            metric_category=TRAJECTORY_CATEGORY,
            metric_name="frac_connected",
            space=space_label,
            summary=scalar_summary(observed["frac_connected"]),
            n_cells=int(emb.shape[0]),
            seed=seed,
            extra=common_extra,
        ),
    ]
