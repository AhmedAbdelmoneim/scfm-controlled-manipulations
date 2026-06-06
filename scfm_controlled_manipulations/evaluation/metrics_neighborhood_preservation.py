"""Neighborhood-preservation metrics (Venna & Kaski 2001).

Trustworthiness uses ``sklearn.manifold.trustworthiness``. Continuity is reserved
for a future release.

Permutation nulls: row-permutation shuffles of ref/man correspondence with the
geometry of each space preserved (same protocol as kNN metrics). By default one
shuffle is sampled; set ``n_null_permutations > 1`` to average across shuffles.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness

from scfm_controlled_manipulations.evaluation.metrics_common import (
    make_metric_row,
    scalar_summary,
)

logger = logging.getLogger(__name__)

METRIC_CATEGORY = "neighborhood_preservation_metrics"

NeighborhoodValueFn = Callable[[Any, Any, int, str], float]


def _null_seed(base_seed: int, metric_name: str, space: str, distance_metric: str, k: int) -> int:
    digest = hashlib.sha256(
        f"{base_seed}|{metric_name}|{space}|{distance_metric}|{k}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _permutation_null_mean(
    ref: Any,
    man: Any,
    *,
    value_fn: NeighborhoodValueFn,
    seed: int,
    metric_name: str,
    space: str,
    distance_metric: str,
    k: int,
    n_cells: int,
    n_null: int,
) -> float:
    null_rng = np.random.default_rng(_null_seed(seed, metric_name, space, distance_metric, k))
    null_sum = 0.0
    for _ in range(n_null):
        perm = null_rng.permutation(n_cells)
        null_sum += value_fn(ref, man[perm], k, distance_metric)
    return null_sum / n_null


def _trustworthiness_value(ref: Any, man: Any, k: int, distance_metric: str) -> float:
    return float(trustworthiness(ref, man, n_neighbors=k, metric=distance_metric))


def _neighborhood_metric_specs() -> tuple[tuple[str, NeighborhoodValueFn], ...]:
    return (("trustworthiness", _trustworthiness_value),)


def compute_neighborhood_preservation_metrics(
    *,
    bundle: Any,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    seed: int,
    distance_metrics: list[str],
    trustworthiness_k_values: list[int],
    n_null_permutations: int = 1,
) -> pd.DataFrame:
    """Compute neighborhood-preservation metrics for one intervention."""
    n_cells = bundle.emb_ref.shape[0]
    k_sorted = sorted(int(k) for k in trustworthiness_k_values)
    n_null = max(1, int(n_null_permutations))
    spaces = ("raw", "embedding")
    metric_specs = _neighborhood_metric_specs()

    logger.info(
        "neighborhood_preservation: intervention=%s n_cells=%d metrics=%s null_perms=%d",
        intervention_id,
        n_cells,
        [name for name, _ in metric_specs],
        n_null,
    )

    def mats_for(space: str) -> tuple[Any, Any]:
        if space == "raw":
            return bundle.raw_ref, bundle.raw_man
        return bundle.emb_ref, bundle.emb_man

    rows: list[dict[str, Any]] = []
    for space in spaces:
        ref_mat, man_mat = mats_for(space)
        for distance_metric in distance_metrics:
            for k in k_sorted:
                for metric_name, value_fn in metric_specs:
                    value = value_fn(ref_mat, man_mat, k, distance_metric)
                    null_value = _permutation_null_mean(
                        ref_mat,
                        man_mat,
                        value_fn=value_fn,
                        seed=seed,
                        metric_name=metric_name,
                        space=space,
                        distance_metric=distance_metric,
                        k=k,
                        n_cells=n_cells,
                        n_null=n_null,
                    )
                    rows.append(
                        make_metric_row(
                            dataset_id=dataset_id,
                            model=model,
                            intervention_id=intervention_id,
                            intervention_name=intervention_name,
                            metric_category=METRIC_CATEGORY,
                            metric_name=metric_name,
                            space=space,
                            summary=scalar_summary(value),
                            null_value=null_value,
                            n_cells=n_cells,
                            seed=seed,
                            extra={
                                "distance_metric": distance_metric,
                                "k": k,
                                "diffusion_t": np.nan,
                            },
                        )
                    )

    return pd.DataFrame(rows)
