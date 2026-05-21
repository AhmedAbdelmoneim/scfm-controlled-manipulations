"""One-time reference-side precomputation before intervention jobs run."""

from __future__ import annotations

import logging
import time

from scfm_controlled_manipulations.evaluation.context import ModelEvaluateContext
from scfm_controlled_manipulations.evaluation.metrics_clustering import run_leiden_labels

logger = logging.getLogger(__name__)


def precompute_reference_leiden(
    model_ctx: ModelEvaluateContext,
    *,
    k_values: list[int],
    distance_metrics: list[str],
    leiden_resolutions: list[float],
    seed: int,
) -> None:
    """Warm Leiden neighbor graphs and labels on the reference embedding (single-threaded)."""
    resolutions = sorted(set(leiden_resolutions))
    jobs = [
        (metric, int(k), float(resolution))
        for metric in distance_metrics
        for k in k_values
        for resolution in resolutions
    ]
    logger.info("Precomputing reference Leiden (%d parameter combos)", len(jobs))
    t0 = time.perf_counter()
    for metric, k, resolution in jobs:
        run_leiden_labels(
            model_ctx.emb_ref,
            k=k,
            metric=metric,
            resolution=resolution,
            seed=seed,
            leiden_cache=model_ctx.leiden_cache,
        )
    logger.info(
        "Reference Leiden precompute done (%.1fs; cache entries=%d)",
        time.perf_counter() - t0,
        len(model_ctx.leiden_cache),
    )
