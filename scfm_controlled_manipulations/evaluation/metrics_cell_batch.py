"""Cell-type and batch integration metrics via scib-metrics Benchmarker."""

from __future__ import annotations

from dataclasses import asdict
from functools import partial
import logging
from typing import Any
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scib_metrics
from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation
from scib_metrics.benchmark._core import _METRIC_TYPE, MetricAnnDataAPI

from scfm_controlled_manipulations.evaluation.data import _as_dense_embedding
from scfm_controlled_manipulations.evaluation.metrics_common import (
    make_metric_row,
    scalar_summary,
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*value_counts is deprecated.*",
    module=r"scib_metrics\..*",
)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

BIO_CATEGORY = "bio_conservation_metrics"
BATCH_CATEGORY = "batch_correction_metrics"
_METRIC_TYPE_TO_CATEGORY = {
    "Bio conservation": BIO_CATEGORY,
    "Batch correction": BATCH_CATEGORY,
}


def _default_bio_conservation() -> BioConservation:
    return BioConservation(
        isolated_labels=True,
        nmi_ari_cluster_labels_kmeans=True,
        nmi_ari_cluster_labels_leiden=True,
        silhouette_label=True,
        clisi_knn=True,
    )


def _disabled_bio_conservation() -> BioConservation:
    return BioConservation(
        isolated_labels=False,
        nmi_ari_cluster_labels_kmeans=False,
        nmi_ari_cluster_labels_leiden=False,
        silhouette_label=False,
        clisi_knn=False,
    )


def _default_batch_correction() -> BatchCorrection:
    return BatchCorrection(
        bras=True,
        ilisi_knn=True,
        kbet_per_label=True,
        graph_connectivity=True,
        pcr_comparison=True,
    )


def _disabled_batch_correction() -> BatchCorrection:
    return BatchCorrection(
        bras=False,
        ilisi_knn=False,
        kbet_per_label=False,
        graph_connectivity=False,
        pcr_comparison=False,
    )


def _obs_col_present(obs_df: pd.DataFrame, col: str | None) -> bool:
    return col is not None and col in obs_df.columns


def log_cell_batch_obs_columns(
    obs_df: pd.DataFrame,
    *,
    cell_type_col: str | None,
    batch_col: str | None,
) -> None:
    """Log whether configured cell-type / batch columns exist in reference ``obs``."""
    if cell_type_col is None and batch_col is None:
        logger.info("scib_benchmark: disabled in config (no column names)")
        return
    if cell_type_col is not None:
        if _obs_col_present(obs_df, cell_type_col):
            logger.info(
                "scib_benchmark: cell_type column %r found in reference obs", cell_type_col
            )
        else:
            logger.info(
                "scib_benchmark: cell_type column %r not in reference obs (bio metrics skipped)",
                cell_type_col,
            )
    if batch_col is not None:
        if _obs_col_present(obs_df, batch_col):
            logger.info("scib_benchmark: batch column %r found in reference obs", batch_col)
        else:
            logger.info(
                "scib_benchmark: batch column %r not in reference obs (batch metrics skipped)",
                batch_col,
            )


def _build_adata(
    *,
    counts: Any,
    embedding: Any,
    obs_df: pd.DataFrame,
    cell_type_col: str,
    batch_col: str,
) -> ad.AnnData:
    x = counts
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float64)
    emb = _as_dense_embedding(embedding).astype(np.float64, copy=False)
    obs = obs_df[[cell_type_col, batch_col]].copy()
    adata = ad.AnnData(X=x, obs=obs)
    adata.obsm["embedding"] = emb
    return adata


def _metadata_for_enabled_metrics(
    obs_df: pd.DataFrame,
    *,
    cell_type_col: str | None,
    batch_col: str | None,
    enable_bio: bool,
    enable_batch: bool,
) -> tuple[pd.DataFrame, str | None, str | None, bool, bool]:
    obs = obs_df.copy()
    label_key = cell_type_col if _obs_col_present(obs, cell_type_col) else None
    batch_key = batch_col if _obs_col_present(obs, batch_col) else None

    if enable_bio and label_key is None:
        logger.warning("scib_benchmark: bio metrics requested but cell_type column is missing")
        enable_bio = False
    if enable_batch and batch_key is None:
        logger.warning("scib_benchmark: batch metrics requested but batch column is missing")
        enable_batch = False
    if not enable_bio and not enable_batch:
        return obs, None, None, False, False

    if label_key is None:
        label_key = "__scfm_dummy_label"
        obs[label_key] = "all_cells"
    if batch_key is None:
        batch_key = "__scfm_dummy_batch"
        obs[batch_key] = "all_batches"
    return obs, label_key, batch_key, enable_bio, enable_batch


def _safe_benchmark(bm: Benchmarker) -> pd.DataFrame:
    """Run Benchmarker metrics, recording NaN for individual metric failures."""
    if not bm._prepared:
        bm.prepare()
    if bm._benchmarked:
        return bm._results

    for emb_key, ad_emb in bm._emb_adatas.items():
        for metric_type, metric_collection in bm._metric_collection_dict.items():
            for metric_name, use_metric_or_kwargs in asdict(metric_collection).items():
                if not use_metric_or_kwargs:
                    continue
                try:
                    metric_fn = getattr(scib_metrics, metric_name)
                    if isinstance(use_metric_or_kwargs, dict):
                        metric_fn = partial(metric_fn, **use_metric_or_kwargs)
                    metric_value = getattr(MetricAnnDataAPI, metric_name)(ad_emb, metric_fn)
                    if isinstance(metric_value, dict):
                        for key, value in metric_value.items():
                            row_name = f"{metric_name}_{key}"
                            bm._results.loc[row_name, emb_key] = value
                            bm._results.loc[row_name, _METRIC_TYPE] = metric_type
                    else:
                        bm._results.loc[metric_name, emb_key] = metric_value
                        bm._results.loc[metric_name, _METRIC_TYPE] = metric_type
                except Exception as exc:
                    logger.warning(
                        "scib_benchmark: %s failed for embedding=%s: %s",
                        metric_name,
                        emb_key,
                        exc,
                    )
                    if metric_name in (
                        "nmi_ari_cluster_labels_leiden",
                        "nmi_ari_cluster_labels_kmeans",
                    ):
                        for suffix in ("nmi", "ari"):
                            row_name = f"{metric_name}_{suffix}"
                            bm._results.loc[row_name, emb_key] = float("nan")
                            bm._results.loc[row_name, _METRIC_TYPE] = metric_type
                    else:
                        bm._results.loc[metric_name, emb_key] = float("nan")
                        bm._results.loc[metric_name, _METRIC_TYPE] = metric_type

    bm._benchmarked = True
    return bm._results


def _benchmarker_to_rows(
    results: pd.DataFrame,
    *,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    space_label: str,
    seed: int,
    n_cells: int,
    embedding_key: str = "embedding",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_name, row in results.iterrows():
        metric_type = row.get(_METRIC_TYPE)
        category = _METRIC_TYPE_TO_CATEGORY.get(str(metric_type))
        if category is None:
            continue
        value = row.get(embedding_key, float("nan"))
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            value_f = float("nan")
        rows.append(
            make_metric_row(
                dataset_id=dataset_id,
                model=model,
                intervention_id=intervention_id,
                intervention_name=intervention_name,
                metric_category=category,
                metric_name=str(metric_name),
                space=space_label,
                summary=scalar_summary(value_f),
                n_cells=n_cells,
                seed=seed,
            )
        )
    return rows


def _run_benchmarker_rows(
    *,
    counts: Any,
    embedding: Any,
    obs_df: pd.DataFrame,
    cell_type_col: str,
    batch_col: str,
    space_label: str,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    seed: int,
    n_cells: int,
    n_jobs: int,
    enable_bio: bool = True,
    enable_batch: bool = True,
) -> list[dict[str, Any]]:
    if not enable_bio and not enable_batch:
        return []
    adata = _build_adata(
        counts=counts,
        embedding=embedding,
        obs_df=obs_df,
        cell_type_col=cell_type_col,
        batch_col=batch_col,
    )
    bm = Benchmarker(
        adata,
        batch_key=batch_col,
        label_key=cell_type_col,
        embedding_obsm_keys=["embedding"],
        bio_conservation_metrics=(
            _default_bio_conservation() if enable_bio else _disabled_bio_conservation()
        ),
        batch_correction_metrics=(
            _default_batch_correction() if enable_batch else _disabled_batch_correction()
        ),
        n_jobs=max(1, int(n_jobs)),
        progress_bar=False,
    )
    results = _safe_benchmark(bm)
    return _benchmarker_to_rows(
        results,
        dataset_id=dataset_id,
        model=model,
        intervention_id=intervention_id,
        intervention_name=intervention_name,
        space_label=space_label,
        seed=seed,
        n_cells=n_cells,
    )


def compute_cell_batch_reference_rows(
    *,
    counts: Any,
    mat: Any,
    obs_df: pd.DataFrame,
    space_label: str,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    seed: int,
    cell_type_col: str | None,
    batch_col: str | None,
    n_cells: int,
    n_jobs: int = 1,
    enable_bio: bool = True,
    enable_batch: bool = True,
) -> list[dict[str, Any]]:
    """Benchmarker metrics for reference embedding only."""
    (
        obs_for_metrics,
        metric_cell_type_col,
        metric_batch_col,
        metric_enable_bio,
        metric_enable_batch,
    ) = _metadata_for_enabled_metrics(
        obs_df,
        cell_type_col=cell_type_col,
        batch_col=batch_col,
        enable_bio=enable_bio,
        enable_batch=enable_batch,
    )
    if metric_cell_type_col is None or metric_batch_col is None:
        return []
    logger.info(
        "scib_benchmark: reference=%s model=%s space=%s n_cells=%d bio=%s batch=%s",
        intervention_id,
        model,
        space_label,
        n_cells,
        metric_enable_bio,
        metric_enable_batch,
    )
    return _run_benchmarker_rows(
        counts=counts,
        embedding=mat,
        obs_df=obs_for_metrics,
        cell_type_col=metric_cell_type_col,
        batch_col=metric_batch_col,
        space_label=space_label,
        dataset_id=dataset_id,
        model=model,
        intervention_id=intervention_id,
        intervention_name=intervention_name,
        seed=seed,
        n_cells=n_cells,
        n_jobs=n_jobs,
        enable_bio=metric_enable_bio,
        enable_batch=metric_enable_batch,
    )
