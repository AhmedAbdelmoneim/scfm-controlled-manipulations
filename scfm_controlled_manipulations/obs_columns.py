"""Resolve observation metadata column names across dataset naming conventions."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

CELL_TYPE_COLUMN_CANDIDATES: tuple[str, ...] = (
    "cell_type",
    "celltype",
    "CellType",
    "cell.type",
    "cell_types",
    "celltypes",
    "cell_type_col",
    "celltype_col",
    "Cell.type",
    "cell_types_col",
)

# Finer-grained labels when ``cell_type`` is missing or too coarse for stratified sampling.
STRATIFY_FALLBACK_CANDIDATES: tuple[str, ...] = (
    "supercluster_term",
    "cluster_id",
    "subcluster_id",
    "scanvi_label",
    "author_cell_type",
    "cluster",
    "clusters",
    "Cluster",
    "cell_annotation",
    "annotation",
)

# Per-atlas cell-type / annotation column overrides (evaluation, dashboard, stratified sampling).
# brain: native ``cell_type`` is only neuron/leukocyte; use 21 superclusters instead.
ATLAS_CELL_TYPE_COLUMNS: dict[str, str] = {
    "brain": "supercluster_term",
}

SWEEP_ATLAS_KEYS: tuple[str, ...] = (
    "arterial",
    "brain",
    "immune",
    "lung",
    "retina",
    "tabula_sapiens",
)

BATCH_COLUMN_CANDIDATES: tuple[str, ...] = (
    "batch",
    "Batch",
    "batch_id",
    "batchid",
    "donor",
    "sample",
    "sample_id",
    "batch_col",
)


def _column_lookup(columns: Iterable[str]) -> dict[str, str]:
    """Map lowercase names to the actual column string in ``obs``."""
    return {str(c).lower(): str(c) for c in columns}


def resolve_obs_column(
    columns: Iterable[str],
    configured: str | None,
    *,
    candidates: tuple[str, ...],
) -> str | None:
    """Pick the first matching obs column: exact config name, then known aliases."""
    col_list = [str(c) for c in columns]
    col_set = set(col_list)
    by_lower = _column_lookup(col_list)

    if configured:
        key = str(configured).strip()
        if key in col_set:
            return key
        resolved = by_lower.get(key.lower())
        if resolved is not None:
            return resolved

    for cand in candidates:
        if cand in col_set:
            return cand
        resolved = by_lower.get(cand.lower())
        if resolved is not None:
            return resolved
    return None


def resolve_cell_type_column(
    obs: pd.DataFrame | Iterable[str],
    configured: str | None = "cell_type",
) -> str | None:
    cols = obs.columns if isinstance(obs, pd.DataFrame) else obs
    return resolve_obs_column(cols, configured, candidates=CELL_TYPE_COLUMN_CANDIDATES)


def atlas_key_from_dataset_id(dataset_id: str | None) -> str | None:
    """Map ``brain`` or ``brain_n200_s0`` → atlas key ``brain``."""
    if not dataset_id:
        return None
    if dataset_id in ATLAS_CELL_TYPE_COLUMNS or dataset_id in SWEEP_ATLAS_KEYS:
        return dataset_id
    for atlas in sorted(SWEEP_ATLAS_KEYS, key=len, reverse=True):
        if dataset_id.startswith(f"{atlas}_"):
            return atlas
    return None


def resolve_cell_type_column_for_dataset(
    obs: pd.DataFrame | Iterable[str],
    configured: str | None = "cell_type",
    *,
    atlas: str | None = None,
    dataset_id: str | None = None,
) -> str | None:
    """Resolve cell-type column with per-atlas overrides from ``ATLAS_CELL_TYPE_COLUMNS``."""
    cols = obs.columns if isinstance(obs, pd.DataFrame) else obs
    col_list = list(cols)
    atlas_key = atlas or atlas_key_from_dataset_id(dataset_id)
    if atlas_key and atlas_key in ATLAS_CELL_TYPE_COLUMNS:
        override = ATLAS_CELL_TYPE_COLUMNS[atlas_key]
        resolved = resolve_obs_column(col_list, override, candidates=(override,))
        if resolved is not None:
            return resolved
    return resolve_cell_type_column(obs, configured)


def resolve_stratify_column(
    obs: pd.DataFrame,
    *,
    atlas: str | None = None,
    configured: str | None = "cell_type",
    atlas_overrides: dict[str, str] | None = None,
    min_labels: int = 5,
) -> str:
    """Column for proportional stratified subsampling.

    Uses per-atlas overrides when set, else ``cell_type`` if it has enough unique
    labels, else the first matching fallback cluster/annotation column.
    """
    cols = list(obs.columns)
    overrides = atlas_overrides if atlas_overrides is not None else ATLAS_CELL_TYPE_COLUMNS

    if atlas and atlas in overrides:
        override = overrides[atlas]
        if override in cols:
            return override
        raise ValueError(f"Atlas {atlas!r} stratify override {override!r} not in obs columns")

    primary = resolve_cell_type_column(cols, configured)
    if primary is not None:
        n_labels = int(obs[primary].astype(str).nunique())
        if n_labels >= min_labels:
            return primary

    fallback = resolve_obs_column(cols, None, candidates=STRATIFY_FALLBACK_CANDIDATES)
    if fallback is not None:
        return fallback

    if primary is not None:
        return primary

    raise ValueError(
        "No stratification column found; "
        f"configured={configured!r}, atlas={atlas!r}, obs columns={cols[:25]}..."
    )


def resolve_batch_column(
    obs: pd.DataFrame | Iterable[str],
    configured: str | None = "batch",
) -> str | None:
    cols = obs.columns if isinstance(obs, pd.DataFrame) else obs
    return resolve_obs_column(cols, configured, candidates=BATCH_COLUMN_CANDIDATES)
