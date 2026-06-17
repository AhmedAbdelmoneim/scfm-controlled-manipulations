"""Resolve observation metadata column names (mirrors scfm_controlled_manipulations.obs_columns)."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

try:
    from scfm_controlled_manipulations.obs_columns import (
        ATLAS_CELL_TYPE_COLUMNS,
        SWEEP_ATLAS_KEYS,
        atlas_key_from_dataset_id,
        resolve_cell_type_column_for_dataset,
    )
except ImportError:  # standalone dashboard install without scfm package
    ATLAS_CELL_TYPE_COLUMNS = {"brain": "supercluster_term"}
    SWEEP_ATLAS_KEYS = (
        "arterial",
        "brain",
        "immune",
        "lung",
        "retina",
        "tabula_sapiens",
    )

    def atlas_key_from_dataset_id(dataset_id: str | None) -> str | None:
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
        cols = obs.columns if isinstance(obs, pd.DataFrame) else obs
        col_list = list(cols)
        atlas_key = atlas or atlas_key_from_dataset_id(dataset_id)
        if atlas_key and atlas_key in ATLAS_CELL_TYPE_COLUMNS:
            override = ATLAS_CELL_TYPE_COLUMNS[atlas_key]
            resolved = resolve_obs_column(col_list, override, candidates=(override,))
            if resolved is not None:
                return resolved
        return resolve_cell_type_column(obs, configured)

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
    return {str(c).lower(): str(c) for c in columns}


def resolve_obs_column(
    columns: Iterable[str],
    configured: str | None,
    *,
    candidates: tuple[str, ...],
) -> str | None:
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


def resolve_batch_column(
    obs: pd.DataFrame | Iterable[str],
    configured: str | None = "batch",
) -> str | None:
    cols = obs.columns if isinstance(obs, pd.DataFrame) else obs
    return resolve_obs_column(cols, configured, candidates=BATCH_COLUMN_CANDIDATES)
