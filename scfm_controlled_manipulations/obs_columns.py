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


def resolve_batch_column(
    obs: pd.DataFrame | Iterable[str],
    configured: str | None = "batch",
) -> str | None:
    cols = obs.columns if isinstance(obs, pd.DataFrame) else obs
    return resolve_obs_column(cols, configured, candidates=BATCH_COLUMN_CANDIDATES)
