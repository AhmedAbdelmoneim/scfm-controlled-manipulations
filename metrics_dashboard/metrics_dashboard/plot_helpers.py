"""Shared helpers for Plotly metric charts."""

from __future__ import annotations

import pandas as pd

from metrics_dashboard.config import MODEL_ORDER


def set1_column_title(cell_df: pd.DataFrame, col_val: str, column_facet: str) -> str:
    if col_val == "all":
        return ""
    if cell_df.empty:
        return str(col_val)
    param_key = (
        str(cell_df["param_key"].dropna().iloc[0])
        if "param_key" in cell_df.columns and cell_df["param_key"].notna().any()
        else column_facet
    )
    return f"{param_key} = {col_val}"


def prepend_reference_points(
    cell: pd.DataFrame,
    ref_df: pd.DataFrame,
    models: list[str],
) -> pd.DataFrame:
    """Insert reference (param_value=0) before each model's manipulation sweep."""
    if cell.empty:
        return cell
    parts: list[pd.DataFrame] = []
    for model in MODEL_ORDER:
        if model not in models:
            continue
        mcell = cell[cell["model"].astype(str) == model]
        if mcell.empty:
            continue
        r = ref_df[ref_df["model"].astype(str) == model]
        if not r.empty:
            ref_row = r.iloc[0:1].copy()
            ref_row["param_value"] = 0.0
            ref_row["param_key"] = "reference"
            parts.append(ref_row)
        parts.append(mcell)
    if not parts:
        return cell
    return pd.concat(parts, ignore_index=True)
