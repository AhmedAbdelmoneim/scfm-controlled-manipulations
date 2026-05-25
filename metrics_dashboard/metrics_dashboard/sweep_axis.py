"""Order and position manipulation sweep axes (numeric fractions vs categorical variants)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from metrics_dashboard.config import PARAM_KEYS

# Canonical order for gene_shuffle variant sweeps (not alphabetical).
GENE_SHUFFLE_VARIANT_ORDER: tuple[str, ...] = (
    "chromosome_control",
    "chromosome",
    "stratified",
    "random",
)


def _numeric_sort_key(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def ordered_param_values(
    values: pd.Series,
    *,
    intervention_name: str | None = None,
    param_key: str | None = None,
) -> list[str]:
    """Return sweep levels in display order."""
    uniq = [str(v) for v in values.dropna().unique()]
    if not uniq:
        return []
    key = param_key
    if key is None and intervention_name:
        key = PARAM_KEYS.get(intervention_name)
    if key == "variant" or intervention_name == "gene_shuffle":
        order = {v: i for i, v in enumerate(GENE_SHUFFLE_VARIANT_ORDER)}
        return sorted(uniq, key=lambda v: order.get(v, len(order) + 1))
    numeric = all(_numeric_sort_key(v) != float("inf") for v in uniq)
    if numeric:
        return sorted(uniq, key=_numeric_sort_key)
    return sorted(uniq)


def sweep_is_numeric(values: pd.Series, *, intervention_name: str | None = None) -> bool:
    if intervention_name == "gene_shuffle":
        return False
    uniq = [str(v) for v in values.dropna().unique()]
    if not uniq:
        return True
    return all(_numeric_sort_key(v) != float("inf") for v in uniq)


def sweep_x_positions(
    df: pd.DataFrame,
    x_col: str,
    *,
    intervention_name: str | None = None,
) -> tuple[np.ndarray, list[str], bool]:
    """Map param_value to x positions and tick labels for line plots."""
    if df.empty or x_col not in df.columns:
        return np.array([]), [], True
    param_key = None
    if "param_key" in df.columns and df["param_key"].notna().any():
        param_key = str(df["param_key"].dropna().iloc[0])
    labels = ordered_param_values(
        df[x_col], intervention_name=intervention_name, param_key=param_key
    )
    categorical = not sweep_is_numeric(df[x_col], intervention_name=intervention_name)
    if categorical:
        index = {lab: i for i, lab in enumerate(labels)}
        x = df[x_col].astype(str).map(index).to_numpy(dtype=float)
        return x, labels, True
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    return x, labels, False
