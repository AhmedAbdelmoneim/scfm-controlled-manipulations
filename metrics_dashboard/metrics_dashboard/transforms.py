"""Plot-ready transforms for dashboard plot sets."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from metrics_dashboard.config import (
    DashboardMetric,
    MAIN_METRICS,
    MANIPULATION_ORDER,
    MODEL_ORDER,
    REFERENCE_INTERVENTION_NAMES,
    SET3_CATEGORY,
    SET3_COLLAPSE_METRIC,
    SET3_SHIFT_METRIC,
    SET3_SPACE,
)
from metrics_dashboard.sweep_axis import ordered_param_values


def std_bounds(row: pd.Series) -> tuple[float, float]:
    """Shaded band: mean ± one standard deviation across cells."""
    mean = float(row["value_mean"])
    if "value_std" not in row.index or pd.isna(row["value_std"]):
        return mean, mean
    std = float(row["value_std"])
    if std <= 0 or np.isnan(std):
        return mean, mean
    return mean - std, mean + std


def average_metrics_across_datasets(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Mean-aggregate rows that share the same keys except dataset_id."""
    if metrics_df.empty or metrics_df["dataset_id"].nunique() <= 1:
        return metrics_df

    group_cols = [
        c
        for c in metrics_df.columns
        if c
        not in (
            "dataset_id",
            "value_mean",
            "value_median",
            "value_std",
            "value_min",
            "value_max",
            "value_q05",
            "value_q25",
            "value_q75",
            "value_q95",
            "null_value",
        )
        and not c.startswith("Unnamed")
    ]
    numeric_cols = [
        c
        for c in (
            "value_mean",
            "value_median",
            "value_std",
            "value_min",
            "value_max",
            "value_q05",
            "value_q25",
            "value_q75",
            "value_q95",
            "null_value",
        )
        if c in metrics_df.columns
    ]
    keys = [c for c in group_cols if c in metrics_df.columns]
    out = (
        metrics_df.groupby(keys, dropna=False, observed=True)[numeric_cols]
        .mean()
        .reset_index()
    )
    out["dataset_id"] = "averaged"
    return out


def filter_for_dashboard_metric(
    metrics_df: pd.DataFrame,
    spec: DashboardMetric,
    models: list[str],
    *,
    pin_hyperparameters: bool = True,
) -> pd.DataFrame:
    """Filter to one dashboard metric and model list.

    When ``pin_hyperparameters`` is False (Set 1 sweeps), keep the sweep dimension on the
    x-axis and pin other hyperparameters. When True (Set 2), pin hyperparameters to defaults.
    """
    sub = metrics_df[
        (metrics_df["metric_category"] == spec.metric_category)
        & (metrics_df["metric_name"] == spec.metric_name)
        & (metrics_df["space"] == spec.space)
        & (metrics_df["model"].astype(str).isin(models))
    ].copy()
    if spec.default_k is not None and "k" in sub.columns:
        k_vals = sub["k"].dropna().unique()
        if len(k_vals):
            target = spec.default_k
            if target not in k_vals:
                target = float(sorted(k_vals)[0])
            sub = sub[sub["k"] == target]
    if pin_hyperparameters and spec.default_diffusion_t is not None and "diffusion_t" in sub.columns:
        t_vals = sub["diffusion_t"].dropna().unique()
        if len(t_vals):
            target = spec.default_diffusion_t
            if target not in t_vals:
                target = float(sorted(t_vals)[0])
            sub = sub[sub["diffusion_t"] == target]
    if spec.default_resolution is not None and "resolution" in sub.columns:
        res_vals = sub["resolution"].dropna().unique()
        if len(res_vals) and spec.x_col == "resolution":
            pass  # keep full resolution sweep for clustering
    return sub


def _sort_numeric_str_labels(values) -> list[str]:
    series = pd.Series(values).dropna().astype(str)
    if series.empty:
        return []
    order = sorted(series.unique(), key=lambda v: float(pd.to_numeric(v, errors="coerce")))
    return [str(v) for v in order]


def _set1_x_col(spec: DashboardMetric, sub: pd.DataFrame) -> str:
    """Sweep dimension plotted on the x-axis within each Set 1 cell."""
    if spec.x_col == "resolution" and "resolution" in sub.columns:
        return "resolution"
    return "param_value"


@dataclass(frozen=True)
class Set1MainLayout:
    data: pd.DataFrame
    metric_labels: list[str]
    manipulations: list[str]
    x_col: str
    y_ranges: dict[str, tuple[float, float]]


def _pin_default_resolution(sub: pd.DataFrame, spec: DashboardMetric) -> pd.DataFrame:
    if sub.empty or spec.default_resolution is None or "resolution" not in sub.columns:
        return sub
    res_vals = sub["resolution"].dropna().unique()
    if not len(res_vals):
        return sub
    target = spec.default_resolution
    if target not in res_vals:
        target = float(sorted(res_vals)[0])
    return sub[sub["resolution"] == target]


def _main_metric_rows(metrics_df: pd.DataFrame, spec: DashboardMetric, models: list[str]) -> pd.DataFrame:
    sub = filter_for_dashboard_metric(metrics_df, spec, models, pin_hyperparameters=True)
    sub = _pin_default_resolution(sub, spec)
    sub = sub[~sub["intervention_name"].isin(REFERENCE_INTERVENTION_NAMES)].copy()
    if sub.empty:
        return sub
    sub["metric_key"] = spec.key
    sub["metric_label"] = spec.label
    if spec.key == "clustering_ari":
        sub["metric_y_min"] = -1.0
        sub["metric_y_max"] = 1.0
    else:
        sub["metric_y_min"] = 0.0
        sub["metric_y_max"] = 1.0
    return sub


def prepare_set1_main_metrics(metrics_df: pd.DataFrame, models: list[str]) -> Set1MainLayout:
    """Build Set 1: fixed main metrics by manipulation strength and model."""
    frames = [
        _main_metric_rows(metrics_df, spec, models)
        for spec in MAIN_METRICS
    ]
    frames = [frame for frame in frames if not frame.empty]
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if data.empty:
        return Set1MainLayout(data, [], [], "param_value", {})

    metric_labels = [
        spec.label
        for spec in MAIN_METRICS
        if spec.label in set(data["metric_label"].dropna().astype(str))
    ]
    interventions = [
        i for i in MANIPULATION_ORDER if i in set(data["intervention_name"].dropna().astype(str))
    ]
    extras = sorted(set(data["intervention_name"].dropna().astype(str)) - set(interventions))
    y_ranges = {
        label: (
            float(part["metric_y_min"].iloc[0]),
            float(part["metric_y_max"].iloc[0]),
        )
        for label, part in data.groupby("metric_label", observed=True)
    }
    return Set1MainLayout(data, metric_labels, interventions + extras, "param_value", y_ranges)


@dataclass(frozen=True)
class Set2RnxLayout:
    data: pd.DataFrame
    manipulations: list[str]
    param_values_by_row: dict[str, list[str]]
    y_range: tuple[float, float] = (-1.0, 1.0)


def prepare_set2_rnx_curves(metrics_df: pd.DataFrame, models: list[str]) -> Set2RnxLayout:
    """Parse stored R_NX curve JSON into long rows for curve-grid plotting."""
    required = {"metric_name", "rnx_curve_json", "model", "intervention_name", "param_value"}
    if metrics_df.empty or not required.issubset(metrics_df.columns):
        return Set2RnxLayout(pd.DataFrame(), [], {})
    sub = metrics_df[
        (metrics_df["metric_category"] == "structure_metrics")
        & (metrics_df["metric_name"] == "rnx_curve")
        & (metrics_df["space"] == "embedding")
        & (metrics_df["model"].astype(str).isin(models))
        & (~metrics_df["intervention_name"].isin(REFERENCE_INTERVENTION_NAMES))
    ].copy()
    rows: list[dict] = []
    for _, row in sub.iterrows():
        payload_text = row.get("rnx_curve_json")
        if pd.isna(payload_text):
            continue
        try:
            payload = json.loads(str(payload_text))
        except json.JSONDecodeError:
            continue
        k_values = payload.get("k", [])
        rnx_values = payload.get("rnx", [])
        for k, rnx in zip(k_values, rnx_values, strict=False):
            rows.append(
                {
                    "dataset_id": row.get("dataset_id"),
                    "model": row.get("model"),
                    "intervention_id": row.get("intervention_id"),
                    "intervention_name": row.get("intervention_name"),
                    "param_key": row.get("param_key"),
                    "param_value": row.get("param_value"),
                    "k": int(k),
                    "rnx": float(rnx),
                }
            )
    data = pd.DataFrame(rows)
    if data.empty:
        return Set2RnxLayout(data, [], {})
    group_cols = ["model", "intervention_name", "param_key", "param_value", "k"]
    data = (
        data.groupby(group_cols, dropna=False, observed=True)["rnx"]
        .mean()
        .reset_index()
    )
    interventions = [
        i for i in MANIPULATION_ORDER if i in set(data["intervention_name"].dropna().astype(str))
    ]
    extras = sorted(set(data["intervention_name"].dropna().astype(str)) - set(interventions))
    param_values_by_row = {
        intervention: ordered_param_values(
            data[data["intervention_name"] == intervention]["param_value"],
            intervention_name=intervention,
        )
        for intervention in interventions + extras
    }
    return Set2RnxLayout(data, interventions + extras, param_values_by_row)


def _reference_baseline_per_model(ref_within: pd.DataFrame) -> pd.DataFrame:
    """``within_ref_pairwise_l2`` is identical across interventions; keep one row per model."""
    if ref_within.empty:
        return ref_within
    return (
        ref_within.groupby("model", observed=True)
        .first()
        .reset_index()
    )


def _ref_within_scale_by_model(ref_within: pd.DataFrame) -> pd.Series:
    """Per-model reference within-cluster mean (denominator for Set 3 normalization)."""
    if ref_within.empty:
        return pd.Series(dtype=float)
    return ref_within.groupby("model", observed=True)["value_mean"].first()


def _normalize_embedding_shift_values(
    df: pd.DataFrame,
    ref_scale: pd.Series,
) -> pd.DataFrame:
    """Divide metric values by reference within-cluster distance for cross-model comparability."""
    if df.empty or ref_scale.empty:
        return df
    out = df.copy()
    denom = out["model"].astype(str).map(ref_scale).astype(float)
    denom = denom.replace(0, np.nan)
    for col in ("value_mean", "value_std", "value_median"):
        if col in out.columns:
            out[col] = out[col].astype(float) / denom
    return out


def prepare_set3_embedding(
    metrics_df: pd.DataFrame,
    models: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse within-man and shift (paired_cell / ref within-cluster distance)."""
    sub = metrics_df[
        (metrics_df["metric_category"] == SET3_CATEGORY)
        & (metrics_df["space"] == SET3_SPACE)
        & (metrics_df["model"].astype(str).isin(models))
    ].copy()

    ref_within = sub[sub["metric_name"] == "within_ref_pairwise_l2"].copy()
    ref_scale = _ref_within_scale_by_model(ref_within)

    collapse = sub[sub["metric_name"] == SET3_COLLAPSE_METRIC].copy()
    shift = _normalize_embedding_shift_values(
        sub[sub["metric_name"] == SET3_SHIFT_METRIC].copy(),
        ref_scale,
    )

    ref_base = _reference_baseline_per_model(ref_within)
    if not ref_base.empty:
        ref_c = ref_base.copy()
        ref_c["metric_name"] = SET3_COLLAPSE_METRIC
        ref_c["param_value"] = 0.0
        ref_c["param_key"] = "reference"
        ref_c["intervention_name"] = "reference"
        collapse = pd.concat([ref_c, collapse], ignore_index=True)
        # Shift row: no reference point (paired L2 at identity is always 0 and clutters the plot).

    return collapse, shift


def sort_models(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model"] = pd.Categorical(
        out["model"].astype(str), categories=MODEL_ORDER, ordered=True
    )
    return out.sort_values("model")
