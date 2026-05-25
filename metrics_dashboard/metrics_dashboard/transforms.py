"""Plot-ready transforms for dashboard plot sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from metrics_dashboard.config import (
    DASHBOARD_METRICS,
    DashboardMetric,
    MANIPULATION_ORDER,
    MODEL_ORDER,
    REFERENCE_INTERVENTION_NAMES,
    SET3_CATEGORY,
    SET3_COLLAPSE_METRIC,
    SET3_SHIFT_METRIC,
    SET3_SPACE,
)


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
    x-axis (all ``diffusion_t`` for KL/JS, all ``k`` for kNN recall) and pin the other
    hyperparameter. When True (Set 2), pin both to a single value.
    """
    sub = metrics_df[
        (metrics_df["metric_category"] == spec.metric_category)
        & (metrics_df["metric_name"] == spec.metric_name)
        & (metrics_df["space"] == spec.space)
        & (metrics_df["model"].astype(str).isin(models))
    ].copy()
    pin_k = True
    if not pin_hyperparameters and spec.metric_name == "knn_recall":
        pin_k = False
    if pin_k and spec.default_k is not None and "k" in sub.columns:
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
    if (
        spec.metric_category == "knn_metrics"
        and spec.metric_name.startswith("diffusion")
        and "diffusion_t" in sub.columns
        and sub["diffusion_t"].notna().any()
    ):
        return "diffusion_t"
    if (
        spec.metric_name == "knn_recall"
        and "k" in sub.columns
        and sub["k"].notna().any()
    ):
        return "k"
    if spec.x_col == "resolution" and "resolution" in sub.columns:
        return "resolution"
    return "param_value"


@dataclass(frozen=True)
class Set1GridLayout:
    data: pd.DataFrame
    row_labels: list[str]
    col_labels_by_row: dict[str, list[str]]
    x_col: str
    column_facet: str = "param_value"


def prepare_set1_grid(
    metrics_df: pd.DataFrame,
    spec: DashboardMetric,
    models: list[str],
) -> Set1GridLayout:
    """Build Set 1 layout: rows = manipulation, columns = config, x = sweep hyperparameter."""
    sub = filter_for_dashboard_metric(metrics_df, spec, models, pin_hyperparameters=False)
    sub = sub[~sub["intervention_name"].isin(REFERENCE_INTERVENTION_NAMES)]
    interventions = [i for i in MANIPULATION_ORDER if i in sub["intervention_name"].unique()]
    extras = sorted(set(sub["intervention_name"].unique()) - set(interventions))
    row_labels = interventions + extras

    x_col = _set1_x_col(spec, sub)
    column_facet = "param_value"
    col_labels_by_row: dict[str, list[str]] = {}

    if x_col == column_facet:
        # e.g. kNN recall: sweep param_value on x-axis, no per-config columns
        col_labels_by_row = {name: ["all"] for name in row_labels}
    else:
        for name in row_labels:
            part = sub[sub["intervention_name"] == name]
            col_labels_by_row[name] = _sort_numeric_str_labels(part[column_facet].unique())

    return Set1GridLayout(
        data=sub,
        row_labels=row_labels,
        col_labels_by_row=col_labels_by_row,
        x_col=x_col,
        column_facet=column_facet,
    )


def prepare_set2_correlation(
    metrics_df: pd.DataFrame,
    spec: DashboardMetric,
    models: list[str],
) -> pd.DataFrame:
    """Wide table: cell_type_asw, batch_ilisi, selected metric mean per run."""
    metric_sub = filter_for_dashboard_metric(metrics_df, spec, models)
    metric_sub = metric_sub[~metric_sub["intervention_name"].isin(REFERENCE_INTERVENTION_NAMES)]
    if spec.x_col == "resolution":
        # one point per resolution — pick default or mean across resolutions
        if spec.default_resolution is not None and "resolution" in metric_sub.columns:
            metric_sub = metric_sub[metric_sub["resolution"] == spec.default_resolution]
        else:
            metric_sub = (
                metric_sub.groupby(
                    ["model", "intervention_id", "intervention_name", "param_value"],
                    observed=True,
                )["value_mean"]
                .mean()
                .reset_index()
            )

    keys = ["model", "intervention_id", "intervention_name", "param_value"]
    if "param_key" in metric_sub.columns:
        keys.append("param_key")
    metric_wide = metric_sub.groupby(keys, observed=True)["value_mean"].mean().reset_index()
    metric_wide = metric_wide.rename(columns={"value_mean": "metric_score"})

    cb = metrics_df[
        (metrics_df["metric_category"] == "cell_type_and_batch_metrics")
        & (metrics_df["space"] == "embedding_manipulated")
        & (metrics_df["model"].astype(str).isin(models))
    ]
    cb = cb[~cb["intervention_name"].isin(REFERENCE_INTERVENTION_NAMES)]

    def _pivot_score(metric_name: str, col: str) -> pd.DataFrame:
        part = cb[cb["metric_name"] == metric_name]
        if part.empty:
            return pd.DataFrame(columns=keys + [col])
        return (
            part.groupby(keys, observed=True)["value_mean"]
            .mean()
            .reset_index()
            .rename(columns={"value_mean": col})
        )

    cell = _pivot_score("cell_type_asw", "cell_type_score")
    batch = _pivot_score("batch_ilisi", "batch_score")
    wide = metric_wide
    for extra in (cell, batch):
        if not extra.empty:
            wide = wide.merge(extra, on=keys, how="left")
    return wide


def prepare_set3_embedding(
    metrics_df: pd.DataFrame,
    models: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse (within_man) and shift (paired_cell); reference rows kept for plot prepending."""
    sub = metrics_df[
        (metrics_df["metric_category"] == SET3_CATEGORY)
        & (metrics_df["space"] == SET3_SPACE)
        & (metrics_df["model"].astype(str).isin(models))
    ].copy()

    collapse = sub[sub["metric_name"] == SET3_COLLAPSE_METRIC].copy()
    shift = sub[sub["metric_name"] == SET3_SHIFT_METRIC].copy()
    ref_within = sub[sub["metric_name"] == "within_ref_pairwise_l2"].copy()

    if not ref_within.empty:
        ref_c = ref_within.copy()
        ref_c["metric_name"] = SET3_COLLAPSE_METRIC
        ref_c["param_value"] = 0.0
        ref_c["param_key"] = "reference"
        ref_c["intervention_name"] = "reference"
        collapse = pd.concat([ref_c, collapse], ignore_index=True)

        ref_s = ref_within.copy()
        ref_s["metric_name"] = SET3_SHIFT_METRIC
        ref_s["param_value"] = 0.0
        ref_s["param_key"] = "reference"
        ref_s["intervention_name"] = "reference"
        shift = pd.concat([ref_s, shift], ignore_index=True)

    return collapse, shift


def sort_models(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model"] = pd.Categorical(
        out["model"].astype(str), categories=MODEL_ORDER, ordered=True
    )
    return out.sort_values("model")
