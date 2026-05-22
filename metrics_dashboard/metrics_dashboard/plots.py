"""Matplotlib/seaborn figures for metrics exploration."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from metrics_dashboard.config import MODEL_ORDER

sns.set_theme(style="whitegrid", context="notebook")


def sort_plot_df(df: pd.DataFrame, x: str = "param_value") -> pd.DataFrame:
    out = df.copy()
    out["model"] = pd.Categorical(out["model"], categories=MODEL_ORDER, ordered=True)
    if x not in out.columns:
        return out
    if pd.api.types.is_numeric_dtype(out[x]):
        return out.sort_values(x)
    return out.sort_values(x, key=lambda s: s.astype(str))


def filter_metrics(
    metrics_df: pd.DataFrame,
    *,
    metric_category: str,
    metric_names: list[str] | None = None,
    spaces: list[str] | None = None,
    models: list[str] | None = None,
    interventions: list[str] | None = None,
    dropna: bool = True,
    y_col: str = "value_mean",
) -> pd.DataFrame:
    sub = metrics_df[metrics_df["metric_category"] == metric_category].copy()
    if metric_names is not None:
        sub = sub[sub["metric_name"].isin(metric_names)]
    if spaces is not None:
        sub = sub[sub["space"].isin(spaces)]
    if models is not None:
        sub = sub[sub["model"].astype(str).isin(models)]
    if interventions is not None:
        sub = sub[sub["intervention_name"].isin(interventions)]
    if dropna and y_col in sub.columns:
        sub = sub.dropna(subset=[y_col])
    return sub


def plot_metric_lines(
    df: pd.DataFrame,
    *,
    x: str = "param_value",
    y: str = "value_mean",
    hue: str = "model",
    col: str | None = "intervention_name",
    row: str | None = None,
    title: str | None = None,
    height: float = 3.2,
    aspect: float = 1.15,
) -> Figure:
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No plottable rows", ha="center", va="center")
        ax.set_axis_off()
        return fig

    plot_df = sort_plot_df(df, x=x)
    kwargs: dict = {
        "data": plot_df,
        "x": x,
        "y": y,
        "hue": hue,
        "kind": "line",
        "marker": "o",
        "linewidth": 2,
        "palette": "tab10",
        "height": height,
        "aspect": aspect,
        "facet_kws": {"sharey": False},
    }
    if col is not None:
        kwargs["col"] = col
    if row is not None:
        kwargs["row"] = row

    g = sns.relplot(**kwargs)
    xlabel = x
    if "param_key" in plot_df.columns and plot_df["param_key"].notna().any():
        xlabel = str(plot_df["param_key"].dropna().iloc[0])
    g.set(xlabel=xlabel)
    for ax in g.axes.flatten():
        ax.tick_params(axis="x", rotation=25)
    if title:
        g.fig.suptitle(title, y=1.03)
    g.fig.tight_layout()
    return g.fig


def plot_heatmap(
    df: pd.DataFrame,
    *,
    index: str,
    columns: str,
    values: str = "value_mean",
    title: str | None = None,
    figsize: tuple[float, float] = (14, 4),
) -> Figure:
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    pivot = df.pivot_table(
        index=index, columns=columns, values=values, aggfunc="first", observed=True
    )
    if index == "model":
        pivot = pivot.reindex([m for m in MODEL_ORDER if m in pivot.index])
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, ax=ax, cmap="viridis", annot=False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
