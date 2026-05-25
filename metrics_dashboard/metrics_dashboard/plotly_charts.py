"""Interactive Plotly charts (zoom / pan) for the metrics dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from metrics_dashboard.config import (
    DashboardMetric,
    MODEL_LABELS,
    MODEL_ORDER,
    model_palette,
)
from metrics_dashboard.plots import _prepend_reference_points, _set1_column_title
from metrics_dashboard.style import streamlit_is_dark
from metrics_dashboard.transforms import Set1GridLayout, sort_models, std_bounds


def _plotly_template() -> str:
    return "plotly_dark" if streamlit_is_dark() else "plotly_white"


def _model_label(m: str) -> str:
    return MODEL_LABELS.get(m, m)


def _add_sweep_traces(
    fig: go.Figure,
    cell_df: pd.DataFrame,
    *,
    x_col: str,
    models: list[str],
    palette: dict[str, str],
    row: int,
    col: int,
    show_legend: bool,
) -> None:
    if cell_df.empty:
        return
    plot_df = sort_models(cell_df)
    for model in MODEL_ORDER:
        if model not in models:
            continue
        mdf = plot_df[plot_df["model"].astype(str) == model]
        if mdf.empty:
            continue
        mdf = mdf.sort_values(x_col, key=lambda s: pd.to_numeric(s, errors="coerce"))
        x = pd.to_numeric(mdf[x_col], errors="coerce")
        y = mdf["value_mean"].astype(float)
        color = palette.get(model, "#888888")
        label = _model_label(model)
        band_lo, band_hi = [], []
        for _, r in mdf.iterrows():
            lo, hi = std_bounds(r)
            band_lo.append(lo)
            band_hi.append(hi)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=band_hi,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=band_lo,
                fill="tonexty",
                fillcolor=_rgba(color, 0.22),
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                legendgroup=label,
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )
        if "null_value" in mdf.columns and mdf["null_value"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mdf["null_value"].astype(float),
                    mode="lines",
                    line=dict(color=color, width=1.2, dash="dash"),
                    opacity=0.55,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _x_axis_title(cell_df: pd.DataFrame, x_col: str) -> str:
    if x_col == "diffusion_t":
        return "Diffusion time t"
    if x_col == "k":
        return "k"
    if x_col == "resolution":
        return "Leiden resolution"
    if "param_key" in cell_df.columns and cell_df["param_key"].notna().any():
        return str(cell_df["param_key"].dropna().iloc[0])
    return x_col


def plot_set1_grid_plotly(
    layout: Set1GridLayout,
    spec: DashboardMetric,
    models: list[str],
    *,
    scale: float = 1.0,
) -> go.Figure:
    row_labels = layout.row_labels
    nrows = max(1, len(row_labels))
    ncols = max(1, max((len(cols) for cols in layout.col_labels_by_row.values()), default=1))
    sub = layout.data
    x_col = layout.x_col
    column_facet = layout.column_facet
    subplot_titles: list[str] = []
    for ri, intervention in enumerate(row_labels):
        row_cols = layout.col_labels_by_row.get(intervention, ["all"])
        for ci in range(ncols):
            if ci < len(row_cols):
                col_val = row_cols[ci]
                cell = layout.data[layout.data["intervention_name"] == intervention]
                if col_val != "all":
                    cell = cell[cell[column_facet].astype(str) == str(col_val)]
                subplot_titles.append(_set1_column_title(cell, col_val, layout.column_facet))
            else:
                subplot_titles.append("")
    row_gap = min(0.06 + 0.025 * scale, 0.14)
    col_gap = min(0.05 + 0.02 * scale, 0.12)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        horizontal_spacing=col_gap,
        vertical_spacing=row_gap,
        shared_yaxes="all",
    )
    palette = model_palette(models)
    legend_shown = False

    for ri, intervention in enumerate(row_labels):
        row_cols = layout.col_labels_by_row.get(intervention, ["all"])
        for ci in range(ncols):
            row, col = ri + 1, ci + 1
            if ci >= len(row_cols):
                fig.update_xaxes(visible=False, row=row, col=col)
                fig.update_yaxes(visible=False, row=row, col=col)
                continue
            col_val = row_cols[ci]
            cell = sub[sub["intervention_name"] == intervention]
            if col_val != "all":
                cell = cell[cell[column_facet].astype(str) == str(col_val)]
            _add_sweep_traces(
                fig,
                cell,
                x_col=x_col,
                models=models,
                palette=palette,
                row=row,
                col=col,
                show_legend=not legend_shown,
            )
            legend_shown = True
            if ri == nrows - 1:
                fig.update_xaxes(title_text=_x_axis_title(cell, x_col), row=row, col=col)
            else:
                fig.update_xaxes(showticklabels=False, row=row, col=col)
            if ci == 0:
                fig.update_yaxes(title_text=f"{intervention}", row=row, col=col)

    cell_h = min(340 * scale, 540)
    cell_w = min(280 * scale, 440)
    fig.update_layout(
        template=_plotly_template(),
        title=dict(text=spec.label, x=0.5, y=0.995),
        height=min(cell_h * nrows + 120, 3400),
        width=min(cell_w * ncols + 140, 3400),
        legend=dict(orientation="h", yanchor="top", y=-0.06, x=0.5, xanchor="center"),
        margin=dict(t=70, b=90, l=50, r=30),
    )
    # Extra headroom so per-row column titles (variant, dropout_rate, …) are not clipped.
    for ann in fig.layout.annotations:
        if ann.text:
            ann.update(y=ann.y + 0.015)
    return fig


def plot_set2_correlation_plotly(
    wide: pd.DataFrame,
    *,
    x_label: str,
    models: list[str],
    scale: float = 1.0,
) -> go.Figure:
    palette = model_palette(models)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Cell-type ASW", "Batch iLISI"])
    pairs = [
        ("metric_score", "cell_type_score"),
        ("metric_score", "batch_score"),
    ]
    legend_shown = False
    for col_idx, (xc, yc) in enumerate(pairs, start=1):
        if xc not in wide.columns or yc not in wide.columns:
            continue
        df = wide.dropna(subset=[xc, yc])
        for model in MODEL_ORDER:
            if model not in models:
                continue
            mdf = df[df["model"].astype(str) == model]
            if mdf.empty:
                continue
            color = palette.get(model, "#888888")
            label = _model_label(model)
            fig.add_trace(
                go.Scatter(
                    x=mdf[xc],
                    y=mdf[yc],
                    mode="markers",
                    name=label,
                    marker=dict(size=9, color=color),
                    legendgroup=label,
                    showlegend=not legend_shown,
                ),
                row=1,
                col=col_idx,
            )
            for _, grp in mdf.groupby("intervention_name", observed=True):
                g = grp.sort_values(xc)
                if len(g) > 1:
                    fig.add_trace(
                        go.Scatter(
                            x=g[xc],
                            y=g[yc],
                            mode="lines",
                            line=dict(color=color, width=1),
                            opacity=0.35,
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=1,
                        col=col_idx,
                    )
        legend_shown = True
        mask = np.isfinite(df[xc]) & np.isfinite(df[yc])
        if mask.sum() >= 3:
            from scipy import stats

            r, p = stats.pearsonr(df.loc[mask, xc], df.loc[mask, yc])
            fig.add_annotation(
                text=f"r = {r:.3f}<br>p = {p:.3g}",
                xref="x domain",
                yref="y domain",
                x=0.04,
                y=0.96,
                showarrow=False,
                row=1,
                col=col_idx,
            )
        fig.update_xaxes(title_text=x_label, row=1, col=col_idx)
    fig.update_layout(
        template=_plotly_template(),
        height=min(420 * scale, 900),
        width=min(900 * scale, 1600),
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0.5, xanchor="center"),
        margin=dict(b=80),
    )
    return fig


def plot_set3_row_plotly(
    collapse_df: pd.DataFrame,
    shift_df: pd.DataFrame,
    manipulations: list[str],
    models: list[str],
    *,
    scale: float = 1.0,
) -> go.Figure:
    ncols = max(1, len(manipulations))
    fig = make_subplots(
        rows=2,
        cols=ncols,
        subplot_titles=list(manipulations) + [""] * ncols,
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
        shared_yaxes="rows",
    )
    palette = model_palette(models)
    ref_collapse = collapse_df[
        collapse_df["intervention_name"].isin({"reference"})
        | (collapse_df["param_key"] == "reference")
    ]
    ref_shift = shift_df[
        shift_df["intervention_name"].isin({"reference"})
        | (shift_df["param_key"] == "reference")
    ]
    row_titles = ["Within-cluster distance", "Embedding shift (paired L2)"]
    legend_shown = False

    for ci, intervention in enumerate(manipulations):
        for row_idx, (df, ref_df, ylab) in enumerate(
            (
                (collapse_df, ref_collapse, row_titles[0]),
                (shift_df, ref_shift, row_titles[1]),
            ),
            start=1,
        ):
            cell = df[df["intervention_name"] == intervention]
            cell = _prepend_reference_points(cell, ref_df, models)
            _add_sweep_traces(
                fig,
                cell,
                x_col="param_value",
                models=models,
                palette=palette,
                row=row_idx,
                col=ci + 1,
                show_legend=not legend_shown,
            )
            legend_shown = True
            if ci == 0:
                fig.update_yaxes(title_text=ylab, row=row_idx, col=ci + 1)
            if row_idx == 2:
                fig.update_xaxes(title_text="Manipulation parameter", row=row_idx, col=ci + 1)
            else:
                fig.update_xaxes(showticklabels=False, row=row_idx, col=ci + 1)

    fig.update_layout(
        template=_plotly_template(),
        height=min(340 * scale * 2 + 60, 2000),
        width=min(280 * scale * ncols + 100, 3200),
        legend=dict(orientation="h", yanchor="top", y=-0.06, x=0.5, xanchor="center"),
        margin=dict(b=90),
    )
    return fig
