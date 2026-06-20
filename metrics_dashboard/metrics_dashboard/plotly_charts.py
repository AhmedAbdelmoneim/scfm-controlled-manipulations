"""Interactive Plotly charts (zoom / pan) for the metrics dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from metrics_dashboard.config import (
    MODEL_LABELS,
    MODEL_ORDER,
    SET3_COLLAPSE_YLABEL,
    SET3_SHIFT_YLABEL,
    model_palette,
)
from metrics_dashboard.plot_helpers import prepend_reference_points
from metrics_dashboard.style import streamlit_is_dark
from metrics_dashboard.sweep_axis import sweep_x_positions
from metrics_dashboard.transforms import Set1MainLayout, Set2RnxLayout, sort_models, std_bounds


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
    intervention_name: str | None = None,
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
        x, tick_labels, categorical = sweep_x_positions(
            mdf, x_col, intervention_name=intervention_name
        )
        order = np.argsort(x)
        x = x[order]
        y = mdf["value_mean"].astype(float).to_numpy()[order]
        mdf = mdf.iloc[order]
        if categorical and tick_labels:
            fig.update_xaxes(
                tickmode="array",
                tickvals=list(range(len(tick_labels))),
                ticktext=tick_labels,
                tickangle=-35,
                row=row,
                col=col,
            )
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


def plot_set1_main_metrics_plotly(
    layout: Set1MainLayout,
    models: list[str],
    *,
    scale: float = 1.0,
) -> go.Figure:
    metric_labels = layout.metric_labels
    manipulations = layout.manipulations
    nrows = max(1, len(metric_labels))
    ncols = max(1, len(manipulations))
    sub = layout.data
    x_col = layout.x_col
    row_gap = min(0.06 + 0.025 * scale, 0.14)
    if nrows > 1:
        row_gap = min(row_gap, 0.9 / (nrows - 1))
    col_gap = min(0.04 + 0.015 * scale, 0.09)
    if ncols > 1:
        col_gap = min(col_gap, 0.9 / (ncols - 1))
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[
            intervention if row_idx == 0 else ""
            for row_idx in range(nrows)
            for intervention in manipulations
        ],
        horizontal_spacing=col_gap,
        vertical_spacing=row_gap,
        shared_yaxes="rows",
    )
    palette = model_palette(models)
    legend_shown = False

    for ri, metric_label in enumerate(metric_labels, start=1):
        for ci, intervention in enumerate(manipulations, start=1):
            cell = sub[
                (sub["metric_label"] == metric_label)
                & (sub["intervention_name"] == intervention)
            ]
            if cell.empty:
                fig.update_xaxes(visible=False, row=ri, col=ci)
                fig.update_yaxes(visible=False, row=ri, col=ci)
                continue
            _add_sweep_traces(
                fig,
                cell,
                x_col=x_col,
                models=models,
                palette=palette,
                row=ri,
                col=ci,
                show_legend=not legend_shown,
                intervention_name=intervention,
            )
            legend_shown = True
            fig.update_yaxes(
                title_text=metric_label if ci == 1 else None,
                range=list(layout.y_ranges.get(metric_label, (0.0, 1.0))),
                row=ri,
                col=ci,
            )
            if ri == nrows:
                fig.update_xaxes(title_text=_x_axis_title(cell, x_col), row=ri, col=ci)
            else:
                fig.update_xaxes(showticklabels=False, row=ri, col=ci)

    cell_h = min(240 * scale, 380)
    cell_w = min(260 * scale, 420)
    fig.update_layout(
        template=_plotly_template(),
        title=dict(text="Set 1 — Main metrics", x=0.5, y=0.995),
        height=min(cell_h * nrows + 120, 5200),
        width=min(cell_w * ncols + 140, 3600),
        legend=dict(orientation="h", yanchor="top", y=-0.06, x=0.5, xanchor="center"),
        margin=dict(t=70, b=90, l=50, r=30),
    )
    # Extra headroom so per-row column titles (variant, dropout_rate, …) are not clipped.
    for ann in fig.layout.annotations:
        if ann.text:
            ann.update(y=ann.y + 0.015)
    return fig


def plot_set2_rnx_curves_plotly(
    layout: Set2RnxLayout,
    models: list[str],
    *,
    scale: float = 1.0,
) -> go.Figure:
    row_labels = layout.manipulations
    nrows = max(1, len(row_labels))
    ncols = max(1, max((len(v) for v in layout.param_values_by_row.values()), default=1))
    subplot_titles: list[str] = []
    for intervention in row_labels:
        col_values = layout.param_values_by_row.get(intervention, [])
        for ci in range(ncols):
            subplot_titles.append(
                f"{intervention}: {col_values[ci]}" if ci < len(col_values) else ""
            )
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=subplot_titles,
        horizontal_spacing=min(0.04 + 0.015 * scale, 0.09),
        vertical_spacing=min(
            min(0.07 + 0.02 * scale, 0.14),
            0.9 / (nrows - 1) if nrows > 1 else 0.0,
        ),
        shared_yaxes="all",
    )
    palette = model_palette(models)
    legend_shown = False

    for ri, intervention in enumerate(row_labels, start=1):
        col_values = layout.param_values_by_row.get(intervention, [])
        for ci in range(ncols):
            row, col = ri, ci + 1
            if ci >= len(col_values):
                fig.update_xaxes(visible=False, row=row, col=col)
                fig.update_yaxes(visible=False, row=row, col=col)
                continue
            param_value = str(col_values[ci])
            cell = layout.data[
                (layout.data["intervention_name"] == intervention)
                & (layout.data["param_value"].astype(str) == param_value)
            ]
            for model in MODEL_ORDER:
                if model not in models:
                    continue
                mdf = cell[cell["model"].astype(str) == model].sort_values("k")
                if mdf.empty:
                    continue
                color = palette.get(model, "#888888")
                label = _model_label(model)
                fig.add_trace(
                    go.Scatter(
                        x=mdf["k"],
                        y=mdf["rnx"],
                        mode="lines",
                        name=label,
                        line=dict(color=color, width=2),
                        legendgroup=label,
                        showlegend=not legend_shown,
                    ),
                    row=row,
                    col=col,
                )
            legend_shown = True
            fig.update_yaxes(
                title_text="R_NX" if ci == 0 else None,
                range=list(layout.y_range),
                row=row,
                col=col,
            )
            if ri == nrows:
                fig.update_xaxes(title_text="k", row=row, col=col)
            else:
                fig.update_xaxes(showticklabels=False, row=row, col=col)

    fig.update_layout(
        template=_plotly_template(),
        title=dict(text="Set 2 — R_NX curves", x=0.5, y=0.995),
        height=min(260 * scale * nrows + 120, 5200),
        width=min(260 * scale * ncols + 140, 3600),
        legend=dict(orientation="h", yanchor="top", y=-0.06, x=0.5, xanchor="center"),
        margin=dict(t=70, b=90, l=50, r=30),
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
    row_titles = [SET3_COLLAPSE_YLABEL, SET3_SHIFT_YLABEL]
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
            cell = prepend_reference_points(cell, ref_df, models)
            _add_sweep_traces(
                fig,
                cell,
                x_col="param_value",
                models=models,
                palette=palette,
                row=row_idx,
                col=ci + 1,
                show_legend=not legend_shown,
                intervention_name=intervention,
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
