"""Matplotlib figures for the ScFMs metrics dashboard."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats

from metrics_dashboard.config import (
    DashboardMetric,
    MODEL_LABELS,
    MODEL_ORDER,
    model_palette,
)
from metrics_dashboard.style import apply_minimal_axes, configure_matplotlib, plot_colors
from metrics_dashboard.sweep_axis import sweep_x_positions
from metrics_dashboard.transforms import Set1GridLayout, sort_models, std_bounds


def _model_label(m: str) -> str:
    return MODEL_LABELS.get(m, m)


def _empty_fig(message: str = "No data", figsize: tuple[float, float] = (4, 3)) -> Figure:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, message, ha="center", va="center", color=plot_colors()["text"])
    ax.set_axis_off()
    return fig


def _plot_sweep_cell(
    ax: plt.Axes,
    cell_df: pd.DataFrame,
    *,
    x_col: str,
    y_label: str,
    palette: dict[str, str],
    show_xlabel: bool = True,
    intervention_name: str | None = None,
) -> None:
    if cell_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        apply_minimal_axes(ax)
        return

    plot_df = sort_models(cell_df)
    colors = plot_colors()
    tick_labels: list[str] = []
    categorical = False
    for model in MODEL_ORDER:
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
        color = palette.get(model, "#888888")
        band_lo = []
        band_hi = []
        for _, row in mdf.iterrows():
            lo, hi = std_bounds(row)
            band_lo.append(lo)
            band_hi.append(hi)
        ax.fill_between(x, band_lo, band_hi, color=color, alpha=colors["ci_alpha"], linewidth=0)
        ax.plot(x, y, marker="o", linewidth=2, label=_model_label(model), color=color)
        if "null_value" in mdf.columns and mdf["null_value"].notna().any():
            ax.plot(
                x,
                mdf["null_value"].astype(float).to_numpy(),
                linestyle="--",
                linewidth=1.2,
                color=color,
                alpha=0.55,
            )
    if categorical and tick_labels:
        ax.set_xticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8)

    if show_xlabel:
        if x_col == "diffusion_t":
            ax.set_xlabel("Diffusion time t")
        elif x_col == "k":
            ax.set_xlabel("k")
        elif x_col == "resolution":
            ax.set_xlabel("Leiden resolution")
        elif "param_key" in plot_df.columns and plot_df["param_key"].notna().any():
            ax.set_xlabel(str(plot_df["param_key"].dropna().iloc[0]))
        else:
            ax.set_xlabel(x_col)
    else:
        ax.tick_params(labelbottom=False)
    ax.set_ylabel(y_label)
    apply_minimal_axes(ax)


def _set1_column_title(cell_df: pd.DataFrame, col_val: str, column_facet: str) -> str:
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


def plot_set1_grid(
    layout: Set1GridLayout,
    spec: DashboardMetric,
    models: list[str],
    *,
    scale: float = 1.0,
) -> Figure:
    configure_matplotlib()
    row_labels = layout.row_labels
    nrows = max(1, len(row_labels))
    ncols = max(1, max((len(cols) for cols in layout.col_labels_by_row.values()), default=1))
    cell_w, cell_h = 5.2 * scale, 4.2 * scale
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(min(cell_w * ncols, 56), min(cell_h * nrows, 42)),
        squeeze=False,
        sharex=False,
        sharey=True,
        gridspec_kw={"hspace": 0.55, "wspace": 0.32},
    )
    palette = model_palette(models)
    sub = layout.data
    x_col = layout.x_col
    column_facet = layout.column_facet

    for ri, intervention in enumerate(row_labels):
        row_cols = layout.col_labels_by_row.get(intervention, ["all"])
        for ci in range(ncols):
            ax = axes[ri, ci]
            if ci >= len(row_cols):
                ax.set_axis_off()
                continue
            col_val = row_cols[ci]
            cell = sub[sub["intervention_name"] == intervention]
            if col_val != "all":
                cell = cell[cell[column_facet].astype(str) == str(col_val)]
            _plot_sweep_cell(
                ax,
                cell,
                x_col=x_col,
                y_label=spec.y_label,
                palette=palette,
                show_xlabel=(ri == nrows - 1),
            )
            col_title = _set1_column_title(cell, col_val, column_facet)
            if col_title:
                ax.set_title(col_title, fontsize=10, pad=10)
            if ci == 0:
                ax.set_ylabel(f"{intervention}\n{spec.y_label}", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(6, len(labels)),
            bbox_to_anchor=(0.5, -0.02),
        )
    fig.suptitle(spec.label, y=0.995, fontsize=12, color=plot_colors()["text"])
    fig.subplots_adjust(top=0.92, bottom=0.08)
    return fig


def _correlation_annotation(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return
    r, p = stats.pearsonr(x[mask], y[mask])
    ax.text(
        0.04,
        0.96,
        f"r = {r:.3f}\np = {p:.3g}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color=plot_colors()["text"],
    )


def plot_set2_correlation(
    wide: pd.DataFrame,
    *,
    y_col: str,
    y_label: str,
    x_label: str,
    connect_by_intervention: bool = True,
    models: list[str],
    scale: float = 1.0,
) -> Figure:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(min(12 * scale, 24), min(5 * scale, 12)))
    palette = model_palette(models)
    pairs = [
        ("metric_score", y_col, f"{x_label} vs {y_label}"),
    ]
    if y_col == "cell_type_score":
        pairs = [("metric_score", "cell_type_score", f"{x_label} vs cell-type ASW")]
    if y_col == "batch_score":
        pairs = [("metric_score", "batch_score", f"{x_label} vs batch iLISI")]

    plot_specs = [
        ("metric_score", "cell_type_score", "Cell-type ASW"),
        ("metric_score", "batch_score", "Batch iLISI"),
    ]
    for ax, (xc, yc, title) in zip(axes, plot_specs, strict=True):
        if xc not in wide.columns or yc not in wide.columns:
            ax.text(0.5, 0.5, f"No {yc} data", ha="center", va="center")
            apply_minimal_axes(ax)
            continue
        df = wide.dropna(subset=[xc, yc]).copy()
        if df.empty:
            ax.text(0.5, 0.5, "No plottable rows", ha="center", va="center")
            apply_minimal_axes(ax)
            continue
        for model in MODEL_ORDER:
            mdf = df[df["model"].astype(str) == model]
            if mdf.empty:
                continue
            color = palette.get(model, "#888888")
            ax.scatter(
                mdf[xc],
                mdf[yc],
                s=36,
                color=color,
                label=_model_label(model),
                alpha=0.85,
                edgecolors="none",
            )
            if connect_by_intervention:
                for _, grp in mdf.groupby("intervention_name", observed=True):
                    g = grp.sort_values(xc)
                    if len(g) > 1:
                        ax.plot(g[xc], g[yc], color=color, alpha=0.35, linewidth=1)
        _correlation_annotation(ax, df[xc].to_numpy(), df[yc].to_numpy())
        ax.set_xlabel(x_label)
        ax.set_ylabel(title.split(" vs ")[-1] if " vs " in title else title)
        ax.set_title(title, fontsize=10)
        apply_minimal_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(6, len(labels)),
            bbox_to_anchor=(0.5, -0.06),
        )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def _prepend_reference_points(
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


def plot_set3_row(
    collapse_df: pd.DataFrame,
    shift_df: pd.DataFrame,
    manipulations: list[str],
    models: list[str],
    *,
    collapse_label: str = "Within-cluster distance",
    shift_label: str = "Embedding shift (paired L2)",
    scale: float = 1.0,
) -> Figure:
    configure_matplotlib()
    ncols = max(1, len(manipulations))
    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(min(5.0 * scale * ncols, 56), min(8 * scale, 20)),
        squeeze=False,
        sharey="row",
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

    for ci, intervention in enumerate(manipulations):
        for row_idx, (df, ref_df, ylab) in enumerate(
            (
                (collapse_df, ref_collapse, collapse_label),
                (shift_df, ref_shift, shift_label),
            )
        ):
            ax = axes[row_idx, ci]
            cell = df[df["intervention_name"] == intervention]
            cell = _prepend_reference_points(cell, ref_df, models)
            _plot_sweep_cell(
                ax,
                cell,
                x_col="param_value",
                y_label=ylab,
                palette=palette,
                show_xlabel=(row_idx == 1),
                intervention_name=intervention,
            )
            if row_idx == 0:
                ax.set_title(intervention, fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(6, len(labels)),
            bbox_to_anchor=(0.5, -0.02),
        )
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    return fig
