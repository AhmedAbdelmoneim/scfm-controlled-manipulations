"""Render metric category plot sections (mirrors marimo notebook)."""

from __future__ import annotations

import streamlit as st

from metrics_dashboard.filters import ExploreFilters
from metrics_dashboard.plots import filter_metrics, plot_heatmap, plot_metric_lines


def _show_fig(fig) -> None:
    st.pyplot(fig, clear_figure=True)


def render_focused_plot(metrics_df, flt: ExploreFilters) -> None:
    """Single plot from sidebar selection."""
    sub = filter_metrics(
        metrics_df,
        metric_category=flt.metric_category,
        metric_names=[flt.metric_name],
        spaces=[flt.space],
        models=flt.models,
        interventions=flt.interventions,
        y_col=flt.y_col,
    )
    if flt.k is not None and "k" in sub.columns:
        sub = sub[sub["k"] == flt.k]
    if flt.diffusion_t is not None and "diffusion_t" in sub.columns:
        sub = sub[sub["diffusion_t"] == flt.diffusion_t]
    if flt.resolution is not None and "resolution" in sub.columns:
        sub = sub[sub["resolution"] == flt.resolution]

    x = "param_value"
    if flt.metric_category == "clustering_metrics":
        x = "resolution"

    title = f"{flt.metric_category} / {flt.metric_name} / {flt.space}"
    _show_fig(
        plot_metric_lines(
            sub,
            x=x,
            y=flt.y_col,
            col="intervention_name",
            title=title,
        )
    )


def render_all_sections(metrics_df, flt: ExploreFilters) -> None:
    """Full notebook-style sections in expanders."""
    y = flt.y_col
    models = flt.models
    interventions = flt.interventions

    with st.expander("Embedding stats", expanded=False):
        stats_names = [
            "mean_row_l2_norm_ref",
            "mean_row_l2_norm_man",
            "col_mean_ref",
            "col_mean_man",
            "col_variance_ref",
            "col_variance_man",
        ]
        for space in ["raw", "embedding"]:
            sub = filter_metrics(
                metrics_df,
                metric_category="embedding_stats",
                metric_names=stats_names,
                spaces=[space],
                models=models,
                interventions=interventions,
                y_col=y,
            )
            _show_fig(
                plot_metric_lines(
                    sub,
                    y=y,
                    row="metric_name",
                    col="intervention_name",
                    title=f"embedding_stats — space={space}",
                )
            )

    with st.expander("Embedding shift", expanded=False):
        shift_names = [
            "paired_cell_l2_norm",
            "shift_pairwise_cosine",
            "within_ref_pairwise_l2",
            "within_man_pairwise_l2",
        ]
        for space in ["raw", "embedding"]:
            sub = filter_metrics(
                metrics_df,
                metric_category="embedding_shift",
                metric_names=shift_names,
                spaces=[space],
                models=models,
                interventions=interventions,
                y_col=y,
            )
            _show_fig(
                plot_metric_lines(
                    sub,
                    y=y,
                    row="metric_name",
                    col="intervention_name",
                    title=f"embedding_shift — space={space}",
                )
            )

    with st.expander("Embedding shift gain", expanded=False):
        gain_names = [
            "paired_cell_l2_norm",
            "shift_pairwise_cosine",
            "within_ref_pairwise_l2",
            "within_man_pairwise_l2",
        ]
        sub = filter_metrics(
            metrics_df,
            metric_category="embedding_shift_gain",
            metric_names=gain_names,
            spaces=["embedding_minus_raw"],
            models=models,
            interventions=interventions,
            y_col=y,
        )
        _show_fig(
            plot_metric_lines(
                sub,
                y=y,
                row="metric_name",
                col="intervention_name",
                title="embedding_shift_gain",
            )
        )

    with st.expander("KNN metrics", expanded=False):
        for space in ["raw", "embedding"]:
            for k in sorted(metrics_df["k"].dropna().unique()):
                sub = filter_metrics(
                    metrics_df,
                    metric_category="knn_metrics",
                    metric_names=["knn_recall"],
                    spaces=[space],
                    models=models,
                    interventions=interventions,
                    y_col=y,
                )
                sub = sub[sub["k"] == k]
                _show_fig(
                    plot_metric_lines(
                        sub,
                        y=y,
                        col="intervention_name",
                        title=f"knn_recall — space={space}, k={int(k)}",
                    )
                )
        for metric in ["diffusion_js", "diffusion_sym_kl"]:
            for space in ["raw", "embedding"]:
                for t in sorted(metrics_df["diffusion_t"].dropna().unique()):
                    sub = filter_metrics(
                        metrics_df,
                        metric_category="knn_metrics",
                        metric_names=[metric],
                        spaces=[space],
                        models=models,
                        interventions=interventions,
                        y_col=y,
                    )
                    sub = sub[sub["diffusion_t"] == t]
                    _show_fig(
                        plot_metric_lines(
                            sub,
                            y=y,
                            col="intervention_name",
                            title=f"{metric} — space={space}, diffusion_t={int(t)}",
                        )
                    )
        sub = filter_metrics(
            metrics_df,
            metric_category="knn_metrics",
            metric_names=["knn_recall"],
            spaces=["embedding"],
            models=models,
            interventions=interventions,
            y_col=y,
        )
        if "k" in sub.columns:
            sub = sub[sub["k"] == 15]
        _show_fig(
            plot_heatmap(
                sub,
                index="model",
                columns="intervention_id",
                values=y,
                title="knn_recall (embedding, k=15)",
            )
        )

    with st.expander("KNN metrics gain", expanded=False):
        for k in sorted(metrics_df["k"].dropna().unique()):
            sub = filter_metrics(
                metrics_df,
                metric_category="knn_metrics_gain",
                metric_names=["knn_recall"],
                spaces=["embedding_minus_raw"],
                models=models,
                interventions=interventions,
                y_col=y,
            )
            sub = sub[sub["k"] == k]
            _show_fig(
                plot_metric_lines(
                    sub,
                    y=y,
                    col="intervention_name",
                    title=f"knn_recall gain — k={int(k)}",
                )
            )

    with st.expander("Clustering metrics", expanded=False):
        sub = filter_metrics(
            metrics_df,
            metric_category="clustering_metrics",
            metric_names=["leiden_ari"],
            spaces=["embedding"],
            models=models,
            interventions=interventions,
            y_col=y,
        )
        _show_fig(
            plot_metric_lines(
                sub,
                x="resolution",
                y=y,
                col="intervention_name",
                title="leiden_ari vs resolution (embedding)",
            )
        )

    with st.expander("Cell type and batch metrics", expanded=False):
        batch = metrics_df[
            metrics_df["metric_category"] == "cell_type_and_batch_metrics"
        ]
        n_null = batch[flt.y_col].isna().sum() if flt.y_col in batch.columns else len(batch)
        if n_null == len(batch) and len(batch) > 0:
            st.warning(
                f"All {len(batch)} rows have null values — likely missing batch/cell_type columns in obs."
            )
        st.dataframe(
            batch[
                ["model", "intervention_name", "space", flt.y_col, "null_value"]
            ].drop_duplicates(),
            use_container_width=True,
        )
