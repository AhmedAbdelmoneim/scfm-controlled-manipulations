"""Sidebar controls for the metrics dashboard."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from metrics_dashboard.catalog import discover_models
from metrics_dashboard.config import DASHBOARD_METRIC_KEYS, DASHBOARD_METRICS, MODEL_ORDER, bundle_root
from metrics_dashboard.style import render_theme_sidebar


@dataclass
class DashboardControls:
    dataset_ids: list[str]
    models: list[str]
    metric_key: str
    plot_scale: float
    interactive_plots: bool


def render_sidebar_controls(datasets: list[str]) -> DashboardControls | None:
    render_theme_sidebar()

    if not datasets:
        st.sidebar.error("No bundles in data/dashboard_bundles. Run make export-dashboard-bundle.")
        return None

    selected_datasets = st.sidebar.multiselect(
        "Dataset(s)",
        datasets,
        default=datasets[:1],
        help="Select one or more datasets (metrics are averaged when multiple).",
    )
    if not selected_datasets:
        return None

    root = bundle_root()
    all_models: list[str] = []
    for ds in selected_datasets:
        all_models.extend(discover_models(ds, root))
    all_models = sorted(set(all_models), key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)

    models = st.sidebar.multiselect("Models", all_models, default=all_models)
    if not models:
        return None

    metric_key = st.sidebar.selectbox(
        "Metric",
        DASHBOARD_METRIC_KEYS,
        format_func=lambda k: DASHBOARD_METRICS[k].label,
    )

    st.sidebar.markdown("**Plots**")
    interactive_plots = st.sidebar.checkbox(
        "Interactive (zoom & pan)",
        value=True,
        help="Plotly toolbar: zoom, pan, reset axes, download PNG.",
    )
    plot_scale = st.sidebar.slider(
        "Plot size",
        min_value=0.75,
        max_value=2.0,
        value=1.25,
        step=0.05,
        help="Scales subplot size (scroll the page for large grids).",
    )

    return DashboardControls(
        dataset_ids=selected_datasets,
        models=models,
        metric_key=metric_key,
        plot_scale=plot_scale,
        interactive_plots=interactive_plots,
    )
