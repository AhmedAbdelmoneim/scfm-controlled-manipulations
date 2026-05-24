"""Sidebar controls for the metrics dashboard."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from metrics_dashboard.catalog import discover_models
from metrics_dashboard.config import DASHBOARD_METRIC_KEYS, DASHBOARD_METRICS, MODEL_ORDER, bundle_root


@dataclass
class DashboardControls:
    dataset_ids: list[str]
    models: list[str]
    metric_key: str


def render_sidebar_controls(datasets: list[str]) -> DashboardControls | None:
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

    return DashboardControls(
        dataset_ids=selected_datasets,
        models=models,
        metric_key=metric_key,
    )
