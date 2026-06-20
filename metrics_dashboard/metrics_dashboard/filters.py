"""Sidebar controls for the metrics dashboard."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from metrics_dashboard.catalog import discover_models
from metrics_dashboard.config import MODEL_ORDER, bundle_root
from metrics_dashboard.style import render_theme_sidebar


@dataclass
class DashboardControls:
    dataset_ids: list[str]
    models: list[str]


def render_sidebar_controls(datasets: list[str]) -> DashboardControls | None:
    render_theme_sidebar()

    if not datasets:
        st.sidebar.error("No bundles in data/dashboard_bundles. Run make export-dashboard-bundle.")
        return None

    selected_dataset = st.sidebar.selectbox(
        "Dataset",
        datasets,
        help="Select one dataset to view.",
    )
    if not selected_dataset:
        return None

    root = bundle_root()
    all_models = discover_models(selected_dataset, root)
    all_models = sorted(set(all_models), key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)

    models = st.sidebar.multiselect("Models", all_models, default=all_models)
    if not models:
        return None

    return DashboardControls(
        dataset_ids=[selected_dataset],
        models=models,
    )
