"""Sidebar controls for the main metrics dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from metrics_dashboard.catalog import discover_datasets, discover_models, dataset_status
from metrics_dashboard.config import (
    DASHBOARD_METRIC_KEYS,
    DASHBOARD_METRICS,
    MODEL_ORDER,
    artifacts_root,
)
from metrics_dashboard.state import get_param, get_param_list, set_param_list, set_params


@dataclass
class DashboardControls:
    dataset_ids: list[str]
    models: list[str]
    metric_key: str


def render_artifacts_root_sidebar() -> Path:
    with st.sidebar.expander("Advanced", expanded=False):
        root_str = st.text_input(
            "Artifacts root (SCFM_ARTIFACTS_ROOT)",
            value=str(artifacts_root()),
            key="artifacts_root_input",
        )
    return Path(root_str)


def _default_datasets(datasets: list[str], root: Path) -> list[str]:
    qp = get_param_list("datasets") or get_param_list("dataset")
    if qp:
        valid = [d for d in qp if d in datasets]
        if valid:
            return valid
    for ds in datasets:
        if dataset_status(ds, root).has_evaluation:
            return [ds]
    return [datasets[0]] if datasets else []


def render_dashboard_controls(
    datasets: list[str],
    root: Path,
) -> DashboardControls | None:
    if not datasets:
        st.sidebar.warning("No datasets under artifacts root.")
        return None

    default_ds = _default_datasets(datasets, root)
    selected_datasets = st.sidebar.multiselect(
        "Dataset(s)",
        datasets,
        default=default_ds,
        help="Select one dataset or multiple to average metrics.",
    )
    if not selected_datasets:
        st.sidebar.info("Select at least one dataset.")
        return None

    ev_models: list[str] = []
    for ds in selected_datasets:
        ev_models.extend(discover_models(root / ds))
    ev_models = sorted(set(ev_models), key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)

    qp_models = get_param_list("models")
    default_models = [m for m in (qp_models or ev_models) if m in ev_models] or ev_models
    models = st.sidebar.multiselect("Models", ev_models, default=default_models)
    if not models:
        st.sidebar.info("Select at least one model.")
        return None

    metric_key = st.sidebar.selectbox(
        "Metric",
        DASHBOARD_METRIC_KEYS,
        format_func=lambda k: DASHBOARD_METRICS[k].label,
        index=0,
    )

    set_params(datasets=",".join(selected_datasets))
    set_param_list("models", models)

    return DashboardControls(
        dataset_ids=selected_datasets,
        models=models,
        metric_key=metric_key,
    )
