"""Streamlit sidebar filters for metrics exploration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from metrics_dashboard.catalog import discover_models, dataset_status
from metrics_dashboard.config import VALUE_COLUMNS, artifacts_root
from metrics_dashboard.state import get_param, get_param_list, set_param_list, set_params


@dataclass
class ExploreFilters:
    dataset_id: str
    models: list[str]
    metric_category: str
    metric_name: str
    space: str
    interventions: list[str]
    y_col: str
    k: float | None
    diffusion_t: float | None
    resolution: float | None
    show_all_sections: bool


def render_artifacts_root_sidebar() -> Path:
    with st.sidebar.expander("Advanced", expanded=False):
        root_str = st.text_input(
            "Artifacts root (SCFM_ARTIFACTS_ROOT)",
            value=str(artifacts_root()),
            key="artifacts_root_input",
        )
    return Path(root_str)


def _default_dataset(datasets: list[str], root: Path) -> str:
    qp = get_param("dataset")
    if qp and qp in datasets:
        return qp
    for ds in datasets:
        if dataset_status(ds, root).has_evaluation:
            return ds
    return datasets[0] if datasets else ""


def render_explore_filters(
    metrics_df: pd.DataFrame,
    datasets: list[str],
    root: Path,
    *,
    prefix: str = "",
    ui: Any | None = None,
) -> ExploreFilters | None:
    """Filters for explore/compare; ``ui`` is sidebar or a column (default sidebar)."""
    p = prefix
    label_suffix = f" ({prefix.rstrip('_')})" if prefix else ""
    box = ui if ui is not None else st.sidebar

    if not datasets:
        st.warning("No datasets found under artifacts root.")
        return None

    default_ds = get_param(f"{p}dataset") or _default_dataset(datasets, root)
    idx = datasets.index(default_ds) if default_ds in datasets else 0
    dataset_id = box.selectbox(
        f"Dataset{label_suffix}",
        datasets,
        index=idx,
        key=f"{p}dataset_select",
    )

    ev_models = discover_models(root / dataset_id / "results" / "evaluation")
    qp_models = get_param_list(f"{p}models") or get_param_list("models")
    available = [m for m in ev_models if m in (qp_models or ev_models)]
    if not available:
        available = ev_models

    models = box.multiselect(
        f"Models{label_suffix}",
        ev_models,
        default=available or ev_models,
        key=f"{p}models_select",
    )
    if not models:
        box.info("Select at least one model.")
        return None

    if not prefix:
        set_params(dataset=dataset_id)
        set_param_list("models", models)

    if metrics_df.empty:
        st.warning(f"No metrics loaded for {dataset_id}.")
        return None

    categories = sorted(metrics_df["metric_category"].dropna().unique())
    default_cat = get_param(f"{p}category") or categories[0]
    cat_idx = categories.index(default_cat) if default_cat in categories else 0
    metric_category = box.selectbox(
        f"Metric category{label_suffix}",
        categories,
        index=cat_idx,
        key=f"{p}category_select",
    )

    cat_df = metrics_df[metrics_df["metric_category"] == metric_category]
    metric_names = sorted(cat_df["metric_name"].dropna().unique())
    default_metric = get_param(f"{p}metric") or (metric_names[0] if metric_names else "")
    met_idx = metric_names.index(default_metric) if default_metric in metric_names else 0
    metric_name = box.selectbox(
        f"Metric{label_suffix}",
        metric_names,
        index=met_idx,
        key=f"{p}metric_select",
    )

    met_df = cat_df[cat_df["metric_name"] == metric_name]
    spaces = sorted(met_df["space"].dropna().unique())
    default_space = get_param(f"{p}space") or (spaces[0] if spaces else "")
    sp_idx = spaces.index(default_space) if default_space in spaces else 0
    space = box.selectbox(
        f"Space{label_suffix}",
        spaces,
        index=sp_idx,
        key=f"{p}space_select",
    )

    interventions = sorted(metrics_df["intervention_name"].dropna().unique())
    selected_interventions = box.multiselect(
        f"Interventions{label_suffix}",
        interventions,
        default=interventions,
        key=f"{p}interventions_select",
    )

    y_col = box.selectbox(
        f"Y column{label_suffix}",
        [c for c in VALUE_COLUMNS if c in metrics_df.columns],
        index=0,
        key=f"{p}y_col_select",
    )

    k_val: float | None = None
    t_val: float | None = None
    res_val: float | None = None

    if metric_category in ("knn_metrics", "knn_metrics_gain"):
        k_opts = sorted(met_df["k"].dropna().unique())
        if k_opts:
            k_val = float(
                box.selectbox(
                    f"k{label_suffix}",
                    k_opts,
                    key=f"{p}k_select",
                )
            )
    if metric_category == "knn_metrics" and metric_name in ("diffusion_js", "diffusion_sym_kl"):
        t_opts = sorted(met_df["diffusion_t"].dropna().unique())
        if t_opts:
            t_val = float(
                box.selectbox(
                    f"diffusion_t{label_suffix}",
                    t_opts,
                    key=f"{p}t_select",
                )
            )
    if metric_category == "clustering_metrics":
        res_opts = sorted(met_df["resolution"].dropna().unique())
        if res_opts:
            res_val = float(
                box.selectbox(
                    f"resolution{label_suffix}",
                    res_opts,
                    key=f"{p}res_select",
                )
            )

    show_all = box.checkbox(
        f"Show all metric sections{label_suffix}",
        value=not prefix,
        key=f"{p}show_all_sections",
    )

    return ExploreFilters(
        dataset_id=dataset_id,
        models=models,
        metric_category=metric_category,
        metric_name=metric_name,
        space=space,
        interventions=selected_interventions,
        y_col=y_col,
        k=k_val,
        diffusion_t=t_val,
        resolution=res_val,
        show_all_sections=show_all,
    )
