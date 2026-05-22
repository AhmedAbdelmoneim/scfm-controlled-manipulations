"""Side-by-side comparison of two metric views."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from metrics_dashboard.catalog import discover_datasets, discover_models
from metrics_dashboard.filters import ExploreFilters, render_explore_filters
from metrics_dashboard.sections import render_focused_plot
from metrics_dashboard.state import get_param, prefixed_param
from metrics_dashboard.ui import init_session_root, load_metrics_for_dataset

st.set_page_config(page_title="Compare", layout="wide")
st.title("Compare views")

root = init_session_root()
datasets = discover_datasets(root)

if not datasets:
    st.warning("No datasets under artifacts root.")
    st.stop()

lock_metric = st.checkbox(
    "Lock View B to View A metric category / metric / space",
    value=False,
)

col_a, col_b = st.columns(2)


def _load_for_prefix(prefix: str) -> tuple[pd.DataFrame, str, list[str]]:
    ds = prefixed_param(prefix, "dataset") or (get_param("dataset") if prefix == "a_" else None)
    if not ds or ds not in datasets:
        ds = datasets[0]
    ev_models = discover_models(root / ds / "results" / "evaluation")
    return (
        load_metrics_for_dataset(ds, ev_models, root) if ev_models else pd.DataFrame(),
        ds,
        ev_models,
    )


with col_a:
    st.subheader("View A")
    df_a, _, _ = _load_for_prefix("a_")
    flt_a = render_explore_filters(df_a, datasets, root, prefix="a_", ui=col_a)

with col_b:
    st.subheader("View B")
    df_b, _, _ = _load_for_prefix("b_")
    flt_b = render_explore_filters(df_b, datasets, root, prefix="b_", ui=col_b)

if flt_a is None or flt_b is None:
    st.stop()

metrics_a = load_metrics_for_dataset(flt_a.dataset_id, flt_a.models, root)
metrics_b = load_metrics_for_dataset(flt_b.dataset_id, flt_b.models, root)

if lock_metric:
    flt_b = ExploreFilters(
        dataset_id=flt_b.dataset_id,
        models=flt_b.models,
        metric_category=flt_a.metric_category,
        metric_name=flt_a.metric_name,
        space=flt_a.space,
        interventions=flt_b.interventions,
        y_col=flt_b.y_col,
        k=flt_a.k,
        diffusion_t=flt_a.diffusion_t,
        resolution=flt_a.resolution,
        show_all_sections=False,
    )

st.divider()

plot_a, plot_b = st.columns(2)
with plot_a:
    st.caption(f"**A:** {flt_a.dataset_id} · {', '.join(flt_a.models)}")
    if metrics_a.empty:
        st.warning("No data for View A.")
    else:
        render_focused_plot(metrics_a, flt_a)

with plot_b:
    st.caption(f"**B:** {flt_b.dataset_id} · {', '.join(flt_b.models)}")
    if metrics_b.empty:
        st.warning("No data for View B.")
    else:
        render_focused_plot(metrics_b, flt_b)

st.info(
    "Open two browser tabs on **Explore** with different `?dataset=` and `?models=` "
    "for full-window side-by-side comparison."
)
