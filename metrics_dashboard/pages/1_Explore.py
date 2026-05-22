"""Explore metrics for one dataset."""

from __future__ import annotations

import streamlit as st

from metrics_dashboard.catalog import discover_datasets, discover_models
from metrics_dashboard.filters import render_explore_filters
from metrics_dashboard.sections import render_all_sections, render_focused_plot
from metrics_dashboard.state import get_param, get_param_list
from metrics_dashboard.ui import init_session_root, load_metrics_for_dataset

st.set_page_config(page_title="Explore", layout="wide")
st.title("Explore metrics")

root = init_session_root()
datasets = discover_datasets(root)

if not datasets:
    st.warning("No datasets under artifacts root.")
    st.stop()

# Bootstrap load for filter widgets
ds_hint = get_param("dataset") or datasets[0]
ds_idx = datasets.index(ds_hint) if ds_hint in datasets else 0
bootstrap_ds = datasets[ds_idx]
ev_models = discover_models(root / bootstrap_ds / "results" / "evaluation")
qp_models = get_param_list("models")
bootstrap_models = [m for m in (qp_models or ev_models) if m in ev_models] or ev_models
bootstrap_df = (
    load_metrics_for_dataset(bootstrap_ds, bootstrap_models, root)
    if bootstrap_models
    else __import__("pandas").DataFrame()
)

flt = render_explore_filters(bootstrap_df, datasets, root)
if flt is None:
    st.stop()

metrics_df = load_metrics_for_dataset(flt.dataset_id, flt.models, root)
if metrics_df.empty:
    st.error(f"No metrics CSVs for {flt.dataset_id} / {flt.models}.")
    st.stop()

st.caption(
    f"**{len(metrics_df):,}** rows · **{flt.dataset_id}** · "
    f"models: {', '.join(flt.models)} · **{metrics_df['intervention_id'].nunique()}** interventions"
)

st.download_button(
    "Download table CSV",
    metrics_df.to_csv(index=False).encode(),
    file_name=f"{flt.dataset_id}_metrics.csv",
    mime="text/csv",
)

st.subheader("Selected metric")
render_focused_plot(metrics_df, flt)

if flt.show_all_sections:
    st.subheader("All metric sections")
    render_all_sections(metrics_df, flt)
