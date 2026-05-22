"""SCEval metrics dashboard — home."""

from __future__ import annotations

import streamlit as st

from metrics_dashboard.catalog import discover_datasets
from metrics_dashboard.state import get_param, get_param_list, set_param_list, set_params
from metrics_dashboard.ui import init_session_root, render_catalog_summary

st.set_page_config(
    page_title="SCEval Metrics",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("SCEval metrics dashboard")
st.markdown(
    """
Explore structure-evaluation metrics across **datasets** and **models**.
Artifacts are read from `{ARTIFACTS_ROOT}/{dataset}/results/evaluation/{model}_metrics.csv`.

Use the sidebar to set the artifacts root, then open **Explore**, **Compare**, or **Model Card** from the pages menu.
Duplicate a browser tab and change URL query params (`?dataset=...&models=pca,scgpt`) for side-by-side comparison in separate tabs.
"""
)

root = init_session_root()
st.sidebar.caption(f"Artifacts: `{root}`")

if st.sidebar.button("Refresh catalog"):
    st.cache_data.clear()
    st.rerun()

datasets = discover_datasets(root)
if datasets:
    default_ds = get_param("dataset") or datasets[0]
    idx = datasets.index(default_ds) if default_ds in datasets else 0
    ds = st.sidebar.selectbox("Quick dataset", datasets, index=idx)
    qp_models = get_param_list("models")
    from metrics_dashboard.catalog import discover_models

    ev_models = discover_models(root / ds / "results" / "evaluation")
    models = st.sidebar.multiselect(
        "Quick models",
        ev_models,
        default=qp_models or ev_models,
    )
    if st.sidebar.button("Apply to URL"):
        set_params(dataset=ds)
        set_param_list("models", models)
        st.rerun()

st.subheader("Dataset catalog")
render_catalog_summary(root)

st.markdown(
    """
### Pages
- **Explore** — filter metrics and plot by category (all models on one chart).
- **Compare** — two views (A | B) in one window for different datasets/models.
- **Model Card** — aggregate metrics across datasets for selected model(s).

See [README.md](README.md) for deployment and query-parameter reference.
"""
)
