"""ScFMs Metrics Dashboard — home."""

from __future__ import annotations

import streamlit as st

from metrics_dashboard.catalog import discover_datasets
from metrics_dashboard.state import get_param_list, set_param_list, set_params
from metrics_dashboard.ui import init_session_root, render_catalog_summary

st.set_page_config(
    page_title="ScFMs Metrics Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ScFMs Metrics Dashboard")
st.markdown(
    """
Explore structure-evaluation metrics for single-cell foundation models under
controlled manipulations. Artifacts are read from
`{ARTIFACTS_ROOT}/{dataset}/results/evaluation/{model}_metrics.csv`.

**Pages**
- **Metrics** — manipulation sweeps, integration correlations, embedding collapse/shift.
- **Dataset summary** — cells, genes, cell types, batches per atlas.
- **Compare** / **Model card** — legacy views (use Metrics for the primary workflow).

Use the sidebar to set the artifacts root. Select datasets and models on the Metrics page.
Toggle light/dark appearance via Streamlit settings (⋮ menu → Settings → Theme).
"""
)

root = init_session_root()
st.sidebar.caption(f"Artifacts: `{root}`")

if st.sidebar.button("Refresh catalog"):
    st.cache_data.clear()
    st.rerun()

datasets = discover_datasets(root)
if datasets:
    default_ds = get_param_list("datasets") or datasets[:1]
    ds_pick = st.sidebar.multiselect("Quick datasets", datasets, default=[d for d in default_ds if d in datasets] or datasets[:1])
    from metrics_dashboard.catalog import discover_models

    if ds_pick:
        ev_models = discover_models(root / ds_pick[0])
        models = st.sidebar.multiselect("Quick models", ev_models, default=ev_models)
        if st.sidebar.button("Apply to URL"):
            set_params(datasets=",".join(ds_pick))
            set_param_list("models", models)
            st.rerun()

st.subheader("Dataset catalog")
render_catalog_summary(root)
