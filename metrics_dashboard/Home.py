"""ScFMs Metrics Dashboard."""

from __future__ import annotations

import streamlit as st

from metrics_dashboard.catalog import catalog_table, discover_datasets
from metrics_dashboard.config import bundle_root

st.set_page_config(
    page_title="ScFMs Metrics Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ScFMs Metrics Dashboard")
st.markdown(
    """
Structure-evaluation metrics for single-cell foundation models under controlled manipulations.
Data is loaded from checked-in bundles in `data/dashboard_bundles/`.

Use **Metrics** in the sidebar for plots. Use **Dataset summary** for atlas sizes.
Switch light/dark mode via the Streamlit menu (⋮ → Settings → Theme).
"""
)

root = bundle_root()
datasets = discover_datasets(root)

if not datasets:
    st.warning(
        f"No bundles found under `{root}`. "
        "Export with: `make export-dashboard-bundle SOURCE=/path/to/sceval/dataset`"
    )
else:
    import pandas as pd

    st.subheader("Available datasets")
    rows = [
        {
            "dataset": s.dataset_id,
            "models": ", ".join(s.models),
            "n_cells": s.n_cells if s.n_cells is not None else "—",
            "updated": s.last_modified.isoformat() if s.last_modified else "—",
        }
        for s in catalog_table(root)
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
