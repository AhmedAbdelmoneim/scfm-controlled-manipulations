"""ScFMs Metrics Dashboard."""

from __future__ import annotations

import bootstrap  # noqa: F401

import logging

import pandas as pd
import streamlit as st

from metrics_dashboard.catalog import catalog_table, discover_datasets
from metrics_dashboard.config import bundle_root
from metrics_dashboard.runtime import log_startup_context
from metrics_dashboard.style import render_theme_sidebar

log = logging.getLogger("scfm_dashboard.home")
log_startup_context()

st.set_page_config(
    page_title="ScFMs Metrics Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_theme_sidebar()

root = bundle_root()
datasets = discover_datasets(root)
log.info("Home: root=%s datasets=%s", root, datasets)

st.title("ScFMs Metrics Dashboard")
st.markdown(
    """
Structure-evaluation metrics for single-cell foundation models under controlled manipulations.
Open **Metrics** in the sidebar for plots, or **Dataset summary** for atlas sizes.
"""
)

if not datasets:
    st.error(
        f"No bundles under `{root}`. Export with "
        "`make export-dashboard-bundle SOURCE=...` and commit `data/dashboard_bundles/`."
    )
else:
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
