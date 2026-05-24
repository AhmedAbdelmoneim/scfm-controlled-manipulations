"""Legacy model card — use Metrics page for primary workflow."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Model card", layout="wide")
st.title("Model card (legacy)")
st.info(
    "Cross-dataset model aggregation is available on the **Metrics** page via "
    "multi-dataset selection (metrics are averaged). Open **Metrics** from the sidebar."
)
