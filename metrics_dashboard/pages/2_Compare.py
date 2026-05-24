"""Legacy compare view — use Metrics page for primary workflow."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Compare", layout="wide")
st.title("Compare (legacy)")
st.info(
    "Side-by-side A/B comparison has been superseded by the **Metrics** page, "
    "which supports multi-dataset averaging and the three standard plot sets. "
    "Open **Metrics** from the sidebar."
)
