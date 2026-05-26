"""Render Plotly (interactive) or matplotlib figures in Streamlit."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st

_PLOTLY_CONFIG: dict[str, Any] = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
}


def show_figure(fig: Any) -> None:
    """Display a figure; Plotly figures get zoom/pan via the mode bar."""
    if isinstance(fig, go.Figure):
        st.plotly_chart(fig, width="stretch", config=_PLOTLY_CONFIG)
    else:
        st.pyplot(fig, clear_figure=True, width="stretch")
