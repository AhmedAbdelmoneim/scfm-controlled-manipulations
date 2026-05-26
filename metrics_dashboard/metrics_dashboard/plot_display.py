"""Render interactive Plotly figures in Streamlit."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

_PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
}


def show_figure(fig: go.Figure) -> None:
    if not isinstance(fig, go.Figure):
        raise TypeError(f"Expected plotly.graph_objects.Figure, got {type(fig)!r}")
    st.plotly_chart(fig, width="stretch", config=_PLOTLY_CONFIG)
