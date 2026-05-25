"""Render matplotlib or interactive Plotly figures in Streamlit."""

from __future__ import annotations

from typing import Any

import streamlit as st

_PLOTLY_CONFIG: dict[str, Any] = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
}


def show_figure(
    fig: Any,
    *,
    interactive: bool,
    key: str | None = None,
) -> None:
    if interactive:
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=key,
            config=_PLOTLY_CONFIG,
        )
    else:
        st.pyplot(fig, clear_figure=True, use_container_width=True, key=key)
