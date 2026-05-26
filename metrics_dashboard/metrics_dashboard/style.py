"""Streamlit appearance helpers for Plotly chart theming."""

from __future__ import annotations

import streamlit as st

APPEARANCE_KEY = "appearance"


def render_theme_sidebar() -> None:
    """Sidebar theme control (Streamlit Cloud hides the ⋮ → Settings theme picker)."""
    st.sidebar.selectbox(
        "Appearance",
        ["Light", "Dark"],
        index=1,
        key=APPEARANCE_KEY,
    )


def streamlit_is_dark() -> bool:
    appearance = st.session_state.get(APPEARANCE_KEY)
    if appearance is not None:
        return str(appearance).lower() == "dark"
    try:
        theme = st.context.theme
        if theme is not None and theme.base:
            return str(theme.base).lower() == "dark"
    except (AttributeError, TypeError):
        pass
    try:
        return str(st.get_option("theme.base")).lower() == "dark"
    except Exception:
        return True
