"""Render matplotlib figures in Streamlit."""

from __future__ import annotations

from typing import Any

import streamlit as st


def show_figure(fig: Any, *, key: str | None = None) -> None:
    st.pyplot(fig, clear_figure=True, use_container_width=True, key=key)
