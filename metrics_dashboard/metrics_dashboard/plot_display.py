"""Render matplotlib figures in Streamlit."""

from __future__ import annotations

from typing import Any

import streamlit as st


def show_figure(fig: Any) -> None:
    st.pyplot(fig, clear_figure=True, width="stretch")
