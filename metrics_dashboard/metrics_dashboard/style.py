"""Matplotlib theme helpers (light/dark aware via Streamlit)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st


def streamlit_is_dark() -> bool:
    try:
        base = st.get_option("theme.base")
        return str(base).lower() == "dark"
    except Exception:
        return False


def plot_colors() -> dict[str, str]:
    if streamlit_is_dark():
        return {
            "text": "#e2e8f0",
            "grid": "#334155",
            "spine": "#475569",
            "null_line": "#94a3b8",
            "ci_alpha": 0.22,
        }
    return {
        "text": "#0f172a",
        "grid": "#e2e8f0",
        "spine": "#cbd5e1",
        "null_line": "#64748b",
        "ci_alpha": 0.18,
    }


def apply_minimal_axes(ax: plt.Axes) -> None:
    colors = plot_colors()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(colors["spine"])
    ax.spines["bottom"].set_color(colors["spine"])
    ax.tick_params(colors=colors["text"])
    ax.xaxis.label.set_color(colors["text"])
    ax.yaxis.label.set_color(colors["text"])
    ax.title.set_color(colors["text"])
    ax.grid(True, alpha=0.35, color=colors["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
