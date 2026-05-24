"""Matplotlib theme helpers aligned with Streamlit light/dark mode."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st


def streamlit_is_dark() -> bool:
    """User-selected theme (Settings → Theme), not only config.toml defaults."""
    try:
        theme = st.context.theme
        if theme is not None and theme.base:
            return str(theme.base).lower() == "dark"
    except (AttributeError, TypeError):
        pass
    try:
        return str(st.get_option("theme.base")).lower() == "dark"
    except Exception:
        return False


def plot_colors() -> dict[str, str]:
    if streamlit_is_dark():
        return {
            "figure": "#0e1117",
            "axes": "#0e1117",
            "text": "#fafafa",
            "grid": "#31333f",
            "spine": "#52555e",
            "null_line": "#94a3b8",
            "ci_alpha": 0.25,
        }
    return {
        "figure": "#ffffff",
        "axes": "#ffffff",
        "text": "#0f172a",
        "grid": "#e2e8f0",
        "spine": "#cbd5e1",
        "null_line": "#64748b",
        "ci_alpha": 0.18,
    }


def configure_matplotlib() -> None:
    colors = plot_colors()
    plt.rcParams.update(
        {
            "figure.facecolor": colors["figure"],
            "axes.facecolor": colors["axes"],
            "text.color": colors["text"],
            "axes.labelcolor": colors["text"],
            "axes.titlecolor": colors["text"],
            "xtick.color": colors["text"],
            "ytick.color": colors["text"],
            "axes.edgecolor": colors["spine"],
            "grid.color": colors["grid"],
            "legend.facecolor": colors["axes"],
            "legend.edgecolor": colors["spine"],
            "legend.labelcolor": colors["text"],
        }
    )


def apply_minimal_axes(ax: plt.Axes) -> None:
    configure_matplotlib()
    colors = plot_colors()
    ax.set_facecolor(colors["axes"])
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
