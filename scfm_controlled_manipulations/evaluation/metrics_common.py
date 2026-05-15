"""Shared helpers for evaluation metric rows."""

from __future__ import annotations

import numpy as np


def distribution_summary(values: np.ndarray) -> tuple[float, float, float]:
    """Mean, median, and sample std for a 1d per-observation array."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(arr)), float(np.median(arr)), float(np.std(arr))


def scalar_summary(value: float) -> tuple[float, float, float]:
    """Summary for a single global statistic (no per-cell distribution)."""
    if np.isnan(value):
        return float("nan"), float("nan"), float("nan")
    return float(value), float(value), float("nan")
