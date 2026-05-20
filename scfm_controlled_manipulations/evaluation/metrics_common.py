"""Shared helpers for evaluation metric rows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DistributionSummary:
    mean: float
    median: float
    std: float
    min: float
    max: float
    q05: float
    q25: float
    q75: float
    q95: float


NAN_SUMMARY = DistributionSummary(
    mean=float("nan"),
    median=float("nan"),
    std=float("nan"),
    min=float("nan"),
    max=float("nan"),
    q05=float("nan"),
    q25=float("nan"),
    q75=float("nan"),
    q95=float("nan"),
)

VALUE_SUMMARY_COLUMNS: tuple[str, ...] = (
    "value_mean",
    "value_median",
    "value_std",
    "value_min",
    "value_max",
    "value_q05",
    "value_q25",
    "value_q75",
    "value_q95",
)


def distribution_summary(values: np.ndarray) -> DistributionSummary:
    """Summarize a 1d per-observation array (mean, median, std, min, max, quantiles)."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return NAN_SUMMARY
    return DistributionSummary(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        q05=float(np.quantile(arr, 0.05)),
        q25=float(np.quantile(arr, 0.25)),
        q75=float(np.quantile(arr, 0.75)),
        q95=float(np.quantile(arr, 0.95)),
    )


def scalar_summary(value: float) -> DistributionSummary:
    """Summary for a single global statistic (no per-cell distribution)."""
    if np.isnan(value):
        return NAN_SUMMARY
    v = float(value)
    return DistributionSummary(
        mean=v,
        median=v,
        std=float("nan"),
        min=v,
        max=v,
        q05=float("nan"),
        q25=float("nan"),
        q75=float("nan"),
        q95=float("nan"),
    )


def summary_to_row_fields(summary: DistributionSummary) -> dict[str, float]:
    return {
        "value_mean": summary.mean,
        "value_median": summary.median,
        "value_std": summary.std,
        "value_min": summary.min,
        "value_max": summary.max,
        "value_q05": summary.q05,
        "value_q25": summary.q25,
        "value_q75": summary.q75,
        "value_q95": summary.q95,
    }
