"""Tests for metrics dashboard transforms and config."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from metrics_dashboard.config import DASHBOARD_METRICS, MODEL_COLORS, MODEL_ORDER
from metrics_dashboard.transforms import (
    average_metrics_across_datasets,
    prepare_set1_grid,
    prepare_set2_correlation,
    std_bounds,
)
from scfm_controlled_manipulations.evaluation.metrics_common import (
    distribution_summary,
    make_metric_row,
)


class StdBoundsTest(unittest.TestCase):
    def test_mean_plus_minus_std(self) -> None:
        row = pd.Series({"value_mean": 0.5, "value_std": 0.1})
        lo, hi = std_bounds(row)
        self.assertAlmostEqual(lo, 0.4)
        self.assertAlmostEqual(hi, 0.6)

    def test_missing_std_collapses_to_mean(self) -> None:
        row = pd.Series({"value_mean": 0.5, "value_std": float("nan")})
        lo, hi = std_bounds(row)
        self.assertAlmostEqual(lo, 0.5)
        self.assertAlmostEqual(hi, 0.5)


class MetricRowSchemaTest(unittest.TestCase):
    def test_make_metric_row_has_std(self) -> None:
        row = make_metric_row(
            dataset_id="ds",
            model="pca",
            intervention_id="i1",
            intervention_name="downsample",
            metric_category="knn_metrics",
            metric_name="knn_recall",
            space="embedding",
            summary=distribution_summary(np.array([0.5, 0.6, 0.7])),
            n_cells=3,
            seed=0,
            null_value=0.4,
        )
        self.assertIn("value_std", row)
        self.assertNotIn("value_ci_lower", row)


class TransformTest(unittest.TestCase):
    def _toy_metrics(self) -> pd.DataFrame:
        rows = []
        for model in ("pca", "scgpt"):
            for frac in (0.5, 0.9):
                rows.append(
                    {
                        "dataset_id": "ds1",
                        "model": model,
                        "intervention_id": f"down_{frac}",
                        "intervention_name": "downsample",
                        "metric_category": "knn_metrics",
                        "metric_name": "knn_recall",
                        "space": "embedding",
                        "value_mean": frac,
                        "value_std": 0.05,
                        "null_value": 0.1,
                        "k": 15,
                        "diffusion_t": np.nan,
                        "param_value": frac,
                        "param_key": "fraction",
                    }
                )
        rows.append(
            {
                "dataset_id": "ds1",
                "model": "pca",
                "intervention_id": "ref",
                "intervention_name": "reference",
                "metric_category": "cell_type_and_batch_metrics",
                "metric_name": "cell_type_asw",
                "space": "embedding_manipulated",
                "value_mean": 0.8,
                "value_std": 0.02,
                "param_value": 0.0,
            }
        )
        return pd.DataFrame(rows)

    def test_average_across_datasets(self) -> None:
        df = self._toy_metrics()
        df2 = df.copy()
        df2["dataset_id"] = "ds2"
        df2["value_mean"] = df2["value_mean"] + 0.1
        combined = pd.concat([df, df2], ignore_index=True)
        avg = average_metrics_across_datasets(combined)
        self.assertEqual(avg["dataset_id"].iloc[0], "averaged")
        self.assertEqual(len(avg), len(df))

    def test_prepare_set1_grid(self) -> None:
        df = self._toy_metrics()
        spec = DASHBOARD_METRICS["knn_recall"]
        layout = prepare_set1_grid(df, spec, ["pca", "scgpt"])
        self.assertIn("downsample", layout.row_labels)
        self.assertFalse(layout.data.empty)
        self.assertEqual(layout.x_col, "param_value")

    def test_prepare_set1_grid_kl_layout(self) -> None:
        """KL/JS: columns = manipulation config, x-axis = diffusion_t."""
        rows_data = []
        for t in (1.0, 2.0, 4.0):
            for frac in (0.2, 0.5, 0.8):
                rows_data.append(
                    {
                        "dataset_id": "ds1",
                        "model": "pca",
                        "intervention_id": "down_a",
                        "intervention_name": "downsample",
                        "metric_category": "knn_metrics",
                        "metric_name": "diffusion_sym_kl",
                        "space": "embedding",
                        "value_mean": 0.1 * t,
                        "value_std": 0.01,
                        "k": 15,
                        "diffusion_t": t,
                        "param_value": str(frac),
                        "param_key": "fraction",
                    }
                )
        df = pd.DataFrame(rows_data)
        spec = DASHBOARD_METRICS["kl_divergence"]
        layout = prepare_set1_grid(df, spec, ["pca"])
        self.assertEqual(layout.x_col, "diffusion_t")
        self.assertEqual(layout.col_labels_by_row["downsample"], ["0.2", "0.5", "0.8"])
        self.assertIn("downsample", layout.row_labels)

    def test_prepare_set2_correlation(self) -> None:
        df = self._toy_metrics()
        spec = DASHBOARD_METRICS["knn_recall"]
        wide = prepare_set2_correlation(df, spec, ["pca"])
        self.assertIn("metric_score", wide.columns)

    def test_model_colors_cover_registry(self) -> None:
        for m in MODEL_ORDER:
            self.assertIn(m, MODEL_COLORS)


if __name__ == "__main__":
    unittest.main()
