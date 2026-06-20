"""Tests for metrics dashboard transforms and config."""

from __future__ import annotations

import json
import unittest

from metrics_dashboard.config import MODEL_COLORS, MODEL_ORDER
from metrics_dashboard.transforms import (
    average_metrics_across_datasets,
    prepare_set1_main_metrics,
    prepare_set2_rnx_curves,
    std_bounds,
)
import numpy as np
import pandas as pd

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
            metric_category="structure_metrics",
            metric_name="viscore_local_sp",
            space="embedding",
            summary=distribution_summary(np.array([0.5, 0.6, 0.7])),
            n_cells=3,
            seed=0,
        )
        self.assertIn("value_std", row)
        self.assertNotIn("value_ci_lower", row)


class TransformTest(unittest.TestCase):
    def _toy_metrics(self) -> pd.DataFrame:
        rows = []
        for model in ("pca", "scgpt"):
            for frac in (0.5, 0.9):
                for metric in ("viscore_local_sp", "viscore_global_sp", "distcorr"):
                    rows.append(
                        {
                            "dataset_id": "ds1",
                            "model": model,
                            "intervention_id": f"down_{frac}",
                            "intervention_name": "downsample",
                            "metric_category": "structure_metrics",
                            "metric_name": metric,
                            "space": "embedding",
                            "value_mean": frac,
                            "value_std": 0.05,
                            "param_value": frac,
                            "param_key": "fraction",
                        }
                    )
                rows.append(
                    {
                        "dataset_id": "ds1",
                        "model": model,
                        "intervention_id": f"down_{frac}",
                        "intervention_name": "downsample",
                        "metric_category": "structure_metrics",
                        "metric_name": "rnx_curve",
                        "space": "embedding",
                        "value_mean": np.nan,
                        "value_std": np.nan,
                        "param_value": frac,
                        "param_key": "fraction",
                        "rnx_curve_json": json.dumps({"k": [1, 2], "rnx": [0.2, 0.4]}),
                    }
                )
                rows.append(
                    {
                        "dataset_id": "ds1",
                        "model": model,
                        "intervention_id": f"down_{frac}",
                        "intervention_name": "downsample",
                        "metric_category": "clustering_metrics",
                        "metric_name": "leiden_ari",
                        "space": "embedding",
                        "value_mean": 0.75,
                        "value_std": 0.02,
                        "param_value": frac,
                        "param_key": "fraction",
                        "resolution": 1.0,
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

    def test_prepare_set1_main_metrics(self) -> None:
        df = self._toy_metrics()
        layout = prepare_set1_main_metrics(df, ["pca", "scgpt"])
        self.assertIn("downsample", layout.manipulations)
        self.assertIn("ViScore local SP", layout.metric_labels)
        self.assertIn("Leiden ARI", layout.metric_labels)
        self.assertFalse(layout.data.empty)
        self.assertEqual(layout.x_col, "param_value")
        self.assertEqual(layout.y_ranges["Leiden ARI"], (-1.0, 1.0))
        self.assertEqual(layout.y_ranges["Distance correlation"], (0.0, 1.0))

    def test_prepare_set2_rnx_curves(self) -> None:
        df = self._toy_metrics()
        layout = prepare_set2_rnx_curves(df, ["pca"])
        self.assertIn("downsample", layout.manipulations)
        self.assertFalse(layout.data.empty)
        self.assertIn("k", layout.data.columns)
        self.assertIn("rnx", layout.data.columns)
        self.assertEqual(sorted(layout.data["k"].unique().tolist()), [1, 2])

    def test_model_colors_cover_registry(self) -> None:
        for m in MODEL_ORDER:
            self.assertIn(m, MODEL_COLORS)


if __name__ == "__main__":
    unittest.main()
