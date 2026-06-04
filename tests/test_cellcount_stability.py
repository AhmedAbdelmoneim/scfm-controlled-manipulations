"""Tests for cell-count stability analysis helpers."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "lib"))

from cellcount_sweep import parse_run_dir  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from analyze_cellcount_stability import (  # noqa: E402
    aggregate_across_seeds,
    find_stabilization_n,
)


class TestCellcountStability(unittest.TestCase):
    def test_parse_run_dir(self) -> None:
        self.assertEqual(parse_run_dir("n2000_s3"), (2000, 3))
        self.assertIsNone(parse_run_dir("immune"))

    def test_find_stabilization_n(self) -> None:
        curve = pd.DataFrame(
            {
                "cell_count": [200, 500, 1000, 2000],
                "mean_across_seeds": [1.0, 1.01, 1.005, 1.004],
                "ci_half_width_rel": [0.2, 0.08, 0.03, 0.02],
            }
        )
        n_star = find_stabilization_n(
            curve, cv_threshold=0.05, plateau_threshold=0.02, consecutive=2
        )
        self.assertEqual(n_star, 1000)

    def test_aggregate_across_seeds(self) -> None:
        metrics = pd.DataFrame(
            {
                "atlas": ["immune"] * 4,
                "cell_count": [200, 200, 500, 500],
                "model": ["pca"] * 4,
                "metric_category": ["knn_metrics"] * 4,
                "metric_name": ["knn_recall"] * 4,
                "space": ["embedding"] * 4,
                "k": [15] * 4,
                "value_mean": [0.5, 0.52, 0.8, 0.81],
                "sweep_seed": [0, 1, 0, 1],
            }
        )
        agg = aggregate_across_seeds(metrics)
        self.assertEqual(len(agg), 2)
        row_200 = agg[agg["cell_count"] == 200].iloc[0]
        self.assertAlmostEqual(row_200["mean_across_seeds"], 0.51)


if __name__ == "__main__":
    unittest.main()
