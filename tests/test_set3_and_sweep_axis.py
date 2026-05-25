"""Tests for Set 3 embedding-shift wiring and sweep axis ordering."""

from __future__ import annotations

import unittest

import pandas as pd

from metrics_dashboard.sweep_axis import ordered_param_values, sweep_is_numeric
from metrics_dashboard.transforms import prepare_set3_embedding


class Set3PrepareTest(unittest.TestCase):
    def _embedding_shift_rows(self) -> pd.DataFrame:
        rows = []
        for metric_name, mean in (
            ("within_ref_pairwise_l2", 90.0),
            ("within_man_pairwise_l2", 80.0),
            ("paired_cell_l2_norm", 50.0),
        ):
            rows.append(
                {
                    "dataset_id": "ds",
                    "model": "pca",
                    "intervention_id": "down_02",
                    "intervention_name": "downsample",
                    "metric_category": "embedding_shift",
                    "metric_name": metric_name,
                    "space": "embedding",
                    "value_mean": mean,
                    "value_std": 1.0,
                    "param_key": "fraction",
                    "param_value": "0.2",
                }
            )
        return pd.DataFrame(rows)

    def test_shift_reference_is_zero_not_within_ref_scale(self) -> None:
        collapse, shift = prepare_set3_embedding(self._embedding_shift_rows(), ["pca"])
        ref = shift[(shift["intervention_name"] == "reference") & (shift["model"] == "pca")]
        self.assertEqual(len(ref), 1)
        self.assertEqual(ref["value_mean"].iloc[0], 0.0)
        ref_c = collapse[(collapse["intervention_name"] == "reference") & (collapse["model"] == "pca")]
        self.assertEqual(ref_c["value_mean"].iloc[0], 90.0)


class SweepAxisTest(unittest.TestCase):
    def test_gene_shuffle_variant_order(self) -> None:
        vals = pd.Series(["random", "chromosome", "chromosome_control", "stratified"])
        order = ordered_param_values(vals, intervention_name="gene_shuffle")
        self.assertEqual(
            order,
            ["chromosome_control", "chromosome", "stratified", "random"],
        )
        self.assertFalse(sweep_is_numeric(vals, intervention_name="gene_shuffle"))


if __name__ == "__main__":
    unittest.main()
