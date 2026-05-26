"""Tests for obs column name resolution."""

from __future__ import annotations

import unittest

import pandas as pd

from scfm_controlled_manipulations.obs_columns import (
    resolve_batch_column,
    resolve_cell_type_column,
)


class ObsColumnResolveTest(unittest.TestCase):
    def test_celltype_alias(self) -> None:
        obs = pd.DataFrame({"celltype": ["A", "B", "A"], "batch": [1, 1, 2]})
        self.assertEqual(resolve_cell_type_column(obs, "cell_type"), "celltype")

    def test_exact_config_preferred(self) -> None:
        obs = pd.DataFrame({"cell_type": ["A"], "celltype": ["B"]})
        self.assertEqual(resolve_cell_type_column(obs, "cell_type"), "cell_type")

    def test_case_insensitive_config(self) -> None:
        obs = pd.DataFrame({"CellType": ["A"]})
        self.assertEqual(resolve_cell_type_column(obs, "celltype"), "CellType")

    def test_missing_returns_none(self) -> None:
        obs = pd.DataFrame({"x": [1]})
        self.assertIsNone(resolve_cell_type_column(obs, "cell_type"))

    def test_batch_alias(self) -> None:
        obs = pd.DataFrame({"sample_id": [1, 2], "celltype": ["A", "B"]})
        self.assertEqual(resolve_batch_column(obs, "batch"), "sample_id")


if __name__ == "__main__":
    unittest.main()
