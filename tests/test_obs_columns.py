"""Tests for obs column name resolution."""

from __future__ import annotations

import unittest

import pandas as pd

from scfm_controlled_manipulations.obs_columns import (
    atlas_key_from_dataset_id,
    resolve_batch_column,
    resolve_cell_type_column,
    resolve_cell_type_column_for_dataset,
    resolve_stratify_column,
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

    def test_atlas_key_from_dataset_id(self) -> None:
        self.assertEqual(atlas_key_from_dataset_id("brain"), "brain")
        self.assertEqual(atlas_key_from_dataset_id("brain_n200_s0"), "brain")
        self.assertEqual(atlas_key_from_dataset_id("tabula_sapiens_n500_s1"), "tabula_sapiens")
        self.assertIsNone(atlas_key_from_dataset_id("unknown"))

    def test_resolve_cell_type_column_for_dataset_brain(self) -> None:
        obs = pd.DataFrame(
            {
                "cell_type": ["neuron", "leukocyte"],
                "supercluster_term": ["A", "B"],
            }
        )
        col = resolve_cell_type_column_for_dataset(obs, "cell_type", dataset_id="brain")
        self.assertEqual(col, "supercluster_term")

    def test_resolve_stratify_column_brain_override(self) -> None:
        obs = pd.DataFrame(
            {
                "cell_type": ["neuron"] * 9 + ["leukocyte"],
                "supercluster_term": ["A"] * 5 + ["B"] * 5,
            }
        )
        col = resolve_stratify_column(obs, atlas="brain")
        self.assertEqual(col, "supercluster_term")

    def test_resolve_stratify_column_fallback_when_coarse(self) -> None:
        obs = pd.DataFrame(
            {
                "cell_type": ["neuron"] * 9 + ["leukocyte"],
                "cluster_id": ["1", "2", "3", "1", "2", "3", "1", "2", "3", "1"],
            }
        )
        col = resolve_stratify_column(obs, atlas=None, min_labels=5)
        self.assertEqual(col, "cluster_id")


if __name__ == "__main__":
    unittest.main()
