"""Tests for dashboard bundle export/load."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from metrics_dashboard.bundle import (
    METRICS_FILENAME,
    SUMMARY_FILENAME,
    export_dataset_bundle,
    is_bundle_dataset_dir,
    load_metrics_table,
)


class BundleExportTest(unittest.TestCase):
    def test_roundtrip_legacy_to_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = "toy_ds"
            ev = root / ds / "results" / "evaluation"
            manip = root / ds / "results" / "manipulations"
            ev.mkdir(parents=True)
            manip.mkdir(parents=True)

            df = pd.DataFrame(
                {
                    "dataset_id": [ds],
                    "model": ["pca"],
                    "intervention_id": ["down_a"],
                    "intervention_name": ["downsample"],
                    "metric_category": ["knn_metrics"],
                    "metric_name": ["knn_recall"],
                    "space": ["embedding"],
                    "value_mean": [0.5],
                    "value_std": [0.1],
                    "null_value": [0.2],
                    "k": [15],
                    "n_cells": [100],
                    "seed": [0],
                }
            )
            df.to_csv(ev / "pca_metrics.csv", index=False)

            out_root = root / "bundles"
            export_dataset_bundle(ds, root / ds, out_root, compression="snappy")

            bundle_dir = out_root / ds
            self.assertTrue(is_bundle_dataset_dir(bundle_dir))
            self.assertTrue((bundle_dir / METRICS_FILENAME).is_file())
            self.assertTrue((bundle_dir / SUMMARY_FILENAME).is_file())

            loaded = load_metrics_table(ds, out_root)
            self.assertEqual(len(loaded), 1)
            self.assertAlmostEqual(float(loaded["value_mean"].iloc[0]), 0.5)

            summary = json.loads((bundle_dir / SUMMARY_FILENAME).read_text())
            self.assertEqual(summary["dataset_id"], ds)

    def test_mixed_param_value_exports_to_parquet(self) -> None:
        from metrics_dashboard.bundle import coerce_metrics_for_parquet

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = "mixed"
            ev = root / ds / "results" / "evaluation"
            ev.mkdir(parents=True)

            rows = [
                {
                    "dataset_id": ds,
                    "model": "pca",
                    "intervention_id": "down_a",
                    "intervention_name": "downsample",
                    "metric_category": "knn_metrics",
                    "metric_name": "knn_recall",
                    "space": "embedding",
                    "value_mean": 0.5,
                    "value_std": 0.1,
                    "k": 15,
                    "n_cells": 10,
                    "seed": 0,
                    "param_key": "fraction",
                    "param_value": 0.5,
                },
                {
                    "dataset_id": ds,
                    "model": "pca",
                    "intervention_id": "shuffle_a",
                    "intervention_name": "gene_shuffle",
                    "metric_category": "knn_metrics",
                    "metric_name": "knn_recall",
                    "space": "embedding",
                    "value_mean": 0.3,
                    "value_std": 0.1,
                    "k": 15,
                    "n_cells": 10,
                    "seed": 0,
                    "param_key": "variant",
                    "param_value": "chromosome",
                },
            ]
            pd.DataFrame(rows).to_csv(ev / "pca_metrics.csv", index=False)

            out = export_dataset_bundle(ds, root / ds, root / "bundles", compression="snappy")
            loaded = pd.read_parquet(out / METRICS_FILENAME)
            variants = set(loaded["param_value"].astype(str))
            self.assertIn("0.5", variants)
            self.assertIn("chromosome", variants)


if __name__ == "__main__":
    unittest.main()
