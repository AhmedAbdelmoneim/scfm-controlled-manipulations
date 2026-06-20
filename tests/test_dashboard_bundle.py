"""Tests for dashboard bundle export/load."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from metrics_dashboard.bundle import (
    METRICS_FILENAME,
    SUMMARY_FILENAME,
    export_dataset_bundle,
    is_bundle_dataset_dir,
    load_metrics_table,
)
import pandas as pd


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
                    "metric_name": ["trustworthiness"],
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
            pd.DataFrame(
                {
                    "dataset_id": [ds],
                    "model": ["pca"],
                    "intervention_id": ["reference"],
                    "intervention_name": ["reference"],
                    "metric_category": ["bio_conservation_metrics"],
                    "metric_name": ["silhouette_label"],
                    "space": ["embedding_reference"],
                    "value_mean": [0.9],
                    "value_std": [0.0],
                    "n_cells": [100],
                    "seed": [0],
                }
            ).to_csv(ev / "pca_scib_metrics.csv", index=False)

            out_root = root / "bundles"
            export_dataset_bundle(ds, root / ds, out_root, compression="snappy")

            bundle_dir = out_root / ds
            self.assertTrue(is_bundle_dataset_dir(bundle_dir))
            self.assertTrue((bundle_dir / METRICS_FILENAME).is_file())
            self.assertTrue((bundle_dir / SUMMARY_FILENAME).is_file())

            loaded = load_metrics_table(ds, out_root)
            self.assertEqual(len(loaded), 1)
            self.assertAlmostEqual(float(loaded["value_mean"].iloc[0]), 0.5)
            self.assertNotIn(
                "bio_conservation_metrics", set(loaded["metric_category"].astype(str))
            )

            summary = json.loads((bundle_dir / SUMMARY_FILENAME).read_text())
            self.assertEqual(summary["dataset_id"], ds)
            manifest = json.loads((bundle_dir / "manifest.json").read_text())
            self.assertNotIn("pca_scib_metrics.csv", manifest["source_files_mtime_ns"])

    def test_mixed_param_value_exports_to_parquet(self) -> None:

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
                    "metric_name": "trustworthiness",
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
                    "metric_name": "trustworthiness",
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
