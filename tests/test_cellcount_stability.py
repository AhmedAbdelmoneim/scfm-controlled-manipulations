"""Tests for cell-count stability analysis helpers."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "lib"))

from cellcount_sweep import (  # noqa: E402
    full_atlas_path,
    parse_run_dir,
    resolve_atlas_source,
    stratified_subsample_indices,
    sweep_paths,
    SIZE_SWEEP_PROCESSED_SUBDIR,
    SIZE_SWEEP_RAW_SUBDIR,
)

sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "metrics_dashboard"))
from analyze_cellcount_stability import (  # noqa: E402
    aggregate_across_seeds,
    filter_stability_plot_metrics,
    find_stabilization_n,
)


class TestCellcountStability(unittest.TestCase):
    def test_full_atlas_path(self) -> None:
        path = full_atlas_path("brain", "/vault/amoneim/atlases")
        self.assertEqual(path.name, "human_brain_cell_atlas.h5ad")

    def test_resolve_atlas_source_full_vs_legacy(self) -> None:
        full = resolve_atlas_source(
            "immune", source_atlases_dir="/vault/amoneim/atlases"
        )
        self.assertEqual(full.name, "human_immune_health_atlas.h5ad")
        legacy = resolve_atlas_source("immune", source_raw_dir="/data/raw_datasets")
        self.assertEqual(legacy, Path("/data/raw_datasets/immune.h5ad"))

    def test_stratified_subsample_indices(self) -> None:
        obs = pd.DataFrame(
            {
                "cell_type": ["A"] * 80 + ["B"] * 20,
            },
            index=[f"c{i}" for i in range(100)],
        )
        idx = stratified_subsample_indices(obs, n_cells=50, seed=0, cell_type_col="cell_type")
        self.assertEqual(len(idx), 50)
        picked = obs.loc[idx, "cell_type"].value_counts()
        self.assertEqual(picked["A"], 40)
        self.assertEqual(picked["B"], 10)

    def test_subsample_atlas_excludes_zero_raw_counts(self) -> None:
        import tempfile

        import anndata as ad
        from cellcount_sweep import subsample_atlas

        raw = ad.AnnData(sp.csr_matrix([[1.0, 0.0], [0.0, 0.0], [2.0, 1.0], [3.0, 0.0]]))
        source = ad.AnnData(
            X=np.zeros((4, 2), dtype=np.float32),
            obs=pd.DataFrame({"cell_type": ["A", "A", "B", "B"]}, index=[f"c{i}" for i in range(4)]),
        )
        source.raw = raw
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source.h5ad"
            out_path = Path(tmp) / "out.h5ad"
            source.write(source_path)

            n_written = subsample_atlas(
                source_path,
                out_path,
                n_cells=2,
                seed=0,
                stratified=False,
            )
            self.assertEqual(n_written, 2)
            out = ad.read_h5ad(out_path)
            self.assertEqual(out.n_obs, 2)
            raw_sums = np.asarray(out.raw.X.sum(axis=1)).ravel()
            self.assertTrue(np.all(raw_sums > 0))

    def test_subsample_immune_promotes_raw_counts_to_x(self) -> None:
        import tempfile

        import anndata as ad
        from cellcount_sweep import subsample_atlas

        raw = ad.AnnData(sp.csr_matrix([[5.0, 0.0], [0.0, 0.0], [2.0, 1.0], [3.0, 0.0]]))
        source = ad.AnnData(
            X=np.array([[-1.0, 0.5], [0.0, 0.0], [1.0, 2.0], [0.5, 0.5]], dtype=np.float32),
            obs=pd.DataFrame({"cell_type": ["A", "A", "B", "B"]}, index=[f"c{i}" for i in range(4)]),
        )
        source.raw = raw
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source.h5ad"
            out_path = Path(tmp) / "out.h5ad"
            source.write(source_path)

            n_written = subsample_atlas(
                source_path,
                out_path,
                n_cells=2,
                seed=0,
                stratified=False,
                atlas="immune",
            )
            self.assertEqual(n_written, 2)
            out = ad.read_h5ad(out_path)
            self.assertIsNone(out.raw)
            x_sums = np.asarray(out.X.sum(axis=1)).ravel()
            self.assertTrue(np.all(x_sums > 0))

    def test_size_sweep_paths(self) -> None:
        paths = sweep_paths(
            vault_root=Path("/vault/proj"),
            atlas="lung",
            cell_count=2000,
            seed=3,
            raw_subdir=SIZE_SWEEP_RAW_SUBDIR,
            processed_subdir=SIZE_SWEEP_PROCESSED_SUBDIR,
        )
        self.assertIn("raw_datasets_size_sweep", str(paths["input_h5ad"]))
        self.assertIn("processed_size_sweep", str(paths["processed_root"]))

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

    def test_aggregate_with_sparse_hyperparams(self) -> None:
        """Optional metric dims are NaN on unrelated rows; aggregation must still work."""
        metrics = pd.DataFrame(
            {
                "atlas": ["immune"] * 4,
                "cell_count": [200, 200, 500, 500],
                "model": ["scimilarity"] * 4,
                "metric_category": ["knn_metrics", "clustering", "knn_metrics", "clustering"],
                "metric_name": ["knn_recall", "ari", "knn_recall", "ari"],
                "space": ["embedding"] * 4,
                "k": [15.0, np.nan, 15.0, np.nan],
                "resolution": [np.nan, 1.0, np.nan, 1.0],
                "value_mean": [0.5, 0.52, 0.8, 0.81],
                "sweep_seed": [0, 1, 0, 1],
            }
        )
        agg = aggregate_across_seeds(metrics)
        self.assertGreater(len(agg), 0)
        self.assertIn("atlas", agg.columns)

    def test_filter_stability_plot_metrics(self) -> None:
        agg = pd.DataFrame(
            {
                "atlas": ["immune"] * 14,
                "cell_count": [200, 500] * 7,
                "model": ["scimilarity"] * 14,
                "metric_category": (
                    ["knn_metrics"] * 8
                    + ["clustering_metrics"] * 2
                    + ["cell_type_and_batch_metrics"] * 4
                ),
                "metric_name": [
                    "diffusion_sym_kl",
                    "diffusion_sym_kl",
                    "diffusion_sym_kl",
                    "diffusion_sym_kl",
                    "diffusion_js",
                    "diffusion_js",
                    "knn_recall",
                    "knn_recall",
                    "leiden_ari",
                    "leiden_ari",
                    "cell_type_asw",
                    "cell_type_asw",
                    "batch_ilisi",
                    "batch_ilisi",
                ],
                "space": ["embedding"] * 10 + ["embedding_manipulated"] * 4,
                "k": [15.0] * 14,
                "diffusion_t": [1.0, 2.0, 4.0, 8.0, 1.0, 2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                "resolution": [-1.0] * 12 + [1.0, 1.0],
                "distance_metric": ["euclidean"] * 14,
                "n_seeds": [5] * 14,
                "mean_across_seeds": np.linspace(0.1, 0.8, 14),
                "std_across_seeds": [0.01] * 14,
                "cv_across_seeds": [0.05] * 14,
                "ci_half_width": [0.02] * 14,
                "ci_half_width_rel": [0.03] * 14,
            }
        )
        plots = filter_stability_plot_metrics(agg)
        # knn + clustering + 4 KL t + 2 JS t + 2 cell/batch (no graph_connectivity in fixture)
        self.assertEqual(plots["plot_key"].nunique(), 10)
        self.assertIn("kl_divergence_t8", set(plots["plot_key"]))
        self.assertIn("cell_type_asw", set(plots["plot_key"]))


if __name__ == "__main__":
    unittest.main()
