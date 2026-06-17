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
    filter_cv_envelope_metrics,
    filter_stability_plot_metrics,
    find_stabilization_n,
    normalize_aggregate_by_cell_count,
    _cv_percentile_by_cell_count,
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
                "metric_name": ["trustworthiness"] * 4,
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
                "metric_name": ["trustworthiness", "ari", "trustworthiness", "ari"],
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
                "atlas": ["immune"] * 12,
                "cell_count": [200, 500] * 6,
                "model": ["scimilarity"] * 12,
                "metric_category": (
                    ["structure_metrics"] * 6
                    + ["clustering_metrics"] * 2
                    + ["bio_conservation_metrics"] * 2
                    + ["batch_correction_metrics"] * 2
                ),
                "metric_name": [
                    "viscore_local_sp",
                    "viscore_local_sp",
                    "distcorr",
                    "distcorr",
                    "viscore_global_sp",
                    "viscore_global_sp",
                    "leiden_ari",
                    "leiden_ari",
                    "silhouette_label",
                    "silhouette_label",
                    "ilisi_knn",
                    "ilisi_knn",
                ],
                "space": ["embedding"] * 8 + ["embedding_manipulated"] * 4,
                "k": [-1.0] * 12,
                "diffusion_t": [-1.0] * 12,
                "resolution": [-1.0] * 10 + [1.0, 1.0],
                "distance_metric": ["euclidean"] * 12,
                "n_seeds": [5] * 12,
                "mean_across_seeds": np.linspace(0.1, 0.8, 12),
                "std_across_seeds": [0.01] * 12,
                "cv_across_seeds": [0.05] * 12,
                "ci_half_width": [0.02] * 12,
                "ci_half_width_rel": [0.03] * 12,
            }
        )
        plots = filter_stability_plot_metrics(agg)
        self.assertGreaterEqual(plots["plot_key"].nunique(), 5)
        self.assertIn("viscore_local_sp", set(plots["plot_key"]))
        self.assertIn("distcorr", set(plots["plot_key"]))
        self.assertIn("silhouette_label", set(plots["plot_key"]))

    def test_filter_cv_envelope_metrics(self) -> None:
        df = pd.DataFrame(
            {
                "space": ["embedding", "embedding", "embedding", "embedding"],
                "metric_name": [
                    "viscore_local_sp",
                    "distcorr",
                    "leiden_ari",
                    "rnx_curve",
                ],
            }
        )
        out = filter_cv_envelope_metrics(df)
        self.assertEqual(
            list(out["metric_name"]),
            ["viscore_local_sp", "distcorr", "leiden_ari"],
        )

    def test_cv_percentile_by_cell_count(self) -> None:
        df = pd.DataFrame(
            {
                "cell_count": [200, 200, 200, 500, 500],
                "cv_across_seeds": [0.1, 0.2, 0.3, 0.4, 0.6],
            }
        )
        p90 = _cv_percentile_by_cell_count(df, 0.9)
        self.assertEqual(list(p90["cell_count"]), [200, 500])
        self.assertAlmostEqual(float(p90.loc[p90["cell_count"] == 200, "cv_quantile"]), 0.28)
        self.assertAlmostEqual(float(p90.loc[p90["cell_count"] == 500, "cv_quantile"]), 0.58)

    def test_normalize_aggregate_by_cell_count(self) -> None:
        rows = pd.DataFrame(
            {
                "cell_count": [200, 1000],
                "mean_across_seeds": [0.4, 0.5],
                "std_across_seeds": [0.04, 0.05],
                "ci_half_width": [0.02, 0.025],
                "cv_across_seeds": [0.1, 0.1],
            }
        )
        norm = normalize_aggregate_by_cell_count(rows)
        self.assertAlmostEqual(norm.loc[0, "mean_across_seeds"], 0.4 / 200)
        self.assertAlmostEqual(norm.loc[1, "mean_across_seeds"], 0.5 / 1000)
        self.assertAlmostEqual(norm.loc[0, "std_across_seeds"], 0.04 / 200)


if __name__ == "__main__":
    unittest.main()
