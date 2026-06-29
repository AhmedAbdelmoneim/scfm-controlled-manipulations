"""Lightweight tests (stdlib ``unittest`` — no pytest dependency)."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any
import unittest

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scfm_controlled_manipulations.io import embedding_path, manipulation_path, manipulations_dir
from scfm_controlled_manipulations.sweep_config import expand_intervention_specs


class PathLayoutTest(unittest.TestCase):
    def test_embedding_and_manipulation_paths(self) -> None:
        emb = embedding_path("/data/embeddings", "pca", "downsample_6e9bcd431d63")
        self.assertEqual(
            emb,
            Path("/data/embeddings/pca/pca_downsample_6e9bcd431d63.h5ad"),
        )
        ref_emb = embedding_path("/data/embeddings", "geneformer", "reference")
        self.assertEqual(
            ref_emb,
            Path("/data/embeddings/geneformer/geneformer_reference.h5ad"),
        )
        man = manipulation_path("/data/results", "gene_shuffle_d9d8843bc9e3")
        self.assertEqual(
            man,
            Path("/data/results/manipulations/gene_shuffle_d9d8843bc9e3.h5ad"),
        )
        self.assertEqual(
            manipulations_dir("/data/results", "/data/manipulations"),
            Path("/data/manipulations"),
        )
        custom_man = manipulation_path(
            "/data/results",
            "gene_shuffle_d9d8843bc9e3",
            "/data/manipulations",
        )
        self.assertEqual(
            custom_man,
            Path("/data/manipulations/gene_shuffle_d9d8843bc9e3.h5ad"),
        )

    def test_run_evaluate_skips_model_with_missing_reference_embedding(self) -> None:
        from scfm_controlled_manipulations.evaluation.run import run_evaluate
        from scfm_controlled_manipulations.io import intervention_id

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_dir = root / "results"
            manip_dir = results_dir / "manipulations"
            emb_root = root / "embeddings"
            manip_dir.mkdir(parents=True)
            (emb_root / "pca").mkdir(parents=True)

            obs_names = ["cell_0", "cell_1", "cell_2"]
            obs = pd.DataFrame(index=obs_names)
            var = pd.DataFrame(index=["gene_0", "gene_1"])
            counts = sp.csr_matrix(np.ones((3, 2), dtype=np.float32))
            ad_ref = ad.AnnData(X=counts, obs=obs.copy(), var=var.copy())
            ad_ref.write_h5ad(manip_dir / "reference.h5ad")

            specs = [{"name": "downsample", "kwargs": {"fraction": 0.5}}]
            iid = intervention_id("downsample", {"fraction": 0.5})
            ad_man = ad.AnnData(X=counts.copy(), obs=obs.copy(), var=var.copy())
            ad_man.write_h5ad(manip_dir / f"{iid}.h5ad")
            ad_emb_man = ad.AnnData(
                X=np.ones((3, 2), dtype=np.float32),
                obs=obs.copy(),
            )
            ad_emb_man.write_h5ad(emb_root / "pca" / f"pca_{iid}.h5ad")

            cfg = {
                "results_dir": str(results_dir),
                "embeddings_root": str(emb_root),
                "interventions": specs,
                "models": ["pca"],
                "reference_intervention_id": "reference",
                "seed": 0,
                "evaluation": {
                    "dataset_id": "toy",
                    "evaluation_workers": 1,
                },
            }
            run_evaluate(cfg)
            self.assertFalse((results_dir / "evaluation" / "pca_metrics.csv").exists())


class SweepTest(unittest.TestCase):
    def test_expand_cartesian(self) -> None:
        specs = [
            {"name": "downsample", "kwargs": {"fraction": [0.5, 0.9]}},
        ]
        out = expand_intervention_specs(specs)
        self.assertEqual(len(out), 2)
        self.assertEqual({o["kwargs"]["fraction"] for o in out}, {0.5, 0.9})


class LeidenMpSafetyTest(unittest.TestCase):
    def test_scanpy_leiden_kwargs_uses_igraph(self) -> None:
        from scfm_controlled_manipulations.evaluation.leiden_cache import scanpy_leiden_kwargs

        kw = scanpy_leiden_kwargs()
        self.assertEqual(kw["flavor"], "igraph")
        self.assertIn("n_iterations", kw)

    def test_leiden_labels_runs_in_process(self) -> None:
        from unittest import mock

        import numpy as np

        from scfm_controlled_manipulations.evaluation.leiden_cache import leiden_labels_for_matrix

        mat = np.random.default_rng(0).standard_normal((40, 8)).astype(np.float32)
        fake_labels = np.array(["0", "1"] * 20)

        with mock.patch(
            "scfm_controlled_manipulations.evaluation.leiden_cache._leiden_labels_compute",
            return_value=fake_labels,
        ) as compute:
            out = leiden_labels_for_matrix(mat, k=5, metric="euclidean", resolution=0.5, seed=0)

        self.assertTrue(np.array_equal(out, fake_labels))
        compute.assert_called_once()

    def test_evaluation_pool_is_spawn(self) -> None:
        import multiprocessing as mp

        self.assertEqual(mp.get_context("spawn").get_start_method(), "spawn")


class CellBatchMetricsTest(unittest.TestCase):
    @staticmethod
    def _toy_bundle(n: int = 80) -> Any:
        from types import SimpleNamespace

        rng = np.random.default_rng(11)
        emb = rng.standard_normal((n, 12)).astype(np.float32)
        obs = pd.DataFrame(
            {
                "cell_type": rng.choice(["A", "B"], size=n),
                "batch": rng.choice(["b1", "b2"], size=n),
            }
        )
        return SimpleNamespace(
            emb_ref=emb,
            emb_man=emb + 0.01,
            obs=obs,
        )

    @staticmethod
    def _write_counts_fixture(tmp: Path, *, n: int, intervention_id: str) -> None:
        manip_dir = tmp / "manipulations"
        manip_dir.mkdir(parents=True)
        rng = np.random.default_rng(11)
        counts = sp.csr_matrix(rng.poisson(1, (n, 20)).astype(np.float32))
        ad.AnnData(X=counts).write_h5ad(manip_dir / f"{intervention_id}.h5ad")

    def test_benchmarker_to_rows_flattens_categories(self) -> None:
        from scib_metrics.benchmark._core import _METRIC_TYPE

        from scfm_controlled_manipulations.evaluation.metrics_cell_batch import (
            BATCH_CATEGORY,
            BIO_CATEGORY,
            _benchmarker_to_rows,
        )

        results = pd.DataFrame(
            {
                "embedding": [0.4, 0.5, 0.6],
                _METRIC_TYPE: [
                    "Bio conservation",
                    "Batch correction",
                    "Bio conservation",
                ],
            },
            index=["silhouette_label", "ilisi_knn", "graph_connectivity"],
        )
        rows = _benchmarker_to_rows(
            results,
            dataset_id="toy",
            model="m",
            intervention_id="i1",
            intervention_name="n",
            space_label="embedding_manipulated",
            seed=0,
            n_cells=80,
        )
        categories = {row["metric_category"] for row in rows}
        self.assertEqual(categories, {BIO_CATEGORY, BATCH_CATEGORY})

    def test_skips_when_columns_missing(self) -> None:
        from scfm_controlled_manipulations.evaluation.data import load_manipulation_counts
        from scfm_controlled_manipulations.evaluation.metrics_cell_batch import (
            compute_cell_batch_reference_rows,
        )

        bundle = self._toy_bundle()
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            self._write_counts_fixture(results_dir, n=80, intervention_id="reference")
            rows = compute_cell_batch_reference_rows(
                counts=load_manipulation_counts(results_dir, "reference"),
                mat=bundle.emb_ref,
                obs_df=bundle.obs,
                space_label="embedding_reference",
                dataset_id="toy",
                model="m",
                intervention_id="reference",
                intervention_name="reference",
                seed=0,
                cell_type_col=None,
                batch_col="batch",
                n_cells=80,
            )
            self.assertEqual(rows, [])

    def test_reference_rows_use_reference_intervention_id(self) -> None:
        from unittest import mock

        from scfm_controlled_manipulations.evaluation.data import load_manipulation_counts
        from scfm_controlled_manipulations.evaluation.metrics_cell_batch import (
            compute_cell_batch_reference_rows,
        )

        bundle = self._toy_bundle(n=60)
        fake_rows = [
            {
                "dataset_id": "toy",
                "model": "m",
                "intervention_id": "reference",
                "intervention_name": "reference",
                "metric_category": "bio_conservation_metrics",
                "metric_name": "silhouette_label",
                "space": "embedding_reference",
                "value_mean": 0.4,
                "n_cells": 60,
                "seed": 0,
            }
        ]
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch(
                "scfm_controlled_manipulations.evaluation.metrics_cell_batch._run_benchmarker_rows",
                return_value=fake_rows,
            ),
        ):
            results_dir = Path(tmp)
            self._write_counts_fixture(results_dir, n=60, intervention_id="reference")
            rows = compute_cell_batch_reference_rows(
                counts=load_manipulation_counts(results_dir, "reference"),
                mat=bundle.emb_ref,
                obs_df=bundle.obs,
                space_label="embedding_reference",
                dataset_id="toy",
                model="m",
                intervention_id="reference",
                intervention_name="reference",
                seed=0,
                cell_type_col="cell_type",
                batch_col="batch",
                n_cells=60,
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["intervention_id"], "reference")
        self.assertEqual(rows[0]["space"], "embedding_reference")

    def test_scib_benchmark_returns_finite_values(self) -> None:
        from scfm_controlled_manipulations.evaluation.data import load_manipulation_counts
        from scfm_controlled_manipulations.evaluation.metrics_cell_batch import (
            compute_cell_batch_reference_rows,
        )

        bundle = self._toy_bundle(n=200)
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            self._write_counts_fixture(results_dir, n=200, intervention_id="reference")
            rows = compute_cell_batch_reference_rows(
                counts=load_manipulation_counts(results_dir, "reference"),
                mat=bundle.emb_ref,
                obs_df=bundle.obs,
                space_label="embedding_reference",
                dataset_id="toy",
                model="m",
                intervention_id="reference",
                intervention_name="reference",
                seed=0,
                cell_type_col="cell_type",
                batch_col="batch",
                n_cells=200,
            )
            metric_names = {row["metric_name"] for row in rows}
            self.assertIn("silhouette_label", metric_names)
            self.assertIn("ilisi_knn", metric_names)
            values = [row["value_mean"] for row in rows]
            finite = sum(1 for v in values if np.isfinite(v))
            self.assertGreater(finite, 0)


class FilterZeroCountCellsTest(unittest.TestCase):
    def test_removes_zero_count_cells_and_logs_count(self) -> None:
        from scfm_controlled_manipulations.qc import filter_zero_count_cells

        X = sp.csr_matrix(
            [
                [1.0, 0.0],
                [0.0, 0.0],
                [2.0, 3.0],
            ]
        )
        adata = ad.AnnData(X=X, obs={"cell_id": ["a", "b", "c"]})
        kept = filter_zero_count_cells(adata)
        self.assertEqual(kept, 2)
        self.assertEqual(list(adata.obs["cell_id"]), ["a", "c"])

    def test_noop_when_all_nonzero(self) -> None:
        from scfm_controlled_manipulations.qc import filter_zero_count_cells

        adata = ad.AnnData(X=sp.csr_matrix([[1.0, 2.0], [3.0, 0.0]]))
        kept = filter_zero_count_cells(adata)
        self.assertEqual(kept, 2)
        self.assertEqual(adata.n_obs, 2)

    def test_uses_raw_counts_when_present(self) -> None:
        from scfm_controlled_manipulations.qc import filter_zero_count_cells

        X = np.array([[-1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
        raw = ad.AnnData(sp.csr_matrix([[5.0, 0.0], [1.0, 2.0]]))
        adata = ad.AnnData(X=X)
        adata.raw = raw
        kept = filter_zero_count_cells(adata)
        self.assertEqual(kept, 2)
        self.assertEqual(adata.n_obs, 2)


class StructureMetricsTest(unittest.TestCase):
    def test_identity_pair_scores_perfect(self) -> None:
        import json

        from scfm_controlled_manipulations.evaluation.metrics_structure import (
            compute_structure_metrics,
        )

        rng = np.random.default_rng(0)
        ref = rng.standard_normal((80, 12))
        bundle = type(
            "Bundle",
            (),
            {
                "emb_ref": ref.copy(),
                "emb_man": ref.copy(),
            },
        )()
        df = compute_structure_metrics(
            bundle=bundle,
            dataset_id="test",
            model="pca",
            intervention_id="iid",
            intervention_name="reference",
            seed=0,
        )
        emb_sl = df[(df["space"] == "embedding") & (df["metric_name"] == "viscore_local_sp")]
        self.assertEqual(len(emb_sl), 1)
        self.assertAlmostEqual(float(emb_sl.iloc[0]["value_mean"]), 1.0, places=5)

        emb_dc = df[(df["space"] == "embedding") & (df["metric_name"] == "distcorr")]
        self.assertAlmostEqual(float(emb_dc.iloc[0]["value_mean"]), 1.0, places=5)

        curve = df[(df["space"] == "embedding") & (df["metric_name"] == "rnx_curve")]
        self.assertEqual(len(curve), 1)
        payload = json.loads(str(curve.iloc[0]["rnx_curve_json"]))
        self.assertIn("rnx", payload)
        self.assertGreater(len(payload["rnx"]), 0)

        twonn = df[
            (df["space"] == "embedding")
            & (df["metric_name"] == "intrinsic_dim_twonn")
            & (df["side"] == "ref")
        ]
        self.assertTrue(np.isfinite(float(twonn.iloc[0]["value_mean"])))


class EmbeddingAlignmentTest(unittest.TestCase):
    def test_dense_embedding_aligned_to_obs_reorders_rows(self) -> None:
        import anndata as ad

        from scfm_controlled_manipulations.evaluation.data import (
            dense_embedding_aligned_to_obs,
        )

        target_obs = ["c", "a", "b"]
        adata = ad.AnnData(
            X=np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
            obs={"cell_id": ["a", "b", "c"]},
        )
        adata.obs_names = ["a", "b", "c"]
        aligned = dense_embedding_aligned_to_obs(adata, pd.Index(target_obs), label="emb")
        np.testing.assert_allclose(aligned.ravel(), [3.0, 1.0, 2.0])

    def test_dense_embedding_aligned_to_obs_rejects_missing_cells(self) -> None:
        import anndata as ad

        from scfm_controlled_manipulations.evaluation.data import (
            dense_embedding_aligned_to_obs,
        )

        adata = ad.AnnData(
            X=np.array([[1.0], [2.0]], dtype=np.float32),
            obs={"cell_id": ["a", "b"]},
        )
        with self.assertRaises(ValueError):
            dense_embedding_aligned_to_obs(adata, pd.Index(["a", "b", "c"]), label="emb")


class DistributionSummaryTest(unittest.TestCase):
    def test_known_array_quantiles(self) -> None:
        from scfm_controlled_manipulations.evaluation.metrics_common import (
            distribution_summary,
            scalar_summary,
        )

        summary = distribution_summary(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
        self.assertAlmostEqual(summary.mean, 2.0, places=6)
        self.assertAlmostEqual(summary.median, 2.0, places=6)
        self.assertAlmostEqual(summary.min, 0.0, places=6)
        self.assertAlmostEqual(summary.max, 4.0, places=6)
        self.assertAlmostEqual(summary.q05, 0.2, places=6)
        self.assertAlmostEqual(summary.q95, 3.8, places=6)

        empty = distribution_summary(np.array([]))
        self.assertTrue(np.isnan(empty.mean))
        scalar = scalar_summary(3.5)
        self.assertAlmostEqual(scalar.mean, 3.5, places=6)
        self.assertTrue(np.isnan(scalar.q25))


class StatsShiftMetricsTest(unittest.TestCase):
    def _toy_bundle(self):
        from scfm_controlled_manipulations.evaluation.data import AlignedBundle

        ref = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        man = ref + np.array([1.0, 0.0], dtype=np.float32)
        return AlignedBundle(
            emb_ref=ref.copy(),
            emb_man=man.copy(),
            obs=pd.DataFrame(index=["a", "b", "c"]),
        )

    def test_paired_shift_and_pairwise_cosine_metrics(self) -> None:
        from scfm_controlled_manipulations.evaluation.metrics_stats_shift import (
            compute_embedding_shift,
        )

        bundle = self._toy_bundle()
        df = compute_embedding_shift(
            bundle=bundle,
            dataset_id="toy",
            model="m",
            intervention_id="i1",
            intervention_name="n",
            seed=0,
            ref_cache=None,
            pairwise_max_pairs=None,
        )
        paired = df[
            (df["metric_name"] == "paired_cell_l2_norm") & (df["space"] == "embedding")
        ].iloc[0]
        self.assertAlmostEqual(paired["value_mean"], 1.0, places=5)
        cos_row = df[
            (df["metric_name"] == "shift_pairwise_cosine") & (df["space"] == "embedding")
        ].iloc[0]
        self.assertAlmostEqual(cos_row["value_mean"], 1.0, places=5)
        self.assertNotIn("shift_dot_with_mean", df["metric_name"].values)

    def test_col_variance_distribution(self) -> None:
        from scfm_controlled_manipulations.evaluation.metrics_stats_shift import (
            compute_embedding_stats,
        )

        bundle = self._toy_bundle()
        df = compute_embedding_stats(
            bundle=bundle,
            dataset_id="toy",
            model="m",
            intervention_id="i1",
            intervention_name="n",
            seed=0,
        )
        col_ref = df[
            (df["metric_name"] == "col_variance_ref") & (df["space"] == "embedding")
        ].iloc[0]
        self.assertIn("value_q25", col_ref.index)
        self.assertFalse(np.isnan(col_ref["value_mean"]))

    def test_col_mean_distribution(self) -> None:
        from scfm_controlled_manipulations.evaluation.metrics_stats_shift import (
            _col_means_dense,
            compute_embedding_stats,
        )

        bundle = self._toy_bundle()
        expected_means = _col_means_dense(bundle.emb_ref)
        df = compute_embedding_stats(
            bundle=bundle,
            dataset_id="toy",
            model="m",
            intervention_id="i1",
            intervention_name="n",
            seed=0,
        )
        col_ref = df[(df["metric_name"] == "col_mean_ref") & (df["space"] == "embedding")].iloc[0]
        self.assertAlmostEqual(col_ref["value_mean"], float(np.mean(expected_means)), places=5)
        self.assertFalse(np.isnan(col_ref["value_mean"]))

    def test_reference_cache_idempotent(self) -> None:
        from scfm_controlled_manipulations.evaluation.context import (
            DatasetEvaluateContext,
            ModelEvaluateContext,
        )
        from scfm_controlled_manipulations.evaluation.data import AlignedBundle
        from scfm_controlled_manipulations.evaluation.metrics_stats_shift import (
            compute_embedding_shift,
            compute_embedding_stats,
        )
        from scfm_controlled_manipulations.evaluation.reference_stats_shift import (
            precompute_reference_stats_shift,
        )

        bundle = self._toy_bundle()
        dataset_ctx = DatasetEvaluateContext(
            obs=bundle.obs,
            n_cells=3,
        )
        model_ctx = ModelEvaluateContext(emb_ref=bundle.emb_ref)
        model_ctx.ref_stats_cache = precompute_reference_stats_shift(
            model_ctx,
            dataset_ctx,
            seed=0,
            pairwise_cell_subsample_n=3,
            pairwise_max_pairs=None,
        )

        man2 = bundle.emb_man + np.array([0.0, 1.0], dtype=np.float32)
        bundle2 = AlignedBundle(
            emb_ref=bundle.emb_ref,
            emb_man=man2,
            obs=bundle.obs,
        )

        df1 = compute_embedding_stats(
            bundle=bundle,
            dataset_id="toy",
            model="m",
            intervention_id="i1",
            intervention_name="n",
            seed=0,
            ref_cache=model_ctx.ref_stats_cache,
        )
        df2 = compute_embedding_stats(
            bundle=bundle2,
            dataset_id="toy",
            model="m",
            intervention_id="i2",
            intervention_name="n",
            seed=0,
            ref_cache=model_ctx.ref_stats_cache,
        )
        ref1 = df1[df1["metric_name"] == "mean_row_l2_norm_ref"]
        ref2 = df2[df2["metric_name"] == "mean_row_l2_norm_ref"]
        pd.testing.assert_frame_equal(
            ref1.drop(columns=["intervention_id"]),
            ref2.drop(columns=["intervention_id"]),
        )

        sh1 = compute_embedding_shift(
            bundle=bundle,
            dataset_id="toy",
            model="m",
            intervention_id="i1",
            intervention_name="n",
            seed=0,
            ref_cache=model_ctx.ref_stats_cache,
        )
        sh2 = compute_embedding_shift(
            bundle=bundle2,
            dataset_id="toy",
            model="m",
            intervention_id="i2",
            intervention_name="n",
            seed=0,
            ref_cache=model_ctx.ref_stats_cache,
        )
        w1 = sh1[sh1["metric_name"] == "within_ref_pairwise_l2"]["value_mean"].values
        w2 = sh2[sh2["metric_name"] == "within_ref_pairwise_l2"]["value_mean"].values
        np.testing.assert_allclose(w1, w2)

    def test_global_distance_correlation(self) -> None:
        from scfm_controlled_manipulations.evaluation.metrics_stats_shift import (
            compute_embedding_shift,
            global_distance_correlation,
        )

        ref = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        man = ref.copy()
        self.assertAlmostEqual(
            global_distance_correlation(ref, man, metric="euclidean"), 1.0, places=5
        )
        warped = man.copy()
        warped[0] += np.array([2.0, 0.0], dtype=np.float64)
        self.assertLess(global_distance_correlation(ref, warped, metric="euclidean"), 1.0)

        bundle = self._toy_bundle()
        df = compute_embedding_shift(
            bundle=bundle,
            dataset_id="toy",
            model="m",
            intervention_id="i1",
            intervention_name="n",
            seed=0,
            distance_correlation_subsample_n=3,
            distance_metrics=["euclidean"],
        )
        dist_rows = df[df["metric_name"] == "global_distance_correlation"]
        self.assertEqual(len(dist_rows), 1)
        self.assertEqual(dist_rows.iloc[0]["space"], "embedding")
        self.assertTrue(np.isfinite(dist_rows.iloc[0]["value_mean"]))


class RunEvaluateScibTest(unittest.TestCase):
    def test_run_evaluate_scib_writes_separate_csv(self) -> None:
        import sys

        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from scripts.benchmark_eval import setup_fixture

        from scfm_controlled_manipulations.evaluation.run_scib import run_evaluate_scib
        from scfm_controlled_manipulations.io import evaluation_scib_metrics_csv_path

        config_path = root / "configs" / "experiments" / "atlases.yaml"
        if not config_path.exists():
            config_path = root / "configs" / "default.yaml"
        with tempfile.TemporaryDirectory() as tmp:
            fixture_root = Path(tmp)
            run_cfg = setup_fixture(
                fixture_root,
                n_cells=200,
                n_genes=400,
                emb_dim=32,
                n_cell_types=3,
                n_batches=2,
                config_path=config_path,
                models=["pca"],
                max_interventions=1,
            )
            run_evaluate_scib(run_cfg)
            out_path = evaluation_scib_metrics_csv_path(run_cfg["results_dir"], "pca")
            self.assertTrue(out_path.is_file())
            df = pd.read_csv(out_path)
            self.assertFalse(df.empty)
            self.assertTrue((df["intervention_id"] == "reference").all())
            self.assertTrue((df["space"] == "embedding_reference").all())
            categories = set(df["metric_category"].unique())
            self.assertIn("bio_conservation_metrics", categories)
            self.assertIn("batch_correction_metrics", categories)


class TrajectoryMetricsTest(unittest.TestCase):
    def test_trajectory_rows_include_observed_and_null_summary(self) -> None:
        from scfm_controlled_manipulations.evaluation.metrics_trajectory import (
            compute_trajectory_reference_rows,
        )

        n_cells = 30
        trajectory = np.repeat(np.arange(3), 10)
        obs = pd.DataFrame({"trajectory": trajectory}, index=[f"cell_{i}" for i in range(n_cells)])
        emb = np.column_stack(
            [
                np.linspace(0.0, 1.0, n_cells),
                np.zeros(n_cells),
            ]
        ).astype(np.float32)

        rows = compute_trajectory_reference_rows(
            mat=emb,
            obs_df=obs,
            trajectory_key="trajectory",
            dataset_id="toy",
            model="pca",
            intervention_id="reference",
            intervention_name="reference",
            space_label="embedding_reference",
            seed=0,
            n_neighbors=5,
            n_dcs=3,
            n_permutations=2,
        )

        df = pd.DataFrame(rows)
        self.assertEqual(
            set(df["metric_name"]), {"ordering_correlation_spearman", "frac_connected"}
        )
        score = df[df["metric_name"] == "ordering_correlation_spearman"].iloc[0]
        self.assertTrue(np.isfinite(score["value_mean"]))
        self.assertTrue(np.isfinite(score["null_mean"]))
        self.assertTrue(np.isfinite(score["null_z"]))
        self.assertTrue(np.isfinite(score["null_p_value"]))
        self.assertEqual(score["metric_category"], "trajectory_metrics")
        self.assertEqual(score["space"], "embedding_reference")


class RunEvaluateTrajectoryTest(unittest.TestCase):
    def test_run_evaluate_trajectory_writes_separate_csv(self) -> None:
        import sys

        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from scripts.benchmark_eval import setup_fixture

        from scfm_controlled_manipulations.evaluation.run_trajectory import run_evaluate_trajectory
        from scfm_controlled_manipulations.io import evaluation_trajectory_metrics_csv_path

        config_path = root / "configs" / "experiments" / "atlases.yaml"
        if not config_path.exists():
            config_path = root / "configs" / "default.yaml"
        with tempfile.TemporaryDirectory() as tmp:
            fixture_root = Path(tmp)
            run_cfg = setup_fixture(
                fixture_root,
                n_cells=60,
                n_genes=120,
                emb_dim=4,
                n_cell_types=3,
                n_batches=2,
                config_path=config_path,
                models=["pca"],
                max_interventions=1,
            )
            run_cfg["evaluation"]["trajectory_n_neighbors"] = 5
            run_cfg["evaluation"]["trajectory_n_dcs"] = 3
            run_cfg["evaluation"]["trajectory_n_permutations"] = 2

            ref_path = Path(run_cfg["results_dir"]) / "manipulations" / "reference.h5ad"
            ad_ref = ad.read_h5ad(ref_path)
            trajectory = np.repeat(np.arange(3), ad_ref.n_obs // 3)
            if trajectory.size < ad_ref.n_obs:
                trajectory = np.concatenate(
                    [trajectory, np.full(ad_ref.n_obs - trajectory.size, trajectory[-1])]
                )
            ad_ref.obs["trajectory"] = trajectory
            ad_ref.write_h5ad(ref_path)

            emb_path = embedding_path(run_cfg["embeddings_root"], "pca", "reference")
            ad_emb = ad.read_h5ad(emb_path)
            ad_emb.X = np.column_stack(
                [
                    np.linspace(0.0, 1.0, ad_emb.n_obs),
                    np.zeros((ad_emb.n_obs, ad_emb.n_vars - 1)),
                ]
            ).astype(np.float32)
            ad_emb.write_h5ad(emb_path)

            run_evaluate_trajectory(run_cfg)
            out_path = evaluation_trajectory_metrics_csv_path(run_cfg["results_dir"], "pca")
            self.assertTrue(out_path.is_file())
            df = pd.read_csv(out_path)
            self.assertFalse(df.empty)
            self.assertTrue((df["intervention_id"] == "reference").all())
            self.assertTrue((df["space"] == "embedding_reference").all())
            self.assertEqual(set(df["metric_category"].unique()), {"trajectory_metrics"})
            self.assertIn("ordering_correlation_spearman", set(df["metric_name"]))


if __name__ == "__main__":
    unittest.main()
