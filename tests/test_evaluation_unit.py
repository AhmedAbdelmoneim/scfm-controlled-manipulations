"""Lightweight tests (stdlib ``unittest`` — no pytest dependency)."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scfm_controlled_manipulations.evaluation.metrics_knn import knn_neighbors, knn_overlap_per_cell
from scfm_controlled_manipulations.io import embedding_path, manipulation_path
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


class SweepTest(unittest.TestCase):
    def test_expand_cartesian(self) -> None:
        specs = [
            {"name": "downsample", "kwargs": {"fraction": [0.5, 0.9]}},
        ]
        out = expand_intervention_specs(specs)
        self.assertEqual(len(out), 2)
        self.assertEqual({o["kwargs"]["fraction"] for o in out}, {0.5, 0.9})


class KnnMaxSliceTest(unittest.TestCase):
    def test_slice_matches_direct_knn(self) -> None:
        from scfm_controlled_manipulations.evaluation.metrics_knn import knn_neighbors

        rng = np.random.default_rng(2)
        mat = rng.standard_normal((40, 6))
        k = 5
        _, idx_direct = knn_neighbors(mat, k, "euclidean")
        _, idx_max = knn_neighbors(mat, 10, "euclidean")
        self.assertTrue(np.array_equal(idx_direct, idx_max[:, :k]))


class LeidenMpSafetyTest(unittest.TestCase):
    def test_scanpy_leiden_kwargs_uses_igraph(self) -> None:
        from scfm_controlled_manipulations.evaluation.leiden_cache import scanpy_leiden_kwargs

        kw = scanpy_leiden_kwargs()
        self.assertEqual(kw["flavor"], "igraph")
        self.assertIn("n_iterations", kw)

    def test_fork_worker_delegates_leiden_to_spawn_pool(self) -> None:
        from unittest import mock

        import numpy as np

        from scfm_controlled_manipulations.evaluation.leiden_cache import leiden_labels_for_matrix

        mat = np.random.default_rng(0).standard_normal((40, 8)).astype(np.float32)
        fake_labels = np.array(["0", "1"] * 20)

        mock_pool = mock.MagicMock()
        mock_pool.apply.return_value = fake_labels

        with (
            mock.patch(
                "scfm_controlled_manipulations.evaluation.leiden_cache.mp.current_process"
            ) as proc,
            mock.patch(
                "scfm_controlled_manipulations.evaluation.leiden_cache._ensure_leiden_isolate_pool",
                return_value=mock_pool,
            ),
        ):
            proc.return_value.name = "ForkProcess-8"
            out = leiden_labels_for_matrix(mat, k=5, metric="euclidean", resolution=0.5, seed=0)

        self.assertTrue(np.array_equal(out, fake_labels))
        mock_pool.apply.assert_called_once()

    def test_resolve_mp_start_method(self) -> None:
        from scfm_controlled_manipulations.evaluation.pool import (
            resolve_evaluation_mp_start_method,
        )

        self.assertEqual(
            resolve_evaluation_mp_start_method(workers=48, configured="fork"),
            "fork",
        )
        self.assertEqual(
            resolve_evaluation_mp_start_method(workers=48, configured="spawn"),
            "spawn",
        )


class OvrRocApCvTest(unittest.TestCase):
    def test_singleton_class_returns_nan_without_warning(self) -> None:
        import warnings

        from scfm_controlled_manipulations.evaluation.metrics_cell_batch import _ovr_roc_ap_cv

        rng = np.random.default_rng(7)
        x = rng.standard_normal((30, 5))
        y = np.zeros(30, dtype=int)
        y[-1] = 1
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            roc_m, roc_s, ap_m, ap_s = _ovr_roc_ap_cv(x, y, seed=0)
        sklearn_msgs = [
            w.message
            for w in caught
            if w.category is not DeprecationWarning
            and "sklearn" in str(getattr(w, "filename", ""))
        ]
        self.assertEqual(sklearn_msgs, [])
        self.assertTrue(np.isnan(roc_m))
        self.assertTrue(np.isnan(ap_m))

    def test_multiclass_cv_no_sklearn_warnings(self) -> None:
        import warnings

        from sklearn.datasets import make_classification

        from scfm_controlled_manipulations.evaluation.metrics_cell_batch import _ovr_roc_ap_cv

        x, y = make_classification(
            n_samples=120,
            n_features=8,
            n_informative=6,
            n_classes=4,
            n_clusters_per_class=1,
            random_state=0,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ovr_roc_ap_cv(x, y, seed=0)
        sklearn_msgs = [
            w.message
            for w in caught
            if "sklearn" in str(getattr(w, "filename", ""))
        ]
        self.assertEqual(sklearn_msgs, [])


class NeighborVectorizedTest(unittest.TestCase):
    def test_entropy_and_ilisi_match_loop_reference(self) -> None:
        from scfm_controlled_manipulations.evaluation.metrics_cell_batch import (
            _encode_labels,
            _ilisi_like_score_per_cell,
            _neighbor_label_entropy_norm_per_cell,
        )
        from scfm_controlled_manipulations.evaluation.metrics_common import distribution_summary

        rng = np.random.default_rng(3)
        labels = rng.integers(0, 4, size=60)
        inverse, _ = _encode_labels(labels.astype(str))
        neighbor_idx = rng.integers(0, 60, size=(60, 8))

        def _entropy_loop(inv: np.ndarray, idx: np.ndarray) -> float:
            n_labels = int(inv.max()) + 1
            ent = []
            for row in idx:
                counts = np.bincount(inv[row], minlength=n_labels).astype(float)
                probs = counts / counts.sum()
                p = probs[probs > 0]
                ent.append(float(-np.sum(p * np.log(p))))
            return float(np.mean(ent) / np.log(n_labels))

        ent_summary = distribution_summary(
            _neighbor_label_entropy_norm_per_cell(inverse, neighbor_idx)
        )
        self.assertAlmostEqual(ent_summary.mean, _entropy_loop(inverse, neighbor_idx), places=10)
        ilisi_summary = distribution_summary(
            _ilisi_like_score_per_cell(inverse, neighbor_idx)
        )
        ilisi_m = ilisi_summary.mean
        self.assertGreater(ilisi_m, 0.0)


class KnnOverlapTest(unittest.TestCase):
    def test_perfect_overlap(self) -> None:
        n = 10
        k = 3
        idx = np.array([np.roll(np.arange(n), -i)[:k] for i in range(n)])
        recall = knn_overlap_per_cell(idx, idx, k)
        self.assertTrue(np.allclose(recall, 1.0))


class KnnPermutationNullTest(unittest.TestCase):
    def _mean_null_recall(
        self, ref: np.ndarray, man: np.ndarray, *, k: int, metric: str, seed: int
    ) -> float:
        from scfm_controlled_manipulations.evaluation.metrics_knn import (
            _knn_null_seed,
            knn_neighbors,
            knn_overlap_per_cell,
        )

        _, ref_idx = knn_neighbors(ref, k, metric)
        _, man_idx = knn_neighbors(man, k, metric)
        null_rng = np.random.default_rng(_knn_null_seed(seed, "embedding", metric, k))
        perm = null_rng.permutation(ref.shape[0])
        null_recall = knn_overlap_per_cell(ref_idx, man_idx[perm], k)
        return float(np.mean(null_recall))

    def test_empirical_null_breaks_pairing(self) -> None:
        from scfm_controlled_manipulations.evaluation.metrics_knn import (
            knn_neighbors,
            knn_overlap_per_cell,
        )

        rng = np.random.default_rng(0)
        n, d, k = 80, 8, 5
        ref = rng.standard_normal((n, d))
        man = ref.copy()
        _, ref_idx = knn_neighbors(ref, k, "euclidean")
        _, man_idx = knn_neighbors(man, k, "euclidean")
        recall = knn_overlap_per_cell(ref_idx, man_idx, k)
        self.assertGreater(float(np.mean(recall)), 0.95)

        null = self._mean_null_recall(ref, man, k=k, metric="euclidean", seed=42)
        self.assertLess(null, float(np.mean(recall)))
        self.assertEqual(
            null,
            self._mean_null_recall(ref, man, k=k, metric="euclidean", seed=42),
        )

    def test_empirical_null_above_analytical_with_structure(self) -> None:
        from sklearn.datasets import make_blobs

        ref, _ = make_blobs(
            n_samples=5000, centers=20, n_features=32, cluster_std=1.5, random_state=0
        )
        null = self._mean_null_recall(ref, ref.copy(), k=15, metric="euclidean", seed=0)
        analytical = 15 / (len(ref) - 1)
        self.assertGreater(null, analytical)

class DiffusionPermutationNullTest(unittest.TestCase):
    def test_permutation_null_exceeds_aligned_identity(self) -> None:
        from scfm_controlled_manipulations.evaluation.metrics_knn import (
            _diffusion_sym_kl_js_means,
            build_weighted_knn_adjacency,
            transition_powers,
        )

        rng = np.random.default_rng(0)
        ref = rng.standard_normal((120, 8))
        adj = build_weighted_knn_adjacency(ref, k=5, metric="euclidean")
        p_t = transition_powers(adj, [2])[2]
        q_t = p_t.copy()
        aligned_sym, aligned_js = _diffusion_sym_kl_js_means(p_t, q_t, row_chunk=64)
        perm = rng.permutation(ref.shape[0])
        null_sym, null_js = _diffusion_sym_kl_js_means(p_t, q_t[perm], row_chunk=64)
        self.assertAlmostEqual(aligned_sym, 0.0, places=5)
        self.assertAlmostEqual(aligned_js, 0.0, places=5)
        self.assertGreater(null_sym, aligned_sym)
        self.assertGreater(null_js, aligned_js)


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
            dense_embedding_aligned_to_obs(
                adata, pd.Index(["a", "b", "c"]), label="emb"
            )


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
            raw_ref=sp.csr_matrix(ref),
            raw_man=sp.csr_matrix(man),
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
            (df["metric_name"] == "paired_cell_l2_norm")
            & (df["space"] == "embedding")
        ].iloc[0]
        self.assertAlmostEqual(paired["value_mean"], 1.0, places=5)
        cos_row = df[
            (df["metric_name"] == "shift_pairwise_cosine")
            & (df["space"] == "embedding")
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
            (df["metric_name"] == "col_variance_ref")
            & (df["space"] == "embedding")
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
        col_ref = df[
            (df["metric_name"] == "col_mean_ref")
            & (df["space"] == "embedding")
        ].iloc[0]
        self.assertAlmostEqual(col_ref["value_mean"], float(np.mean(expected_means)), places=5)
        self.assertFalse(np.isnan(col_ref["value_mean"]))

    def test_reference_cache_idempotent(self) -> None:
        from scfm_controlled_manipulations.evaluation.context import (
            DatasetEvaluateContext,
            ModelEvaluateContext,
        )
        from scfm_controlled_manipulations.evaluation.metrics_stats_shift import (
            compute_embedding_stats,
            compute_embedding_shift,
        )
        from scfm_controlled_manipulations.evaluation.reference_stats_shift import (
            ReferenceStatsShiftCache,
            precompute_reference_stats_shift,
        )

        bundle = self._toy_bundle()
        dataset_ctx = DatasetEvaluateContext(
            raw_ref=bundle.raw_ref,
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
        bundle2 = type(bundle)(
            raw_ref=bundle.raw_ref,
            raw_man=sp.csr_matrix(bundle.raw_man.toarray() + np.array([0.0, 1.0])),
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


if __name__ == "__main__":
    unittest.main()
