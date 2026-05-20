"""Lightweight tests (stdlib ``unittest`` — no pytest dependency)."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

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

        ent_m, _, _ = distribution_summary(
            _neighbor_label_entropy_norm_per_cell(inverse, neighbor_idx)
        )
        self.assertAlmostEqual(ent_m, _entropy_loop(inverse, neighbor_idx), places=10)
        ilisi_m, _, _ = distribution_summary(_ilisi_like_score_per_cell(inverse, neighbor_idx))
        self.assertGreater(ilisi_m, 0.0)


class KnnOverlapTest(unittest.TestCase):
    def test_perfect_overlap(self) -> None:
        n = 10
        k = 3
        idx = np.array([np.roll(np.arange(n), -i)[:k] for i in range(n)])
        recall, jaccard = knn_overlap_per_cell(idx, idx, k)
        self.assertTrue(np.allclose(recall, 1.0))
        self.assertTrue(np.allclose(jaccard, 1.0))


class EmpiricalKnnNullTest(unittest.TestCase):
    def test_empirical_null_breaks_pairing(self) -> None:
        from scfm_controlled_manipulations.evaluation.metrics_knn import (
            empirical_knn_recall_null,
            knn_indices,
            knn_overlap_per_cell,
        )

        rng = np.random.default_rng(0)
        n, d, k = 80, 8, 5
        ref = rng.standard_normal((n, d))
        man = ref.copy()
        _, ref_idx = knn_indices(ref, k=k, metric="euclidean")
        _, man_idx = knn_indices(man, k=k, metric="euclidean")
        recall, _ = knn_overlap_per_cell(ref_idx, man_idx, k)
        self.assertGreater(float(np.mean(recall)), 0.95)

        null = empirical_knn_recall_null(ref, man, k=k, metric="euclidean", seed=42)
        self.assertLess(null, float(np.mean(recall)))
        self.assertEqual(
            null,
            empirical_knn_recall_null(ref, man, k=k, metric="euclidean", seed=42),
        )

    def test_empirical_null_above_analytical_with_structure(self) -> None:
        from sklearn.datasets import make_blobs

        from scfm_controlled_manipulations.evaluation.metrics_knn import empirical_knn_recall_null

        ref, _ = make_blobs(n_samples=5000, centers=20, n_features=32, cluster_std=1.5, random_state=0)
        null = empirical_knn_recall_null(ref, ref.copy(), k=15, metric="euclidean", seed=0)
        analytical = 15 / (len(ref) - 1)
        self.assertGreater(null, analytical)


class DiffusionPermutationNullTest(unittest.TestCase):
    def test_permutation_null_exceeds_aligned_identity(self) -> None:
        from scfm_controlled_manipulations.evaluation.metrics_knn import (
            _diffusion_sym_kl_js_means,
            build_weighted_knn_adjacency,
            sparse_transition_power,
        )

        rng = np.random.default_rng(0)
        ref = rng.standard_normal((120, 8))
        adj = build_weighted_knn_adjacency(ref, k=5, metric="euclidean")
        p_t = sparse_transition_power(adj, t=2)
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


class EmbeddingShiftTest(unittest.TestCase):
    def test_centroid_shift_dense(self) -> None:
        from scfm_controlled_manipulations.evaluation.metrics_stats_shift import (
            _centroid_l2_dense,
        )

        ref = np.zeros((5, 2), dtype=np.float32)
        man = np.ones((5, 2), dtype=np.float32)
        self.assertAlmostEqual(_centroid_l2_dense(ref, man), np.sqrt(2.0), places=4)


if __name__ == "__main__":
    unittest.main()
