"""Tests for thread limits and process-safe disk cache."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scfm_controlled_manipulations.compute_env import (
    apply_thread_limits,
    blas_thread_limited,
    thread_limit_environ,
)
from scfm_controlled_manipulations.evaluation.context import DatasetEvaluateContext, ModelEvaluateContext
from scfm_controlled_manipulations.evaluation.disk_cache import load_or_build_pickle
from scfm_controlled_manipulations.evaluation.knn_cache import KnnIndexCache
from scfm_controlled_manipulations.evaluation.metrics_knn import knn_neighbors
from scfm_controlled_manipulations.evaluation.run import merge_evaluation_config
from scfm_controlled_manipulations.evaluation.worker import (
    SharedEvalContext,
    load_worker_bootstrap,
    write_worker_bootstrap,
)


class BlasThreadLimitTest(unittest.TestCase):
    def test_blas_thread_limited_does_not_raise_numba_threads(self) -> None:
        os.environ["NUMBA_NUM_THREADS"] = "1"
        with blas_thread_limited(8):
            self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "8")
            self.assertEqual(os.environ.get("NUMBA_NUM_THREADS"), "1")
        self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "1")

    def test_knn_neighbors_with_n_jobs_does_not_touch_scanpy_n_jobs(self) -> None:
        import scanpy as sc

        sc.settings.n_jobs = 1
        rng = np.random.default_rng(3)
        mat = rng.standard_normal((24, 5))
        knn_neighbors(mat, 3, "euclidean", n_jobs=4)
        self.assertEqual(sc.settings.n_jobs, 1)


class ThreadLimitTest(unittest.TestCase):
    def test_thread_limit_environ_sets_blas_vars(self) -> None:
        env = thread_limit_environ(threads_per_process=1)
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")
        self.assertEqual(env["SKLEARN_NUM_THREADS"], "1")

    def test_apply_thread_limits_updates_process_environ(self) -> None:
        apply_thread_limits(threads_per_process=2)
        self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "2")


class DiskCacheTest(unittest.TestCase):
    def test_load_or_build_pickle_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "value.pkl"
            calls = {"n": 0}

            def builder() -> int:
                calls["n"] += 1
                return 42

            self.assertEqual(load_or_build_pickle(path, builder, label="test"), 42)
            self.assertEqual(load_or_build_pickle(path, builder, label="test"), 42)
            self.assertEqual(calls["n"], 1)


class TransitionBundleCacheTest(unittest.TestCase):
    def test_transition_bundle_builder_runs_once(self) -> None:
        import numpy as np
        import scipy.sparse as sp

        from scfm_controlled_manipulations.evaluation.metrics_knn import (
            _load_or_compute_transition_powers,
        )

        rng = np.random.default_rng(0)
        mat = rng.standard_normal((64, 8))
        calls = {"n": 0}

        def adj_builder() -> sp.csr_matrix:
            from scfm_controlled_manipulations.evaluation.metrics_knn import (
                build_weighted_knn_adjacency,
            )

            calls["n"] += 1
            return build_weighted_knn_adjacency(mat, k=5, metric="euclidean")

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            kwargs = dict(
                cache_dir=cache_dir,
                dataset_id="ds",
                model="pca",
                space="embedding",
                metric="euclidean",
                k=5,
                n_cells=64,
                side="ref",
                adj_builder=adj_builder,
                t_values=[1, 2, 4],
            )
            first = _load_or_compute_transition_powers(**kwargs)
            second = _load_or_compute_transition_powers(**kwargs)
            self.assertEqual(set(first.keys()), {1, 2, 4})
            self.assertEqual(set(second.keys()), {1, 2, 4})
            self.assertEqual(calls["n"], 1)


class KnnDiskWarmTest(unittest.TestCase):
    def test_warm_reference_from_disk_matches_direct_knn(self) -> None:
        rng = np.random.default_rng(0)
        mat = rng.standard_normal((48, 6))
        k = 5
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            cache = KnnIndexCache()
            direct = knn_neighbors(mat, k, "euclidean", n_jobs=1)
            loaded = cache.warm_reference_from_disk(
                mat,
                space="embedding",
                k_max=k,
                metric="euclidean",
                cache_dir=cache_dir,
                dataset_id="ds",
                model="pca",
                n_cells=48,
            )
            self.assertTrue(np.allclose(direct[0], loaded[0]))
            self.assertTrue(np.array_equal(direct[1], loaded[1]))
            cached = cache.neighbors(mat, k, "euclidean")
            self.assertTrue(np.array_equal(cached[1], direct[1]))


class BootstrapSnapshotTest(unittest.TestCase):
    def test_bootstrap_roundtrip_rebinds_caches(self) -> None:
        rng = np.random.default_rng(1)
        raw = sp.csr_matrix(rng.standard_normal((32, 4)))
        emb = rng.standard_normal((32, 8)).astype(np.float32)
        dataset_ctx = DatasetEvaluateContext(
            raw_ref=raw,
            obs=pd.DataFrame(index=range(32)),
            n_cells=32,
        )
        model_ctx = ModelEvaluateContext(emb_ref=emb)
        model_ctx.leiden_cache.labels(
            emb, k=5, metric="euclidean", resolution=1.0, seed=7
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            shared = SharedEvalContext(
                dataset_ctx=dataset_ctx,
                model_ctx=model_ctx,
                results_dir=tmp_path,
                embeddings_root=tmp_path,
                model="pca",
                ref_id="reference",
                dataset_id="ds",
                seed=7,
                k_values=[5],
                trustworthiness_k_values=[5],
                distance_metrics=["euclidean"],
                diffusion_t_values=[1],
                leiden_resolutions=[1.0],
                cache_path=tmp_path,
                cell_type_col=None,
                batch_col=None,
                stats_shift_pairwise_max_pairs=None,
                knn_alpha=10.0,
                knn_bandwidth_k=None,
                knn_n_null_permutations=1,
                static_row_templates=[],
            )
            path = tmp_path / "bootstrap.pkl"
            write_worker_bootstrap(path, shared)
            reloaded = load_worker_bootstrap(path)
            self.assertEqual(len(reloaded.model_ctx.leiden_cache), 1)
            labels = reloaded.model_ctx.leiden_cache.labels(
                reloaded.model_ctx.emb_ref,
                k=5,
                metric="euclidean",
                resolution=1.0,
                seed=7,
            )
            self.assertEqual(len(labels), 32)


class EvaluationConfigThreadTest(unittest.TestCase):
    def test_validate_accepts_setup_and_worker_threads(self) -> None:
        out = merge_evaluation_config(
            {
                "evaluation": {
                    "evaluation_workers": 4,
                    "evaluation_setup_threads": 8,
                    "evaluation_worker_threads": 2,
                }
            }
        )
        self.assertEqual(out["evaluation_setup_threads"], 8)
        self.assertEqual(out["evaluation_worker_threads"], 2)


if __name__ == "__main__":
    unittest.main()
