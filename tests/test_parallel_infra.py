"""Tests for thread limits and process-safe disk cache."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scfm_controlled_manipulations.compute_env import (
    apply_thread_limits,
    blas_thread_limited,
    thread_limit_environ,
)
from scfm_controlled_manipulations.evaluation.context import DatasetEvaluateContext, ModelEvaluateContext
from scfm_controlled_manipulations.evaluation.disk_cache import (
    load_or_build_pickle,
    write_pickle_cache,
)
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

    def test_write_pickle_cache_creates_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            (results / "evaluation").mkdir(parents=True)
            cache_path = results / "evaluation_cache" / "snapshot.pkl"
            self.assertFalse(cache_path.parent.exists())
            write_pickle_cache(cache_path, {"ok": True})
            self.assertTrue(cache_path.is_file())


class BootstrapSnapshotTest(unittest.TestCase):
    def test_bootstrap_roundtrip_rebinds_caches(self) -> None:
        rng = np.random.default_rng(1)
        emb = rng.standard_normal((32, 8)).astype(np.float32)
        dataset_ctx = DatasetEvaluateContext(
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
                distance_metrics=["euclidean"],
                leiden_resolutions=[1.0],
                cache_path=tmp_path,
                stats_shift_pairwise_max_pairs=None,
                distance_correlation_subsample_n=500,
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
