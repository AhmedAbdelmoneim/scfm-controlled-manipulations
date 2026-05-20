"""Tests for thread limits and process-safe disk cache."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from scfm_controlled_manipulations.compute_env import apply_thread_limits, thread_limit_environ
from scfm_controlled_manipulations.evaluation.disk_cache import load_or_build_pickle


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

if __name__ == "__main__":
    unittest.main()
