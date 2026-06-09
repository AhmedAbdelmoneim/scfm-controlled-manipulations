"""In-memory cache for kNN neighbor indices (exact sklearn results, keyed by matrix identity)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from scfm_controlled_manipulations.evaluation.disk_cache import load_or_build_pickle
from scfm_controlled_manipulations.evaluation.metrics_knn import _knn_cache_path, knn_neighbors


class KnnIndexCache:
    """Cache ``knn_neighbors`` results for reused matrix objects (e.g. reference matrices)."""

    def __init__(self) -> None:
        self._store: dict[tuple[int, str, int], tuple[np.ndarray, np.ndarray]] = {}
        self._lock = threading.Lock()

    def seed(
        self,
        mat: Any,
        k_max: int,
        metric: str,
        result: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Install a precomputed (dist, idx) pair for ``mat`` without recomputing."""
        key = (id(mat), metric, int(k_max))
        with self._lock:
            self._store.setdefault(key, result)

    def neighbors(
        self,
        mat: Any,
        k_max: int,
        metric: str,
        *,
        knn_n_jobs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (id(mat), metric, int(k_max))
        with self._lock:
            cached = self._store.get(key)
        if cached is not None:
            return cached
        result = knn_neighbors(mat, k_max, metric, n_jobs=knn_n_jobs)
        with self._lock:
            self._store.setdefault(key, result)
            return self._store[key]

    def warm_reference_from_disk(
        self,
        mat: Any,
        *,
        space: str,
        k_max: int,
        metric: str,
        cache_dir: Path,
        dataset_id: str,
        model: str,
        n_cells: int,
        knn_n_jobs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load or build reference kNN on disk, then seed the in-memory cache."""
        path = _knn_cache_path(
            cache_dir,
            dataset_id=dataset_id,
            model=model,
            space=space,
            metric=metric,
            k=k_max,
            n_cells=n_cells,
            side="ref",
        )
        label = f"knn side=ref space={space} metric={metric} k={k_max} ({n_cells} cells)"

        def _build() -> tuple[np.ndarray, np.ndarray]:
            return knn_neighbors(mat, k_max, metric, n_jobs=knn_n_jobs)

        result = load_or_build_pickle(path, _build, label=label)
        self.seed(mat, k_max, metric, result)
        return result

    def __len__(self) -> int:
        return len(self._store)

    def __getstate__(self) -> dict[tuple[int, str, int], tuple[np.ndarray, np.ndarray]]:
        return self._store

    def __setstate__(self, state: dict[tuple[int, str, int], tuple[np.ndarray, np.ndarray]]) -> None:
        self._store = state
        self._lock = threading.Lock()
