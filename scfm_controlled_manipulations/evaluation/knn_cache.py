"""In-memory cache for kNN neighbor indices (exact sklearn results, keyed by matrix identity)."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from scfm_controlled_manipulations.evaluation.metrics_knn import knn_neighbors


class KnnIndexCache:
    """Cache ``knn_neighbors`` results for reused matrix objects (e.g. reference matrices)."""

    def __init__(self) -> None:
        self._store: dict[tuple[int, str, int], tuple[np.ndarray, np.ndarray]] = {}
        self._lock = threading.Lock()

    def neighbors(self, mat: Any, k_max: int, metric: str) -> tuple[np.ndarray, np.ndarray]:
        key = (id(mat), metric, int(k_max))
        with self._lock:
            cached = self._store.get(key)
        if cached is not None:
            return cached
        result = knn_neighbors(mat, k_max, metric, n_jobs=1)
        with self._lock:
            self._store.setdefault(key, result)
            return self._store[key]

    def __len__(self) -> int:
        return len(self._store)
