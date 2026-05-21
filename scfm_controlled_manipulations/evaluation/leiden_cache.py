"""Reuse scanpy neighbor graphs and Leiden labels for identical (matrix, k, metric, resolution)."""

from __future__ import annotations

from dataclasses import dataclass, field
import multiprocessing as mp
import os
import threading
from typing import Any

import anndata as ad
import numpy as np
import scanpy as sc

from scfm_controlled_manipulations.compute_env import apply_thread_limits

_LEIDEN_ISOLATE_POOL: mp.pool.Pool | None = None
_LEIDEN_ISOLATE_POOL_PID: int | None = None


def scanpy_leiden_kwargs() -> dict[str, Any]:
    """Scanpy Leiden kwargs for the main process (igraph is fastest)."""
    return {
        "flavor": "igraph",
        "n_iterations": 2,
        "directed": False,
    }


def _leiden_labels_compute(
    mat: np.ndarray,
    *,
    k: int,
    metric: str,
    resolution: float,
    seed: int,
) -> np.ndarray:
    """Neighbors + Leiden in the current process (main process only)."""
    adata_tmp = ad.AnnData(np.asarray(mat))
    sc.pp.neighbors(
        adata_tmp,
        n_neighbors=int(k),
        metric=metric,
        use_rep="X",
        random_state=seed,
    )
    sc.tl.leiden(
        adata_tmp,
        resolution=float(resolution),
        random_state=seed,
        key_added="leiden_eval",
        **scanpy_leiden_kwargs(),
    )
    return adata_tmp.obs["leiden_eval"].astype(str).to_numpy()


def _leiden_spawn_entry(
    payload: tuple[np.ndarray, int, str, float, int],
) -> np.ndarray:
    """Picklable entry for a fresh spawn child (no inherited OpenMP from fork workers)."""
    apply_thread_limits(threads_per_process=1)
    mat, k, metric, resolution, seed = payload
    return _leiden_labels_compute(
        mat,
        k=k,
        metric=metric,
        resolution=resolution,
        seed=seed,
    )


def _ensure_leiden_isolate_pool() -> mp.pool.Pool:
    """One spawn worker per fork-pool process for Leiden after kNN/OpenMP."""
    global _LEIDEN_ISOLATE_POOL, _LEIDEN_ISOLATE_POOL_PID
    pid = os.getpid()
    if _LEIDEN_ISOLATE_POOL is None or _LEIDEN_ISOLATE_POOL_PID != pid:
        if _LEIDEN_ISOLATE_POOL is not None:
            _LEIDEN_ISOLATE_POOL.close()
            _LEIDEN_ISOLATE_POOL.join()
        _LEIDEN_ISOLATE_POOL = mp.get_context("spawn").Pool(1)
        _LEIDEN_ISOLATE_POOL_PID = pid
    return _LEIDEN_ISOLATE_POOL


def init_leiden_isolate_pool() -> None:
    """Call from fork pool worker initializer before any Leiden in that worker."""
    if mp.current_process().name != "MainProcess":
        _ensure_leiden_isolate_pool()


def leiden_labels_for_matrix(
    mat: np.ndarray,
    *,
    k: int,
    metric: str,
    resolution: float,
    seed: int,
) -> np.ndarray:
    """Run neighbors + Leiden; use a spawn subprocess inside fork pool workers."""
    payload = (np.asarray(mat), int(k), str(metric), float(resolution), int(seed))
    if mp.current_process().name == "MainProcess":
        return _leiden_labels_compute(
            payload[0],
            k=payload[1],
            metric=payload[2],
            resolution=payload[3],
            seed=payload[4],
        )
    pool = _ensure_leiden_isolate_pool()
    return pool.apply(_leiden_spawn_entry, (payload,))


@dataclass
class LeidenCache:
    """Thread-safe cache for neighbors graphs and Leiden cluster labels."""

    _labels: dict[tuple[int, str, int, float, int], np.ndarray] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def labels(
        self,
        mat: np.ndarray,
        *,
        k: int,
        metric: str,
        resolution: float,
        seed: int,
    ) -> np.ndarray:
        label_key = (id(mat), metric, int(k), float(resolution), int(seed))
        with self._lock:
            cached = self._labels.get(label_key)
        if cached is not None:
            return cached

        labels = leiden_labels_for_matrix(
            mat,
            k=k,
            metric=metric,
            resolution=resolution,
            seed=seed,
        )
        with self._lock:
            self._labels.setdefault(label_key, labels)
            return self._labels[label_key]

    def __len__(self) -> int:
        return len(self._labels)
