"""Multiprocessing context selection for evaluation workers."""

from __future__ import annotations

import multiprocessing as mp
import sys


def resolve_evaluation_mp_start_method(
    *,
    workers: int = 1,
    configured: str | None = None,
) -> str:
    """Return ``fork`` or ``spawn`` for the intervention process pool.

    Default ``fork`` on Linux preserves copy-on-write sharing of reference data.
    Fork workers delegate neighbors+Leiden to a nested ``spawn`` child (see
    ``leiden_cache.init_leiden_isolate_pool``) so OpenMP from kNN does not collide with
    ``fork()`` inside scanpy/igraph.

    Set ``evaluation.evaluation_mp_start_method: spawn`` to reload full context per worker
    (higher memory) if nested Leiden isolation is insufficient.
    """
    method = str(configured or "fork").strip().lower()
    if method == "auto":
        method = "fork" if sys.platform == "linux" else "spawn"
    if method not in ("fork", "spawn"):
        raise ValueError(
            f"evaluation_mp_start_method must be fork, spawn, or auto; got {configured!r}"
        )
    if workers <= 1:
        return method
    return method


def evaluation_mp_context(
    *,
    workers: int = 1,
    configured: str | None = None,
) -> mp.context.BaseContext:
    return mp.get_context(
        resolve_evaluation_mp_start_method(workers=workers, configured=configured)
    )


def use_fork_shared_memory(
    *,
    workers: int = 1,
    configured: str | None = None,
) -> bool:
    return resolve_evaluation_mp_start_method(workers=workers, configured=configured) == "fork"
