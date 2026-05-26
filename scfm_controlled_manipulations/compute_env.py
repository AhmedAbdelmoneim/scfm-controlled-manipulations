"""Limit BLAS / OpenMP / sklearn threading so process pools stay within a fixed CPU budget."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import os
from typing import Iterator

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)

# sklearn and joblib respect these when set before import; we set them at worker entry too.
_SKLEARN_ENV_VARS = (
    "SKLEARN_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
)


def thread_limit_environ(*, threads_per_process: int = 1) -> dict[str, str]:
    """Environment overrides that pin each process to ``threads_per_process`` compute threads."""
    count = str(max(1, int(threads_per_process)))
    values = dict.fromkeys(_THREAD_ENV_VARS, count)
    values.update(dict.fromkeys(_SKLEARN_ENV_VARS, count))
    values["OMP_DYNAMIC"] = "FALSE"
    values["MKL_DYNAMIC"] = "FALSE"
    values["OPENBLAS_DYNAMIC_ARCH"] = "FALSE"
    values["OPENBLAS_NO_AFFINITY"] = "1"
    values["GOTO_NUM_THREADS"] = count
    # scib-metrics pulls JAX; default to CPU so workers do not probe missing TPU/GPU libs.
    if "JAX_PLATFORMS" not in os.environ:
        values["JAX_PLATFORMS"] = "cpu"
    return values


def apply_thread_limits(*, threads_per_process: int = 1) -> dict[str, str]:
    """Apply thread-limit env vars in the current process; returns the values applied."""
    env = thread_limit_environ(threads_per_process=threads_per_process)
    os.environ.update(env)
    _configure_scanpy_jobs(threads_per_process)
    return env


def _configure_scanpy_jobs(threads_per_process: int) -> None:
    try:
        import scanpy as sc
    except ImportError:
        return
    sc.settings.n_jobs = max(1, int(threads_per_process))


@contextmanager
def thread_limited(threads_per_process: int = 1) -> Iterator[None]:
    """Temporarily apply thread limits, restoring prior env on exit."""
    prior: dict[str, str | None] = {}
    new_env = thread_limit_environ(threads_per_process=threads_per_process)
    for key, value in new_env.items():
        prior[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        _configure_scanpy_jobs(threads_per_process)
        yield
    finally:
        for key, old in prior.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def snapshot_environ(keys: Mapping[str, str]) -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in keys}
