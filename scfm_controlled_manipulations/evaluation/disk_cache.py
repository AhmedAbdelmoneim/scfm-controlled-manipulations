"""Process-safe on-disk pickle cache with exclusive file locking (Unix)."""

from __future__ import annotations

from collections.abc import Callable
import logging
import os
from pathlib import Path
import pickle
import tempfile
from typing import TypeVar

try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix platforms
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

T = TypeVar("T")


def read_pickle_cache(path: Path) -> T:
    """Load a pickle written by :func:`load_or_build_pickle` or :func:`write_pickle_cache`."""
    with open(path, "rb") as handle:
        return pickle.load(handle)


def write_pickle_cache(path: Path, value: T) -> None:
    """Atomically write a pickle (e.g. worker bootstrap snapshots)."""
    _write_pickle_atomic(path, value)


def _write_pickle_atomic(path: Path, value: T) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as tmp_handle:
            pickle.dump(value, tmp_handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load_or_build_pickle(
    path: Path,
    builder: Callable[[], T],
    *,
    label: str,
) -> T:
    """Return cached object at ``path``, building atomically under an exclusive lock on miss."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        logger.debug("disk cache hit: %s (%s)", path.name, label)
        return read_pickle_cache(path)

    if fcntl is None:
        logger.warning(
            "fcntl unavailable; building %s without cross-process lock (race possible)",
            label,
        )
        if path.is_file():
            return read_pickle_cache(path)
        value = builder()
        _write_pickle_atomic(path, value)
        return value

    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "wb") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if path.is_file():
            logger.debug("disk cache hit after lock: %s (%s)", path.name, label)
            return read_pickle_cache(path)

        logger.info("disk cache miss — building %s", label)
        value = builder()
        _write_pickle_atomic(path, value)
        return value
