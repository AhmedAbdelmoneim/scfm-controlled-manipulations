from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def kwargs_hash(kwargs: dict[str, Any]) -> str:
    """Stable short hash of kwargs for ``intervention_id``."""
    payload = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def intervention_id(name: str, kwargs: dict[str, Any]) -> str:
    """``{name}_{hash_of_kwargs}`` so multiple variants of the same intervention do not collide."""
    return f"{name}_{kwargs_hash(kwargs)}"


def manipulations_dir(
    results_dir: str | Path,
    configured_dir: str | Path | None = None,
) -> Path:
    """Directory containing manipulation h5ads.

    Defaults to the historical ``{results_dir}/manipulations`` layout.
    """
    if configured_dir is not None:
        return Path(configured_dir)
    return Path(results_dir) / "manipulations"


def manipulation_path(
    results_dir: str | Path,
    intervention_id: str,
    manipulations_dir_path: str | Path | None = None,
) -> Path:
    """Manipulated AnnData path for ``intervention_id``."""
    root = manipulations_dir(results_dir, manipulations_dir_path)
    return root / f"{intervention_id}.h5ad"


def embedding_path(embeddings_root: str | Path, model: str, intervention_id: str) -> Path:
    """External embeddings: ``{embeddings_root}/{model}/{model}_{intervention_id}.h5ad``.

    Example: ``embeddings/pca/pca_downsample_6e9bcd431d63.h5ad``, reference
    ``embeddings/pca/pca_reference.h5ad`` when ``reference_intervention_id`` is ``reference``.
    """
    root = Path(embeddings_root)
    return root / model / f"{model}_{intervention_id}.h5ad"


def evaluation_dir(results_dir: str | Path) -> Path:
    """Directory for structure-evaluation outputs: ``{results_dir}/evaluation``."""
    return Path(results_dir) / "evaluation"


def evaluation_metrics_csv_path(results_dir: str | Path, model: str) -> Path:
    """Per-model consolidated metrics CSV under ``evaluation_dir``."""
    return evaluation_dir(results_dir) / f"{model}_metrics.csv"


def evaluation_scib_metrics_csv_path(results_dir: str | Path, model: str) -> Path:
    """Per-model scIB Benchmarker metrics CSV under ``evaluation_dir``."""
    return evaluation_dir(results_dir) / f"{model}_scib_metrics.csv"


def evaluation_trajectory_metrics_csv_path(results_dir: str | Path, model: str) -> Path:
    """Per-model trajectory inference metrics CSV under ``evaluation_dir``."""
    return evaluation_dir(results_dir) / f"{model}_trajectory_metrics.csv"


def evaluation_cache_dir(results_dir: str | Path) -> Path:
    """On-disk cache for expensive reference-side evaluation artifacts."""
    return Path(results_dir) / "evaluation_cache"


def list_manipulation_ids(
    results_dir: str | Path,
    manipulations_dir_path: str | Path | None = None,
) -> list[str]:
    """Stem names of manipulation h5ads excluding ``reference``."""
    manip_dir = manipulations_dir(results_dir, manipulations_dir_path)
    if not manip_dir.is_dir():
        return []
    ids = sorted(p.stem for p in manip_dir.glob("*.h5ad") if p.is_file() and p.stem != "reference")
    return ids
