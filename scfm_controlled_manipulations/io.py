from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import anndata as ad


def kwargs_hash(kwargs: dict[str, Any]) -> str:
    """Stable short hash of kwargs for ``intervention_id``."""
    payload = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def intervention_id(name: str, kwargs: dict[str, Any]) -> str:
    """``{name}_{hash_of_kwargs}`` so multiple variants of the same intervention do not collide."""
    return f"{name}_{kwargs_hash(kwargs)}"


def manipulation_path(results_dir: str | Path, intervention_id: str) -> Path:
    """Manipulated AnnData: ``{results_dir}/manipulations/{intervention_id}.h5ad``."""
    root = Path(results_dir)
    return root / "manipulations" / f"{intervention_id}.h5ad"


def embedding_path(embeddings_root: str | Path, model: str, intervention_id: str) -> Path:
    """External embeddings: ``{embeddings_root}/{model}/{intervention_id}.h5ad``."""
    root = Path(embeddings_root)
    return root / model / f"{intervention_id}.h5ad"


def metric_results_path(results_dir: str | Path, metric_name: str) -> Path:
    """Metric table: ``{results_dir}/metrics/{metric}.parquet``."""
    root = Path(results_dir)
    safe = metric_name.replace("/", "_")
    return root / "metrics" / f"{safe}.parquet"


def load_embedding(path: str | Path) -> Any:
    """Load embedding matrix from an h5ad file (returns ``adata.X``)."""
    adata = ad.read_h5ad(path)
    return adata.X
