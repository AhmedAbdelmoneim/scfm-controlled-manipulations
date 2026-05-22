"""Paths and constants for the metrics dashboard."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ARTIFACTS_ROOT = Path(
    "/vault/amoneim/scfm-controlled-manipulations/processed/sceval"
)

MODEL_ORDER = [
    "pca",
    "scgpt",
    "geneformer",
    "scfoundation",
    "scimilarity",
    "scconcept",
]

PARAM_KEYS = {
    "downsample": "fraction",
    "gene_dropout": "dropout_rate",
    "local_smoothing": "k",
    "poisson_resampling": "iterations",
    "gene_shuffle": "variant",
}

VALUE_COLUMNS = [
    "value_mean",
    "value_median",
    "value_std",
    "value_min",
    "value_max",
]


def artifacts_root() -> Path:
    """Root directory containing per-dataset SCEval processed outputs."""
    return Path(os.environ.get("SCFM_ARTIFACTS_ROOT", DEFAULT_ARTIFACTS_ROOT))


def results_dir(dataset_id: str, root: Path | None = None) -> Path:
    base = root or artifacts_root()
    return base / dataset_id / "results"


def evaluation_dir(dataset_id: str, root: Path | None = None) -> Path:
    return results_dir(dataset_id, root) / "evaluation"


def manipulations_dir(dataset_id: str, root: Path | None = None) -> Path:
    return results_dir(dataset_id, root) / "manipulations"
