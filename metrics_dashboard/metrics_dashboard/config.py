"""Paths, model registry, and dashboard metric metadata."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ARTIFACTS_ROOT = Path(
    "/vault/amoneim/scfm-controlled-manipulations/processed/sceval"
)

# Canonical model order and display labels (add new models here).
MODEL_ORDER = [
    "pca",
    "scgpt",
    "geneformer",
    "scfoundation",
    "scimilarity",
    "scconcept",
]

MODEL_LABELS: dict[str, str] = {
    "pca": "PCA",
    "scgpt": "scGPT",
    "geneformer": "Geneformer",
    "scfoundation": "scFoundation",
    "scimilarity": "SCimilarity",
    "scconcept": "scConcept",
}

# High-contrast, colorblind-friendly palette (one color per model, stable across plots).
MODEL_COLORS: dict[str, str] = {
    "pca": "#0072B2",
    "scgpt": "#E69F00",
    "geneformer": "#009E73",
    "scfoundation": "#CC79A7",
    "scimilarity": "#D55E00",
    "scconcept": "#56B4E9",
}

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
    "value_q05",
    "value_q25",
    "value_q75",
    "value_q95",
]

# Manipulations shown in Set 1/3 grids (order preserved).
MANIPULATION_ORDER = [
    "downsample",
    "gene_dropout",
    "poisson_resampling",
    "local_smoothing",
    "gene_shuffle",
]

REFERENCE_INTERVENTION_NAMES = frozenset({"reference"})


@dataclass(frozen=True)
class DashboardMetric:
    """User-facing metric selector mapped to evaluation CSV rows."""

    key: str
    label: str
    description: str
    metric_category: str
    metric_name: str
    space: str
    default_k: int | None = 15
    default_diffusion_t: int | None = None
    default_resolution: float | None = None
    x_col: str = "param_value"  # clustering uses "resolution"

    @property
    def y_label(self) -> str:
        return self.label


DASHBOARD_METRICS: dict[str, DashboardMetric] = {
    "kl_divergence": DashboardMetric(
        key="kl_divergence",
        label="Symmetric KL divergence",
        description=(
            "Symmetric KL between PHATE-style kNN random-walk transition distributions "
            "in embedding space (ref vs manipulated). Lower is better."
        ),
        metric_category="knn_metrics",
        metric_name="diffusion_sym_kl",
        space="embedding",
        default_k=15,
        default_diffusion_t=10,
    ),
    "js_divergence": DashboardMetric(
        key="js_divergence",
        label="Jensen–Shannon divergence",
        description=(
            "JS divergence between diffusion transition distributions in embedding space. "
            "Lower indicates closer neighborhood structure."
        ),
        metric_category="knn_metrics",
        metric_name="diffusion_js",
        space="embedding",
        default_k=15,
        default_diffusion_t=10,
    ),
    "knn_recall": DashboardMetric(
        key="knn_recall",
        label="kNN recall",
        description=(
            "Fraction of reference kNN neighbors preserved in the manipulated embedding "
            "(per cell, averaged). Higher is better."
        ),
        metric_category="knn_metrics",
        metric_name="knn_recall",
        space="embedding",
        default_k=15,
    ),
    "clustering_ari": DashboardMetric(
        key="clustering_ari",
        label="Leiden clustering ARI",
        description=(
            "Adjusted Rand index between independent Leiden clusterings on reference "
            "and manipulated embeddings. Higher is more stable."
        ),
        metric_category="clustering_metrics",
        metric_name="leiden_ari",
        space="embedding",
        default_resolution=1.0,
        x_col="resolution",
    ),
}

DASHBOARD_METRIC_KEYS = list(DASHBOARD_METRICS.keys())

PLOT_SET_DESCRIPTIONS = {
    "set1": (
        "Sweep plots: each row is a manipulation, each column a sweep facet "
        "(e.g. k or diffusion time). Solid lines show the mean over cells; shaded "
        "bands are mean ± 1 standard deviation across cells. Dashed lines are "
        "permutation null means."
    ),
    "set2": (
        "Integration vs structure: each point is one run (model × intervention). "
        "Lines connect points within the same manipulation. Correlation and "
        "p-value are Pearson on displayed points."
    ),
    "set3": (
        "Embedding collapse (within-cluster distance) and shift along manipulation "
        "degree. Solid lines are cell means; shaded bands are mean ± 1 std across "
        "cells. The first point is always the reference."
    ),
}

# Set 3 embedding metrics (embedding space).
SET3_COLLAPSE_METRIC = "within_man_pairwise_l2"
SET3_SHIFT_METRIC = "paired_cell_l2_norm"
SET3_CATEGORY = "embedding_shift"
SET3_SPACE = "embedding"


def model_palette(models: list[str] | None = None) -> dict[str, str]:
    """Return color map for models present (or full registry)."""
    if models is None:
        return dict(MODEL_COLORS)
    return {m: MODEL_COLORS.get(m, "#888888") for m in models}


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


def reference_h5ad_path(dataset_id: str, root: Path | None = None) -> Path:
    return manipulations_dir(dataset_id, root) / "reference.h5ad"
