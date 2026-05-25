"""Paths, model registry, and dashboard metric metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

def _find_repo_root() -> Path:
    """Locate repo root by ``data/dashboard_bundles`` (works locally and on Streamlit Cloud)."""
    import logging
    import os

    log = logging.getLogger("scfm_dashboard.config")
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    for parent in here.parents:
        candidates.append(parent)
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if parent not in candidates:
            candidates.append(parent)
    for parent in candidates:
        bundle_dir = parent / "data" / "dashboard_bundles"
        if bundle_dir.is_dir():
            log.info("repo_root=%s (found %s)", parent, bundle_dir)
            print(f"[scfm_dashboard] repo_root={parent}", flush=True)
            return parent
    fallback = here.parents[2]
    log.warning(
        "dashboard_bundles not found (cwd=%s); fallback repo_root=%s",
        os.getcwd(),
        fallback,
    )
    print(f"[scfm_dashboard] WARNING: no dashboard_bundles; fallback={fallback}", flush=True)
    return fallback


BUNDLE_ROOT = _find_repo_root() / "data" / "dashboard_bundles"

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
    key: str
    label: str
    description: str
    metric_category: str
    metric_name: str
    space: str
    default_k: int | None = 15
    default_diffusion_t: int | None = None
    default_resolution: float | None = None
    x_col: str = "param_value"

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
        "Grid layout: **rows** = manipulation type; **columns** = manipulation config "
        "(e.g. fraction, dropout rate, shuffle variant); **x-axis** = diffusion time *t* "
        "for KL/JS, **k** for kNN recall, or the relevant sweep for other metrics. "
        "Colored lines = models. "
        "Bands = mean ± 1 std across cells; dashed = permutation null mean."
    ),
    "set2": (
        "Integration vs structure: each point is one run. Lines connect points within "
        "the same manipulation. Correlation and p-value are Pearson on displayed points."
    ),
    "set3": (
        "Embedding collapse and shift along manipulation degree. "
        "Solid lines = cell means; shaded bands = mean ± 1 std. First point = reference."
    ),
}

SET3_COLLAPSE_METRIC = "within_man_pairwise_l2"
SET3_SHIFT_METRIC = "paired_cell_l2_norm"
SET3_CATEGORY = "embedding_shift"
SET3_SPACE = "embedding"


def bundle_root() -> Path:
    """Checked-in dashboard bundles under ``data/dashboard_bundles``."""
    return BUNDLE_ROOT


def model_palette(models: list[str] | None = None) -> dict[str, str]:
    if models is None:
        return dict(MODEL_COLORS)
    return {m: MODEL_COLORS.get(m, "#888888") for m in models}
