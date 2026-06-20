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
    "viscore_local_sp": DashboardMetric(
        key="viscore_local_sp",
        label="ViScore local SP",
        description=(
            "ViScore local structure preservation (Sl): AUC of the R_NX curve with log K axis. "
            "Higher is better; bounded in [-1, 1]."
        ),
        metric_category="structure_metrics",
        metric_name="viscore_local_sp",
        space="embedding",
    ),
    "viscore_global_sp": DashboardMetric(
        key="viscore_global_sp",
        label="ViScore global SP",
        description=(
            "ViScore global structure preservation (Sg): AUC of the R_NX curve with linear K axis. "
            "Higher is better; bounded in [-1, 1]."
        ),
        metric_category="structure_metrics",
        metric_name="viscore_global_sp",
        space="embedding",
    ),
    "distcorr": DashboardMetric(
        key="distcorr",
        label="Distance correlation",
        description=(
            "Székely distance correlation between reference and manipulated distance matrices. "
            "Higher is better."
        ),
        metric_category="structure_metrics",
        metric_name="distcorr",
        space="embedding",
    ),
    "clustering_ari": DashboardMetric(
        key="clustering_ari",
        label="Leiden ARI",
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
MAIN_METRIC_KEYS = [
    "viscore_local_sp",
    "viscore_global_sp",
    "distcorr",
    "clustering_ari",
]
MAIN_METRICS = [DASHBOARD_METRICS[key] for key in MAIN_METRIC_KEYS]

# Fixed subplot scale (formerly a sidebar slider).
DEFAULT_PLOT_SCALE = 1.25

PLOT_SET_DESCRIPTIONS = {
    "set1": (
        "Main metrics across manipulation strength. Metric/manipulation panels show "
        "manipulation parameter on the x-axis and metric value on the y-axis. "
        "Colored lines = models; fixed default y-ranges make panels comparable. "
        "Bands = mean ± 1 std across cells; dashed = permutation null mean."
    ),
    "set2": (
        "R_NX curves in depth. Rows are manipulation types, columns are manipulation "
        "parameters, x-axis is neighborhood size k, y-axis is R_NX, and colored lines "
        "are models."
    ),
    "set3": (
        "Columns = manipulation; x-axis = sweep level (numeric fraction/rate/k or shuffle "
        "variant). Top row: within-manipulation pairwise distance (collapse; reference point "
        "at x=0). Bottom row: per-cell shift from reference divided by ref within-cluster "
        "distance. Lines = models."
    ),
}

SET3_COLLAPSE_METRIC = "within_man_pairwise_l2"
SET3_SHIFT_METRIC = "paired_cell_l2_norm"
SET3_CATEGORY = "embedding_shift"
SET3_SPACE = "embedding"
SET3_COLLAPSE_YLABEL = "Within-cluster distance"
SET3_SHIFT_YLABEL = "Paired shift / ref within"


def bundle_root() -> Path:
    """Checked-in dashboard bundles under ``data/dashboard_bundles``."""
    return BUNDLE_ROOT


def model_palette(models: list[str] | None = None) -> dict[str, str]:
    if models is None:
        return dict(MODEL_COLORS)
    return {m: MODEL_COLORS.get(m, "#888888") for m in models}
