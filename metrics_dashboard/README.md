# SCEval metrics dashboard

Streamlit app for exploring structure-evaluation metrics produced by `make evaluate`.

## Artifacts layout

The app reads (read-only) from:

```text
{SCFM_ARTIFACTS_ROOT}/{dataset_id}/results/
  evaluation/{model}_metrics.csv
  manipulations/{intervention_id}.h5ad   # intervention sweep parameters
```

Default root: `/vault/amoneim/scfm-controlled-manipulations/processed/sceval`

## Run locally

From the repo root:

```bash
make dashboard
```

Or from this directory:

```bash
cd metrics_dashboard
uv sync
SCFM_ARTIFACTS_ROOT=/vault/amoneim/scfm-controlled-manipulations/processed/sceval \
  uv run streamlit run Home.py --server.port 8501
```

## Pages

| Page | Purpose |
|------|---------|
| **Home** | Catalog of datasets and evaluation readiness |
| **Explore** | Filter and plot metrics for one dataset (all selected models on one chart) |
| **Compare** | Two views (A \| B) in one window — different datasets/models/configs |
| **Model Card** | Cross-dataset heatmaps and scorecard per model |

## URL query parameters

Share or duplicate a browser tab with the same view:

| Param | Example | Used on |
|-------|---------|---------|
| `dataset` | `dendritic_cells` | Explore, Home |
| `models` | `pca,scgpt,geneformer` | Explore |
| `a_dataset` | `dendritic_cells` | Compare view A |
| `a_models` | `pca,scgpt` | Compare view A |
| `b_dataset` | `human_pbmc` | Compare view B |
| `b_models` | `geneformer` | Compare view B |

Example:

```text
http://localhost:8501/Explore?dataset=dendritic_cells&models=pca,scgpt,geneformer
```

Open two tabs with different `dataset` / `models` for side-by-side comparison without the Compare page.

## Deploy with Streamlit

1. Set **main file** to `metrics_dashboard/Home.py` (or run from that directory).
2. Python 3.11; install with `uv sync` or `pip install -e .` from `metrics_dashboard/`.
3. Set secret / environment variable `SCFM_ARTIFACTS_ROOT` to a path visible on the host (vault mount or rsynced `*/results/evaluation/` trees).

If the host cannot mount the vault, sync artifacts:

```bash
rsync -av /vault/.../processed/sceval/*/results/evaluation/ /data/sceval/
rsync -av /vault/.../processed/sceval/*/results/manipulations/ /data/sceval-manip/
```

(Param joins need manipulation h5ads for sweep x-axes.)

## Relation to the marimo notebook

[`notebooks/3-analyze-metrics.py`](../notebooks/3-analyze-metrics.py) remains useful for ad-hoc analysis. This dashboard adds dataset/model pickers, compare panes, and model-card aggregation as evaluations complete on more atlases.
