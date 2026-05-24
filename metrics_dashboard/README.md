# ScFMs Metrics Dashboard

Streamlit app for exploring structure-evaluation metrics produced by `make evaluate`.

## Artifacts layout

The dashboard reads either a **minimal bundle** (recommended for git) or a full SCEval tree on disk.

### Minimal bundle (Parquet, no h5ads in repo)

```text
{ARTIFACTS_ROOT}/{dataset_id}/
  metrics.parquet      # all models, params pre-joined
  summary.json         # cells / genes / cell types / batches
  manifest.json        # export provenance
```

Export from vault:

```bash
make export-dashboard-bundle SOURCE=/vault/.../processed/sceval/dendritic_cells
# or all datasets:
make export-dashboard-bundle SOURCE=/vault/.../processed/sceval
```

Output defaults to `data/dashboard_bundles/`. Point the dashboard at that directory
(`SCFM_ARTIFACTS_ROOT=data/dashboard_bundles` or the sidebar path).

### Legacy SCEval tree

```text
{SCFM_ARTIFACTS_ROOT}/{dataset_id}/results/
  evaluation/{model}_metrics.csv
  manipulations/{intervention_id}.h5ad
```

Default root: `/vault/amoneim/scfm-controlled-manipulations/processed/sceval`

Sweep plots use `value_mean` for lines and `value_mean ± value_std` for shaded bands
(spread across cells, not a confidence interval for the mean).

## Run locally

```bash
make dashboard
```

Or:

```bash
cd metrics_dashboard && uv sync
SCFM_ARTIFACTS_ROOT=/vault/amoneim/scfm-controlled-manipulations/processed/sceval \
  uv run streamlit run Home.py --server.port 8501
```

## Pages

| Page | Purpose |
|------|---------|
| **Home** | Catalog and quick dataset/model URL presets |
| **Explore** (Metrics) | Three plot sets: manipulation sweeps, integration correlations, collapse/shift |
| **Dataset summary** | Cells, genes, cell types, batches from `reference.h5ad` |
| **Compare** / **Model card** | Legacy stubs — use Explore for primary workflow |

## Configuration

- **Models & colors:** `metrics_dashboard/config.py` (`MODEL_ORDER`, `MODEL_COLORS`)
- **Dashboard metrics:** `DASHBOARD_METRICS` (KL, JS, kNN recall, Leiden ARI)

## URL query parameters

| Param | Example |
|-------|---------|
| `datasets` | `dendritic_cells,human_pbmc` |
| `models` | `pca,scgpt,geneformer` |

## Theme

Toggle light/dark via Streamlit **Settings → Theme**. Plot styling adapts automatically.

## Tests

```bash
python -m unittest tests.test_dashboard_transforms
```
