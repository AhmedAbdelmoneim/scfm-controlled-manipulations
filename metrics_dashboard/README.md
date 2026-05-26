# ScFMs Metrics Dashboard

Streamlit app for structure-evaluation metrics. Data comes from checked-in Parquet bundles under `data/dashboard_bundles/` (no vault mount or path configuration in the UI).

## Run

```bash
make dashboard
```

### Streamlit Community Cloud

- **Main file:** `metrics_dashboard/Home.py`
- **App directory:** `metrics_dashboard/` (use `requirements.txt` only — do not add `uv.lock` here or Cloud will install a stale `metrics-dashboard` wheel and shadow the app code)
- Ensure `data/dashboard_bundles/` is committed in the repo root.

Use the sidebar **Appearance** control for light/dark plots (defaults to dark).

## Export bundles (from vault)

```bash
make export-dashboard-bundle SOURCE=/vault/.../processed/sceval/dendritic_cells
# Commit data/dashboard_bundles/{dataset_id}/
```

Each bundle contains `metrics.parquet`, `summary.json`, and `manifest.json`.

## Pages

- **Home** — dataset catalog
- **Metrics** — three plot sets
- **Dataset summary** — cells, genes, cell types, batches

## Tests

```bash
make test   # includes dashboard tests via PYTHONPATH=metrics_dashboard
```
