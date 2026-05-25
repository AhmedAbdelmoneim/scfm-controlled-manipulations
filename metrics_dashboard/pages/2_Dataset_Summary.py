"""Dataset summary — cells, genes, cell types, batches."""

from __future__ import annotations

import bootstrap  # noqa: F401

import logging

import streamlit as st

from metrics_dashboard.runtime import log_startup_context

log = logging.getLogger("scfm_dashboard.summary_page")
log_startup_context()

from metrics_dashboard.catalog import catalog_table, discover_datasets
from metrics_dashboard.config import bundle_root
from metrics_dashboard.load import load_dataset_summary
from metrics_dashboard.style import render_theme_sidebar

render_theme_sidebar()

root = bundle_root()
datasets = discover_datasets(root)

st.title("Dataset summary")
st.markdown("Counts from exported `summary.json` in each bundle.")

if not datasets:
    st.warning(f"No bundles under `{root}`.")
    st.stop()

import pandas as pd

rows = [load_dataset_summary(ds, root) for ds in datasets]
df = pd.DataFrame(rows)
cols = ["dataset_id", "n_cells", "n_genes", "n_cell_types", "n_batches"]
for extra in ("cell_type_column", "batch_column", "cell_type_column_configured"):
    if extra in df.columns:
        cols.append(extra)
if "error" in df.columns:
    cols.append("error")
st.dataframe(df[[c for c in cols if c in df.columns]], use_container_width=True, hide_index=True)

with st.expander("Field definitions"):
    st.markdown(
        """
- **n_cells**: observations in the reference atlas
- **n_genes**: features in reference
- **n_cell_types**: unique cell-type labels (from `cell_type`, `celltype`, etc.)
- **n_batches**: unique batch labels (from `batch`, `sample_id`, etc.)
- **cell_type_column** / **batch_column**: which `obs` columns were matched when exported
        """
    )

st.subheader("Catalog")
st.dataframe(
    pd.DataFrame(
        [
            {
                "dataset": s.dataset_id,
                "models": ", ".join(s.models),
                "n_cells": s.n_cells,
            }
            for s in catalog_table(root)
        ]
    ),
    use_container_width=True,
    hide_index=True,
)
