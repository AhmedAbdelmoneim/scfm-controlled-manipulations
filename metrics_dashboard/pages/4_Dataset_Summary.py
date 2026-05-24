"""Dataset summary — cells, genes, cell types, batches."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from metrics_dashboard.catalog import catalog_table, discover_datasets
from metrics_dashboard.load import load_dataset_summary_cached, reference_h5ad_path
from metrics_dashboard.ui import init_session_root

st.set_page_config(page_title="Dataset summary", layout="wide")
st.title("Dataset summary")
st.markdown(
    "Per-atlas counts from `reference.h5ad` in the manipulations folder. "
    "Use this to understand dataset scale before interpreting metrics."
)

root = init_session_root()
datasets = discover_datasets(root)

if not datasets:
    st.warning("No datasets under artifacts root.")
    st.stop()

selected = st.multiselect("Datasets", datasets, default=datasets[:1] if datasets else [])
if not selected:
    st.info("Select at least one dataset.")
    st.stop()

rows: list[dict] = []
for ds in selected:
    ref_path = reference_h5ad_path(ds, root)
    mtime = ref_path.stat().st_mtime_ns if ref_path.is_file() else 0
    summary = load_dataset_summary_cached(ds, str(root), mtime)
    rows.append(summary)

df = pd.DataFrame(rows)
display_cols = ["dataset_id", "n_cells", "n_genes", "n_cell_types", "n_batches"]
if "error" in df.columns:
    display_cols.append("error")
st.dataframe(df[[c for c in display_cols if c in df.columns]], use_container_width=True, hide_index=True)

with st.expander("Field definitions"):
    st.markdown(
        """
- **n_cells**: observations in the reference AnnData object.
- **n_genes**: variables (features) in reference.
- **n_cell_types**: unique values in the `cell_type` obs column (0 if missing).
- **n_batches**: unique values in the `batch` obs column (0 if missing).
        """
    )

st.subheader("Catalog")
statuses = catalog_table(root)
if statuses:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "dataset": s.dataset_id,
                    "metric_csvs": s.n_metric_csvs,
                    "models": ", ".join(s.models),
                    "status": "ready" if s.has_evaluation else "pending",
                }
                for s in statuses
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
