"""Shared Streamlit UI helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from metrics_dashboard.catalog import catalog_table, discover_datasets
from metrics_dashboard.filters import render_artifacts_root_sidebar
from metrics_dashboard.load import _eval_cache_key, load_dataset_metrics_cached


def get_root() -> Path:
    if "artifacts_root" not in st.session_state:
        from metrics_dashboard.config import artifacts_root

        st.session_state.artifacts_root = str(artifacts_root())
    return Path(st.session_state.artifacts_root)


def init_session_root() -> Path:
    root = render_artifacts_root_sidebar()
    st.session_state.artifacts_root = str(root)
    return root


def load_metrics_for_dataset(dataset_id: str, models: list[str], root: Path) -> pd.DataFrame:
    model_tuple = tuple(sorted(models))
    version = _eval_cache_key(dataset_id, model_tuple, root)
    return load_dataset_metrics_cached(dataset_id, model_tuple, str(root), version)


def render_catalog_summary(root: Path) -> None:
    statuses = catalog_table(root)
    if not statuses:
        st.info("No datasets under artifacts root.")
        return
    rows = [
        {
            "dataset": s.dataset_id,
            "metric_csvs": s.n_metric_csvs,
            "models": ", ".join(s.models) if s.models else "—",
            "last_modified": s.last_modified.isoformat() if s.last_modified else "—",
            "status": "ready" if s.has_evaluation else "pending",
        }
        for s in statuses
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
