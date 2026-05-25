"""Metrics plots — manipulation sweeps, correlations, collapse/shift."""

from __future__ import annotations

import bootstrap  # noqa: F401

import logging
import time
import traceback

import streamlit as st

from metrics_dashboard.catalog import discover_datasets
from metrics_dashboard.config import (
    DASHBOARD_METRICS,
    MANIPULATION_ORDER,
    PLOT_SET_DESCRIPTIONS,
    bundle_root,
)
from metrics_dashboard.filters import render_sidebar_controls
from metrics_dashboard.load import load_multi_dataset_metrics
from metrics_dashboard.plots import plot_set1_grid, plot_set2_correlation, plot_set3_row
from metrics_dashboard.runtime import log_startup_context
from metrics_dashboard.style import configure_matplotlib
from metrics_dashboard.transforms import (
    average_metrics_across_datasets,
    prepare_set1_grid,
    prepare_set2_correlation,
    prepare_set3_embedding,
)

log = logging.getLogger("scfm_dashboard.metrics_page")
log_startup_context()
configure_matplotlib()

try:
    root = bundle_root()
    log.info("Metrics page start root=%s", root)
    datasets = discover_datasets(root)
    if not datasets:
        st.error(f"No data in `{root}`. Export bundles and commit `data/dashboard_bundles/`.")
        st.stop()

    controls = render_sidebar_controls(datasets)
    if controls is None:
        st.stop()

    with st.status("Loading metrics…", expanded=True) as status:
        t0 = time.perf_counter()
        metrics_df = load_multi_dataset_metrics(controls.dataset_ids, controls.models, root)
        status.update(label=f"Loaded {len(metrics_df):,} rows in {time.perf_counter() - t0:.1f}s")

    if metrics_df.empty:
        st.error("No metrics rows for the selected datasets and models.")
        st.stop()

    if len(controls.dataset_ids) > 1:
        metrics_df = average_metrics_across_datasets(metrics_df)
        st.caption(
            f"Averaging **{len(controls.dataset_ids)}** datasets · **{len(metrics_df):,}** rows · "
            f"models: {', '.join(controls.models)}"
        )
    else:
        st.caption(
            f"**{controls.dataset_ids[0]}** · **{len(metrics_df):,}** rows · "
            f"models: {', '.join(controls.models)}"
        )

    spec = DASHBOARD_METRICS[controls.metric_key]
    st.header(spec.label)
    st.markdown(spec.description)

    with st.expander("How to read these plots", expanded=False):
        st.markdown(f"**Set 1:** {PLOT_SET_DESCRIPTIONS['set1']}")
        st.markdown(f"**Set 2:** {PLOT_SET_DESCRIPTIONS['set2']}")
        st.markdown(f"**Set 3:** {PLOT_SET_DESCRIPTIONS['set3']}")

    st.subheader("Set 1 — Manipulation sweeps")
    with st.spinner("Building Set 1 plot…"):
        t0 = time.perf_counter()
        layout1 = prepare_set1_grid(metrics_df, spec, controls.models)
        ncol = max((len(c) for c in layout1.col_labels_by_row.values()), default=1)
        log.info(
            "set1 grid %d x %d (max cols) prepared in %.2fs",
            len(layout1.row_labels),
            ncol,
            time.perf_counter() - t0,
        )
        if layout1.data.empty:
            st.info("No rows for this metric.")
        else:
            k_note = ""
            if "k" in layout1.data.columns and layout1.data["k"].notna().any():
                k_vals = sorted(layout1.data["k"].dropna().unique())
                if len(k_vals) == 1:
                    k_note = f" · k = {int(k_vals[0]) if k_vals[0] == int(k_vals[0]) else k_vals[0]}"
            st.caption(
                f"Set 1: **{len(layout1.row_labels)}** manipulations × config columns "
                f"(up to **{ncol}** per row) · x-axis = **{layout1.x_col}**{k_note}."
            )
            fig1 = plot_set1_grid(layout1, spec, controls.models)
            log.info("set1 figure built in %.2fs", time.perf_counter() - t0)
            st.pyplot(fig1, clear_figure=True, use_container_width=True)

    st.subheader("Set 2 — Integration vs metric score")
    with st.spinner("Building Set 2 plot…"):
        t0 = time.perf_counter()
        wide = prepare_set2_correlation(metrics_df, spec, controls.models)
        if wide.empty or "metric_score" not in wide.columns:
            st.info("Insufficient data for correlation plots.")
        else:
            fig2 = plot_set2_correlation(
                wide,
                y_col="cell_type_score",
                y_label="Cell-type ASW",
                x_label=spec.label,
                models=controls.models,
            )
            log.info("set2 figure built in %.2fs", time.perf_counter() - t0)
            st.pyplot(fig2, clear_figure=True, use_container_width=True)

    st.subheader("Set 3 — Embedding collapse and shift")
    with st.spinner("Building Set 3 plot…"):
        t0 = time.perf_counter()
        collapse_df, shift_df = prepare_set3_embedding(metrics_df, controls.models)
        manipulations = [
            m
            for m in MANIPULATION_ORDER
            if m in set(collapse_df["intervention_name"].dropna().astype(str))
        ]
        if not manipulations:
            st.info("No embedding shift data for selected models.")
        else:
            fig3 = plot_set3_row(collapse_df, shift_df, manipulations, controls.models)
            log.info("set3 figure built in %.2fs", time.perf_counter() - t0)
            st.pyplot(fig3, clear_figure=True, use_container_width=True)

    log.info("Metrics page render complete")

except Exception:
    log.exception("Metrics page failed")
    st.error("The metrics page failed to render. See Streamlit Cloud logs for details.")
    with st.expander("Error details"):
        st.code(traceback.format_exc())
