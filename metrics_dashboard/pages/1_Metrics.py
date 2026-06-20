"""Metrics plots: main sweeps, R_NX curves, and embedding shift/collapse."""

from __future__ import annotations

import bootstrap  # noqa: F401

import logging
import time
import traceback

import streamlit as st

from metrics_dashboard.catalog import discover_datasets
from metrics_dashboard.config import DEFAULT_PLOT_SCALE, MANIPULATION_ORDER, PLOT_SET_DESCRIPTIONS, bundle_root
from metrics_dashboard.filters import render_sidebar_controls
from metrics_dashboard.load import load_multi_dataset_metrics
from metrics_dashboard.plot_display import show_figure
from metrics_dashboard.plotly_charts import (
    plot_set1_main_metrics_plotly,
    plot_set2_rnx_curves_plotly,
    plot_set3_row_plotly,
)
from metrics_dashboard.runtime import log_startup_context
from metrics_dashboard.transforms import (
    average_metrics_across_datasets,
    prepare_set1_main_metrics,
    prepare_set2_rnx_curves,
    prepare_set3_embedding,
)

log = logging.getLogger("scfm_dashboard.metrics_page")
log_startup_context()

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

    st.header("Structure metrics")
    st.markdown(
        "Dashboard views focus on structure preservation and embedding shift/collapse. "
        "scIB bio/batch metrics are intentionally excluded from this dashboard."
    )

    with st.expander("How to read these plots", expanded=False):
        st.markdown(f"**Set 1:** {PLOT_SET_DESCRIPTIONS['set1']}")
        st.markdown(f"**Set 2:** {PLOT_SET_DESCRIPTIONS['set2']}")
        st.markdown(f"**Set 3:** {PLOT_SET_DESCRIPTIONS['set3']}")

    st.subheader("Set 1 — Main metrics")
    with st.spinner("Building Set 1 plot…"):
        t0 = time.perf_counter()
        layout1 = prepare_set1_main_metrics(metrics_df, controls.models)
        log.info(
            "set1 main metrics %d metrics x %d manipulations prepared in %.2fs",
            len(layout1.metric_labels),
            len(layout1.manipulations),
            time.perf_counter() - t0,
        )
        if layout1.data.empty:
            st.info("No rows for the main metrics.")
        else:
            st.caption(
                f"Set 1: **{len(layout1.metric_labels)}** metric rows × "
                f"**{len(layout1.manipulations)}** manipulation columns · "
                "x-axis = manipulation parameter; y-axis ranges are fixed by metric."
            )
            fig1 = plot_set1_main_metrics_plotly(layout1, controls.models, scale=DEFAULT_PLOT_SCALE)
            log.info("set1 figure built in %.2fs", time.perf_counter() - t0)
            show_figure(fig1)

    st.subheader("Set 2 — R_NX curves")
    with st.spinner("Building Set 2 plot…"):
        t0 = time.perf_counter()
        rnx_layout = prepare_set2_rnx_curves(metrics_df, controls.models)
        if rnx_layout.data.empty:
            st.info("No R_NX curve rows for the selected datasets and models.")
        else:
            st.caption(
                f"Set 2: **{len(rnx_layout.manipulations)}** manipulation rows; "
                "columns are manipulation parameter values; y-axis defaults to R_NX score range."
            )
            fig2 = plot_set2_rnx_curves_plotly(rnx_layout, controls.models, scale=DEFAULT_PLOT_SCALE)
            log.info("set2 figure built in %.2fs", time.perf_counter() - t0)
            show_figure(fig2)

    st.subheader("Set 3 — Embedding shift and collapse")
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
            fig3 = plot_set3_row_plotly(
                collapse_df,
                shift_df,
                manipulations,
                controls.models,
                scale=DEFAULT_PLOT_SCALE,
            )
            log.info("set3 figure built in %.2fs", time.perf_counter() - t0)
            show_figure(fig3)

    log.info("Metrics page render complete")

except Exception:
    log.exception("Metrics page failed")
    st.error("The metrics page failed to render. See Streamlit Cloud logs for details.")
    with st.expander("Error details"):
        st.code(traceback.format_exc())
