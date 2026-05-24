"""ScFMs Metrics Dashboard — main exploration view."""

from __future__ import annotations

import streamlit as st

from metrics_dashboard.catalog import discover_datasets
from metrics_dashboard.config import (
    DASHBOARD_METRICS,
    MANIPULATION_ORDER,
    PLOT_SET_DESCRIPTIONS,
)
from metrics_dashboard.dashboard_filters import render_dashboard_controls
from metrics_dashboard.load import load_multi_dataset_metrics_cached
from metrics_dashboard.plots import plot_set1_grid, plot_set2_correlation, plot_set3_row
from metrics_dashboard.transforms import (
    average_metrics_across_datasets,
    prepare_set1_grid,
    prepare_set2_correlation,
    prepare_set3_embedding,
)
from metrics_dashboard.ui import init_session_root

st.set_page_config(page_title="ScFMs Metrics Dashboard", layout="wide")

root = init_session_root()
datasets = discover_datasets(root)

if not datasets:
    st.warning("No datasets under artifacts root.")
    st.stop()

controls = render_dashboard_controls(datasets, root)
if controls is None:
    st.stop()

cache_key = "|".join(sorted(controls.dataset_ids)) + "|" + ",".join(sorted(controls.models))
metrics_df = load_multi_dataset_metrics_cached(
    tuple(sorted(controls.dataset_ids)),
    tuple(sorted(controls.models)),
    str(root),
    cache_key,
)
if metrics_df.empty:
    st.error("No metrics CSVs found for the selected datasets and models.")
    st.stop()

if len(controls.dataset_ids) > 1:
    metrics_df = average_metrics_across_datasets(metrics_df)
    st.caption(
        f"Averaging metrics across **{len(controls.dataset_ids)}** datasets · "
        f"**{len(metrics_df):,}** rows · models: {', '.join(controls.models)}"
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
    st.markdown(f"**Set 1 — Manipulation sweeps:** {PLOT_SET_DESCRIPTIONS['set1']}")
    st.markdown(f"**Set 2 — Integration correlations:** {PLOT_SET_DESCRIPTIONS['set2']}")
    st.markdown(f"**Set 3 — Collapse & shift:** {PLOT_SET_DESCRIPTIONS['set3']}")

st.subheader("Set 1 — Manipulation sweeps")
sub1, row_labels, col_labels, facet_col = prepare_set1_grid(
    metrics_df, spec, controls.models
)
if sub1.empty:
    st.info("No rows for this metric with current filters.")
else:
    st.pyplot(
        plot_set1_grid(sub1, spec, row_labels, col_labels, facet_col, controls.models),
        clear_figure=True,
        use_container_width=True,
    )

st.subheader("Set 2 — Integration vs metric score")
wide = prepare_set2_correlation(metrics_df, spec, controls.models)
if wide.empty or "metric_score" not in wide.columns:
    st.info("Insufficient data for correlation plots.")
else:
    st.pyplot(
        plot_set2_correlation(
            wide,
            y_col="cell_type_score",
            y_label="Cell-type ASW",
            x_label=spec.label,
            models=controls.models,
        ),
        clear_figure=True,
        use_container_width=True,
    )

st.subheader("Set 3 — Embedding collapse and shift")
collapse_df, shift_df = prepare_set3_embedding(metrics_df, controls.models)
manipulations = [
    i
    for i in collapse_df["intervention_name"].dropna().unique()
    if str(i) != "reference"
]
manipulations = [m for m in MANIPULATION_ORDER if m in manipulations] + sorted(
    set(manipulations) - set(MANIPULATION_ORDER)
)
if not manipulations:
    st.info("No embedding shift data for selected models.")
else:
    st.pyplot(
        plot_set3_row(collapse_df, shift_df, manipulations, controls.models),
        clear_figure=True,
        use_container_width=True,
    )
