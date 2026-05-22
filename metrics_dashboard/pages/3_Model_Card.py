"""Cross-dataset model card — aggregate metrics per model."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from metrics_dashboard.catalog import catalog_table, discover_datasets
from metrics_dashboard.config import MODEL_ORDER
from metrics_dashboard.load import (
    _cross_dataset_cache_key,
    load_model_metrics_across_datasets_cached,
)
from metrics_dashboard.plots import filter_metrics, plot_heatmap, plot_metric_lines
from metrics_dashboard.ui import init_session_root

st.set_page_config(page_title="Model Card", layout="wide")
st.title("Model card")

root = init_session_root()
statuses = catalog_table(root)
ready_datasets = [s.dataset_id for s in statuses if s.has_evaluation]

all_models = sorted(
    {m for s in statuses for m in s.models},
    key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99,
)

if not all_models:
    st.warning("No evaluation artifacts found yet.")
    st.stop()

selected_models = st.multiselect(
    "Model(s)",
    all_models,
    default=all_models[:1],
)

st.subheader("Coverage")
coverage_rows = []
for s in statuses:
    for model in selected_models:
        has = model in s.models
        coverage_rows.append(
            {
                "dataset": s.dataset_id,
                "model": model,
                "status": "ready" if has else "pending",
                "rows": "—",
            }
        )
st.dataframe(pd.DataFrame(coverage_rows), use_container_width=True, hide_index=True)

if not ready_datasets:
    st.stop()

metric_category = st.selectbox(
    "Metric category",
    [
        "knn_metrics",
        "embedding_shift",
        "embedding_stats",
        "clustering_metrics",
        "embedding_shift_gain",
        "knn_metrics_gain",
        "cell_type_and_batch_metrics",
    ],
)

combined_frames: list[pd.DataFrame] = []
for model in selected_models:
    ds_tuple = tuple(ready_datasets)
    df = load_model_metrics_across_datasets_cached(
        model,
        ds_tuple,
        str(root),
        _cross_dataset_cache_key(model, ds_tuple, root),
    )
    if not df.empty:
        combined_frames.append(df)

if not combined_frames:
    st.error("No metrics loaded for selected models.")
    st.stop()

combined = pd.concat(combined_frames, ignore_index=True)
cat_df = combined[combined["metric_category"] == metric_category]
metric_names = sorted(cat_df["metric_name"].dropna().unique())
if not metric_names:
    st.warning(f"No rows for category {metric_category}.")
    st.stop()

metric_name = st.selectbox("Metric", metric_names)
met_df = cat_df[cat_df["metric_name"] == metric_name]
spaces = sorted(met_df["space"].dropna().unique())
space = st.selectbox("Space", spaces)
met_df = met_df[met_df["space"] == space]

k_val = t_val = res_val = None
if metric_category in ("knn_metrics", "knn_metrics_gain") and met_df["k"].notna().any():
    k_val = st.selectbox("k", sorted(met_df["k"].dropna().unique()))
    met_df = met_df[met_df["k"] == k_val]
if metric_category == "knn_metrics" and metric_name in ("diffusion_js", "diffusion_sym_kl"):
    if met_df["diffusion_t"].notna().any():
        t_val = st.selectbox("diffusion_t", sorted(met_df["diffusion_t"].dropna().unique()))
        met_df = met_df[met_df["diffusion_t"] == t_val]
if metric_category == "clustering_metrics" and met_df["resolution"].notna().any():
    res_val = st.selectbox("resolution", sorted(met_df["resolution"].dropna().unique()))
    met_df = met_df[met_df["resolution"] == res_val]

y_col = st.selectbox("Y column", ["value_mean", "value_median"], index=0)

st.subheader("Scorecard (quick summary)")
scorecard_rows = []
for ds in ready_datasets:
    for model in selected_models:
        ds_tuple = (ds,)
        mdf = load_model_metrics_across_datasets_cached(
            model,
            ds_tuple,
            str(root),
            _cross_dataset_cache_key(model, ds_tuple, root),
        )
        if mdf.empty:
            continue
        knn = filter_metrics(
            mdf,
            metric_category="knn_metrics",
            metric_names=["knn_recall"],
            spaces=["embedding"],
            y_col="value_mean",
        )
        if "k" in knn.columns:
            knn = knn[knn["k"] == 15]
        knn_mean = knn["value_mean"].mean() if not knn.empty else float("nan")

        clust = filter_metrics(
            mdf,
            metric_category="clustering_metrics",
            metric_names=["leiden_ari"],
            spaces=["embedding"],
            y_col="value_mean",
        )
        if "resolution" in clust.columns:
            clust = clust[clust["resolution"] == 1.0]
        ari_mean = clust["value_mean"].mean() if not clust.empty else float("nan")

        scorecard_rows.append(
            {
                "dataset": ds,
                "model": model,
                "knn_recall_emb_k15_mean": knn_mean,
                "leiden_ari_res1_mean": ari_mean,
            }
        )
if scorecard_rows:
    st.dataframe(pd.DataFrame(scorecard_rows), use_container_width=True, hide_index=True)

st.subheader("Heatmap (dataset × intervention)")
if met_df["value_mean"].notna().any():
    agg = (
        met_df.groupby(["dataset_id", "intervention_name"], observed=True)[y_col]
        .mean()
        .reset_index()
    )
    fig = plot_heatmap(
        agg,
        index="dataset_id",
        columns="intervention_name",
        values=y_col,
        title=f"{metric_category} / {metric_name} / {space}",
        figsize=(12, max(3, len(ready_datasets) * 0.5)),
    )
    st.pyplot(fig, clear_figure=True)
else:
    st.warning("All values are null for this selection.")

st.subheader("Lines by dataset")
for ds in sorted(met_df["dataset_id"].dropna().unique()):
    sub = met_df[met_df["dataset_id"] == ds]
    if sub.empty:
        continue
    x = "resolution" if metric_category == "clustering_metrics" else "param_value"
    fig = plot_metric_lines(
        sub,
        x=x,
        y=y_col,
        hue="model",
        col="intervention_name",
        title=f"{ds} — {metric_name}",
    )
    st.pyplot(fig, clear_figure=True)
