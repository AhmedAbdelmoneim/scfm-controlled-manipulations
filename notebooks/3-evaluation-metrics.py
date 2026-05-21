import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Structure evaluation (notebook)

    One atlas, one model, one intervention family: **embedding_stats**,
    **embedding_shift**, and **knn_metrics** (neighborhood overlap + diffusion),
    then bar plots per manipulation.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import anndata as ad
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import seaborn as sns

    from scfm_controlled_manipulations.evaluation.data import (
        AlignedBundle,
        _as_float_csr as as_float_csr,
    )
    from scfm_controlled_manipulations.evaluation.context import (
        DatasetEvaluateContext,
        ModelEvaluateContext,
    )
    from scfm_controlled_manipulations.evaluation.metrics_knn import compute_knn_metrics
    from scfm_controlled_manipulations.evaluation.metrics_stats_shift import (
        compute_embedding_shift,
        compute_embedding_stats,
    )
    from scfm_controlled_manipulations.evaluation.reference_stats_shift import (
        precompute_reference_stats_shift,
    )
    from scfm_controlled_manipulations.io import evaluation_cache_dir

    sns.set_theme(style="whitegrid")
    return (
        AlignedBundle,
        DatasetEvaluateContext,
        ModelEvaluateContext,
        Path,
        ad,
        as_float_csr,
        compute_embedding_shift,
        compute_embedding_stats,
        compute_knn_metrics,
        evaluation_cache_dir,
        mo,
        np,
        pd,
        plt,
        precompute_reference_stats_shift,
        sp,
    )


@app.cell
def _(Path, evaluation_cache_dir):
    cfg_base_dir = Path("/vault/amoneim/scfm-controlled-manipulations/processed")
    cfg_atlas = "immune"
    cfg_model = "pca"
    cfg_seed = 42
    cfg_intervention_family = "local_smoothing"
    cfg_stats_shift_pairwise_cell_subsample_n = 500
    cfg_stats_shift_pairwise_max_pairs = 10_000
    cfg_k_values = [15, 30]
    cfg_distance_metrics = ["euclidean"]
    cfg_diffusion_t_values = [1, 2, 4, 8, 16, 32]
    cfg_knn_alpha = 10.0
    cfg_knn_bandwidth_k = None
    cfg_knn_n_null_permutations = 1
    cfg_eval_cache_dir = evaluation_cache_dir(
        cfg_base_dir / cfg_atlas / "results"
    )
    return (
        cfg_atlas,
        cfg_base_dir,
        cfg_diffusion_t_values,
        cfg_distance_metrics,
        cfg_eval_cache_dir,
        cfg_intervention_family,
        cfg_knn_alpha,
        cfg_knn_bandwidth_k,
        cfg_knn_n_null_permutations,
        cfg_k_values,
        cfg_model,
        cfg_seed,
        cfg_stats_shift_pairwise_cell_subsample_n,
        cfg_stats_shift_pairwise_max_pairs,
    )


@app.cell
def _(
    AlignedBundle,
    DatasetEvaluateContext,
    ModelEvaluateContext,
    ad,
    as_float_csr,
    np,
    pd,
    precompute_reference_stats_shift,
    sp,
):
    def intervention_param_columns(params_dict):
        columns = {}
        for key, val in params_dict.items():
            if key.startswith("operator"):
                continue
            if isinstance(val, (list, np.ndarray)) and len(val) >= 20:
                continue
            columns[f"intervention_{key}"] = val
        return columns

    def make_aligned_bundle(raw_ref_mat, raw_man_mat, emb_ref_mat, emb_man_mat, raw_ref_obs):
        return AlignedBundle(
            raw_ref=as_float_csr(raw_ref_mat),
            raw_man=as_float_csr(raw_man_mat),
            emb_ref=np.asarray(emb_ref_mat, dtype=np.float32),
            emb_man=np.asarray(emb_man_mat, dtype=np.float32),
            obs=raw_ref_obs,
        )

    def add_manipulation_columns(metrics_df, manipulation_item):
        out = metrics_df.copy()
        out["manipulation_id"] = manipulation_item["manipulation_id"]
        for col, val in intervention_param_columns(
            manipulation_item["intervention_params"]
        ).items():
            out[col] = val
        return out

    def run_metrics_for_family(*, compute_fn, manipulations, raw_ref_mat, emb_ref_mat, raw_ref_obs, **compute_kwargs):
        frames = []
        for item in manipulations:
            bundle = make_aligned_bundle(
                raw_ref_mat,
                item["raw_man_mat"],
                emb_ref_mat,
                item["emb_man_mat"],
                raw_ref_obs,
            )
            metrics_df = compute_fn(
                bundle=bundle,
                intervention_id=item["manipulation_id"],
                **compute_kwargs,
            )
            frames.append(add_manipulation_columns(metrics_df, item))
        return pd.concat(frames, ignore_index=True)

    def load_family_data(*, base_dir, atlas, model, intervention_family):
        manip_dir = base_dir / atlas / "results" / "manipulations"
        emb_dir = base_dir / atlas / "embeddings" / model

        def load_h5ad(path):
            adata = ad.read_h5ad(path)
            x = adata.X
            if sp.issparse(x):
                x = x.toarray()
            return (
                np.asarray(x, dtype=np.float32),
                adata.obs_names.copy(),
                adata.obs.copy(),
            )

        entries = []
        for path in sorted(manip_dir.glob(f"{intervention_family}_*.h5ad")):
            backed = ad.read_h5ad(path, backed="r")
            params = dict(
                backed.uns.get("scfm_intervention", {}).get(intervention_family, {})
            )
            backed.file.close()
            entries.append({"manipulation_id": path.stem, "intervention_params": params})
        if not entries:
            raise FileNotFoundError(f"No manipulations for {intervention_family} in {manip_dir}")

        sort_key = "intervention_k" if intervention_family == "local_smoothing" else "manipulation_id"

        def sort_val(e):
            cols = intervention_param_columns(e["intervention_params"])
            return cols.get(sort_key, e["manipulation_id"])

        entries.sort(key=sort_val)

        raw_ref, raw_obs_names, raw_obs = load_h5ad(manip_dir / "reference.h5ad")
        emb_ref, emb_obs_names, _ = load_h5ad(emb_dir / f"{model}_reference.h5ad")
        if not raw_obs_names.equals(emb_obs_names):
            raise ValueError("raw_ref and emb_ref obs_names differ")

        manipulations = []
        for entry in entries:
            mid = entry["manipulation_id"]
            raw_man, raw_man_obs, _ = load_h5ad(manip_dir / f"{mid}.h5ad")
            emb_man, emb_man_obs, _ = load_h5ad(emb_dir / f"{model}_{mid}.h5ad")
            if not raw_obs_names.equals(raw_man_obs) or not raw_obs_names.equals(emb_man_obs):
                raise ValueError(f"obs misaligned for {mid}")
            manipulations.append({
                **entry,
                "raw_man_mat": raw_man,
                "emb_man_mat": emb_man,
            })

        summary = (
            f"{atlas} | {model} | {intervention_family} | "
            f"{len(manipulations)} variants | {raw_ref.shape[0]} cells"
        )
        return manipulations, emb_ref, raw_ref, raw_obs, summary

    def build_eval_context(*, emb_ref_mat, raw_ref_mat, raw_ref_obs, seed, pairwise_cell_subsample_n, pairwise_max_pairs):
        model_ctx = ModelEvaluateContext(emb_ref=emb_ref_mat)
        dataset_ctx = DatasetEvaluateContext(
            raw_ref=as_float_csr(raw_ref_mat),
            obs=raw_ref_obs,
            n_cells=int(raw_ref_mat.shape[0]),
        )
        model_ctx.ref_stats_cache = precompute_reference_stats_shift(
            model_ctx,
            dataset_ctx,
            seed=seed,
            pairwise_cell_subsample_n=pairwise_cell_subsample_n,
            pairwise_max_pairs=pairwise_max_pairs,
        )
        cache = model_ctx.ref_stats_cache
        summary = (
            f"ref cache: n_sub={cache.pairwise_cell_indices.size} "
            f"raw_pairs={cache.raw_within_pairwise_l2.size} "
            f"emb_pairs={cache.emb_within_pairwise_l2.size}"
        )
        return model_ctx, dataset_ctx, summary

    def lookup_metric_row(
        metrics_df,
        manipulation_id,
        *,
        metric_category,
        metric_name,
        space,
        field="value_mean",
        distance_metric=None,
        k=None,
        diffusion_t=None,
    ):
        if metrics_df is None or metrics_df.empty:
            return float("nan")
        id_col = "manipulation_id" if "manipulation_id" in metrics_df.columns else "intervention_id"
        mask = (
            (metrics_df[id_col].astype(str) == str(manipulation_id))
            & (metrics_df["space"] == space)
            & (metrics_df["metric_category"] == metric_category)
            & (metrics_df["metric_name"] == metric_name)
        )
        if distance_metric is not None and "distance_metric" in metrics_df.columns:
            mask &= metrics_df["distance_metric"] == distance_metric
        if k is not None and "k" in metrics_df.columns:
            mask &= metrics_df["k"] == k
        if "diffusion_t" in metrics_df.columns:
            if diffusion_t is None:
                mask &= metrics_df["diffusion_t"].isna()
            else:
                mask &= metrics_df["diffusion_t"] == diffusion_t
        subset = metrics_df.loc[mask]
        if subset.empty or field not in subset.columns:
            return float("nan")
        val = subset.iloc[0][field]
        return float("nan") if pd.isna(val) else float(val)

    def tidy_embedding_stats(metrics_df, manipulations):
        specs = [
            ("l2_norm", "mean_row_l2_norm_ref", "mean_row_l2_norm_man"),
            ("mean_per_dim", "col_mean_ref", "col_mean_man"),
            ("var_per_dim", "col_variance_ref", "col_variance_man"),
        ]

        def row(item):
            mid = item["manipulation_id"]
            base = {"manipulation_id": mid, **intervention_param_columns(item["intervention_params"])}
            rows = []
            for space in ("raw", "embedding"):
                for label, ref_n, man_n in specs:
                    for state, name in (("ref", ref_n), ("manip", man_n)):
                        rows.append({
                            **base,
                            "space": space,
                            "metric": label,
                            "state": state,
                            "value_mean": lookup_metric_row(
                                metrics_df, mid, metric_category="embedding_stats",
                                metric_name=name, space=space,
                            ),
                            "value_std": lookup_metric_row(
                                metrics_df, mid, metric_category="embedding_stats",
                                metric_name=name, space=space, field="value_std",
                            ),
                        })
            return rows

        out = []
        for item in manipulations:
            out.extend(row(item))
        return pd.DataFrame(out)

    def tidy_embedding_shift(metrics_df, manipulations):
        specs = [
            ("within_ref_spread", "within_ref_pairwise_l2"),
            ("within_man_spread", "within_man_pairwise_l2"),
            ("shift_magnitude", "paired_cell_l2_norm"),
            ("shift_pairwise_cosine", "shift_pairwise_cosine"),
        ]
        rows = []
        for item in manipulations:
            mid = item["manipulation_id"]
            base = {"manipulation_id": mid, **intervention_param_columns(item["intervention_params"])}
            for space in ("raw", "embedding"):
                for label, name in specs:
                    rows.append({
                        **base,
                        "space": space,
                        "metric": label,
                        "value_mean": lookup_metric_row(
                            metrics_df, mid, metric_category="embedding_shift",
                            metric_name=name, space=space,
                        ),
                        "value_std": lookup_metric_row(
                            metrics_df, mid, metric_category="embedding_shift",
                            metric_name=name, space=space, field="value_std",
                        ),
                    })
        return pd.DataFrame(rows)

    def tidy_knn_metrics(metrics_df, manipulations, k_values, distance_metrics, diffusion_t_values):
        rows = []
        for item in manipulations:
            mid = item["manipulation_id"]
            base = {"manipulation_id": mid, **intervention_param_columns(item["intervention_params"])}
            for dm in distance_metrics:
                for k in k_values:
                    for space in ("raw", "embedding"):
                        rows.append({
                            **base,
                            "space": space,
                            "metric": "knn_recall",
                            "distance_metric": dm,
                            "k": int(k),
                            "diffusion_t": np.nan,
                            "value_mean": lookup_metric_row(
                                metrics_df, mid, metric_category="knn_metrics",
                                metric_name="knn_recall", space=space,
                                distance_metric=dm, k=int(k), diffusion_t=None,
                            ),
                            "value_std": lookup_metric_row(
                                metrics_df, mid, metric_category="knn_metrics",
                                metric_name="knn_recall", space=space, field="value_std",
                                distance_metric=dm, k=int(k), diffusion_t=None,
                            ),
                            "null_value": lookup_metric_row(
                                metrics_df, mid, metric_category="knn_metrics",
                                metric_name="knn_recall", space=space, field="null_value",
                                distance_metric=dm, k=int(k), diffusion_t=None,
                            ),
                        })
                    for t in diffusion_t_values:
                        for label, name in (
                            ("diffusion_sym_kl", "diffusion_sym_kl"),
                            ("diffusion_js", "diffusion_js"),
                        ):
                            rows.append({
                                **base,
                                "space": "embedding",
                                "metric": label,
                                "distance_metric": dm,
                                "k": int(k),
                                "diffusion_t": int(t),
                                "value_mean": lookup_metric_row(
                                    metrics_df, mid, metric_category="knn_metrics",
                                    metric_name=name, space="embedding",
                                    distance_metric=dm, k=int(k), diffusion_t=int(t),
                                ),
                                "value_std": lookup_metric_row(
                                    metrics_df, mid, metric_category="knn_metrics",
                                    metric_name=name, space="embedding", field="value_std",
                                    distance_metric=dm, k=int(k), diffusion_t=int(t),
                                ),
                                "null_value": lookup_metric_row(
                                    metrics_df, mid, metric_category="knn_metrics",
                                    metric_name=name, space="embedding", field="null_value",
                                    distance_metric=dm, k=int(k), diffusion_t=int(t),
                                ),
                            })
        return pd.DataFrame(rows)

    return (
        build_eval_context,
        intervention_param_columns,
        load_family_data,
        run_metrics_for_family,
        tidy_embedding_shift,
        tidy_embedding_stats,
        tidy_knn_metrics,
    )


@app.cell
def _(intervention_param_columns, np, pd, plt):
    def manipulation_inventory_df(manipulations):
        return pd.DataFrame([
            {"manipulation_id": m["manipulation_id"], **intervention_param_columns(m["intervention_params"])}
            for m in manipulations
        ])

    def x_col_from_df(df):
        return "intervention_k" if "intervention_k" in df.columns else "manipulation_id"

    def grouped_bars_with_errors(ax, df_subset, x_col, hue_col=None):
        x_values = list(df_subset[x_col].drop_duplicates())
        x_pos = np.arange(len(x_values))

        def safe_yerr(stds):
            return np.where(np.isnan(stds), 0.0, stds)

        if hue_col is None:
            ordered = df_subset.set_index(x_col).reindex(x_values)
            ax.bar(x_pos, ordered["value_mean"].values, yerr=safe_yerr(ordered["value_std"].values), capsize=4)
        else:
            hues = list(df_subset[hue_col].drop_duplicates())
            width = 0.8 / max(len(hues), 1)
            for i, hue in enumerate(hues):
                ordered = df_subset[df_subset[hue_col] == hue].set_index(x_col).reindex(x_values)
                ax.bar(
                    x_pos + (i - (len(hues) - 1) / 2) * width,
                    ordered["value_mean"].values,
                    width,
                    yerr=safe_yerr(ordered["value_std"].values),
                    capsize=3,
                    label=str(hue),
                )
            ax.legend(title=hue_col)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(v) for v in x_values], rotation=45, ha="right")
        ax.set_xlabel(x_col)

    def plot_metric_panels(df_tidy, *, title, spaces, metric_specs, layout, hue_col=None):
        x_col = x_col_from_df(df_tidy)
        nrows, ncols = layout
        figsize = (ncols * 4.5, nrows * 3.8)
        for space in spaces:
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            axes = np.atleast_1d(axes).ravel()
            fig.suptitle(f"{title} — {space}", y=1.02)
            for ax, (key, subtitle) in zip(axes, metric_specs):
                sub = df_tidy[(df_tidy["space"] == space) & (df_tidy["metric"] == key)]
                grouped_bars_with_errors(ax, sub, x_col, hue_col)
                ax.set_title(subtitle)
                ax.set_ylabel("value_mean")
            plt.tight_layout()
            plt.show()

    def plot_knn_overlap(df_tidy, k_values):
        x_col = x_col_from_df(df_tidy)
        overlap = df_tidy[df_tidy["metric"] == "knn_recall"]
        for k in k_values:
            for space in ("raw", "embedding"):
                fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                fig.suptitle(f"kNN recall — {space}, k={k}", y=1.02)
                sub = overlap[(overlap["space"] == space) & (overlap["k"] == k)]
                grouped_bars_with_errors(ax, sub, x_col)
                ax.set_ylabel("value_mean")
                plt.tight_layout()
                plt.show()

    def plot_knn_diffusion(df_tidy, k_values, diffusion_t_values):
        x_col = x_col_from_df(df_tidy)
        diff = df_tidy[df_tidy["metric"].isin(("diffusion_sym_kl", "diffusion_js"))]
        for k in k_values:
            for t in diffusion_t_values:
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                fig.suptitle(f"Diffusion — embedding, k={k}, t={t}", y=1.02)
                for ax, metric in zip(axes, ("diffusion_sym_kl", "diffusion_js")):
                    sub = diff[(diff["k"] == k) & (diff["diffusion_t"] == t) & (diff["metric"] == metric)]
                    grouped_bars_with_errors(ax, sub, x_col)
                    ax.set_title(metric)
                    ax.set_ylabel("value_mean")
                plt.tight_layout()
                plt.show()

    return (
        manipulation_inventory_df,
        plot_knn_diffusion,
        plot_knn_overlap,
        plot_metric_panels,
    )


@app.cell
def _(
    cfg_atlas,
    cfg_base_dir,
    cfg_intervention_family,
    cfg_model,
    load_family_data,
):
    data_manipulations, emb_ref_mat, raw_ref_mat, raw_ref_obs, load_summary = load_family_data(
        base_dir=cfg_base_dir,
        atlas=cfg_atlas,
        model=cfg_model,
        intervention_family=cfg_intervention_family,
    )
    print(load_summary)
    return data_manipulations, emb_ref_mat, raw_ref_mat, raw_ref_obs


@app.cell
def _(data_manipulations, manipulation_inventory_df):
    manipulation_inventory_df(data_manipulations)
    return


@app.cell
def _(
    build_eval_context,
    cfg_seed,
    cfg_stats_shift_pairwise_cell_subsample_n,
    cfg_stats_shift_pairwise_max_pairs,
    emb_ref_mat,
    raw_ref_mat,
    raw_ref_obs,
):
    notebook_model_ctx, notebook_dataset_ctx, ref_cache_summary = build_eval_context(
        emb_ref_mat=emb_ref_mat,
        raw_ref_mat=raw_ref_mat,
        raw_ref_obs=raw_ref_obs,
        seed=cfg_seed,
        pairwise_cell_subsample_n=cfg_stats_shift_pairwise_cell_subsample_n,
        pairwise_max_pairs=cfg_stats_shift_pairwise_max_pairs,
    )
    print(ref_cache_summary)
    return notebook_dataset_ctx, notebook_model_ctx


@app.cell
def _(
    cfg_atlas,
    cfg_intervention_family,
    cfg_model,
    cfg_seed,
    cfg_stats_shift_pairwise_max_pairs,
    compute_embedding_shift,
    compute_embedding_stats,
    data_manipulations,
    emb_ref_mat,
    notebook_model_ctx,
    raw_ref_mat,
    raw_ref_obs,
    run_metrics_for_family,
):
    common = dict(
        manipulations=data_manipulations,
        raw_ref_mat=raw_ref_mat,
        emb_ref_mat=emb_ref_mat,
        raw_ref_obs=raw_ref_obs,
        dataset_id=cfg_atlas,
        model=cfg_model,
        intervention_name=cfg_intervention_family,
        seed=cfg_seed,
    )
    df_embedding_stats = run_metrics_for_family(
        compute_fn=compute_embedding_stats,
        ref_cache=notebook_model_ctx.ref_stats_cache,
        **common,
    )
    df_embedding_shift = run_metrics_for_family(
        compute_fn=compute_embedding_shift,
        ref_cache=notebook_model_ctx.ref_stats_cache,
        pairwise_max_pairs=cfg_stats_shift_pairwise_max_pairs,
        **common,
    )
    print("stats:", sorted(df_embedding_stats["metric_name"].unique()))
    print("shift:", sorted(df_embedding_shift["metric_name"].unique()))
    return df_embedding_shift, df_embedding_stats


@app.cell
def _(
    cfg_atlas,
    cfg_diffusion_t_values,
    cfg_distance_metrics,
    cfg_eval_cache_dir,
    cfg_intervention_family,
    cfg_knn_alpha,
    cfg_knn_bandwidth_k,
    cfg_knn_n_null_permutations,
    cfg_k_values,
    cfg_model,
    cfg_seed,
    compute_knn_metrics,
    data_manipulations,
    emb_ref_mat,
    notebook_dataset_ctx,
    raw_ref_mat,
    raw_ref_obs,
    run_metrics_for_family,
):
    df_knn_metrics = run_metrics_for_family(
        compute_fn=compute_knn_metrics,
        manipulations=data_manipulations,
        raw_ref_mat=raw_ref_mat,
        emb_ref_mat=emb_ref_mat,
        raw_ref_obs=raw_ref_obs,
        dataset_id=cfg_atlas,
        model=cfg_model,
        intervention_name=cfg_intervention_family,
        seed=cfg_seed,
        distance_metrics=cfg_distance_metrics,
        k_values=cfg_k_values,
        diffusion_t_values=cfg_diffusion_t_values,
        cache_dir=cfg_eval_cache_dir,
        knn_cache=notebook_dataset_ctx.knn_cache,
        alpha=cfg_knn_alpha,
        bandwidth_k=cfg_knn_bandwidth_k,
        n_null_permutations=cfg_knn_n_null_permutations,
    )
    print("knn:", sorted(df_knn_metrics["metric_name"].unique()))
    return (df_knn_metrics,)


@app.cell
def _(
    cfg_diffusion_t_values,
    cfg_distance_metrics,
    cfg_k_values,
    data_manipulations,
    df_embedding_shift,
    df_embedding_stats,
    df_knn_metrics,
    tidy_embedding_shift,
    tidy_embedding_stats,
    tidy_knn_metrics,
):
    df_stats_tidy = tidy_embedding_stats(df_embedding_stats, data_manipulations)
    df_shift_tidy = tidy_embedding_shift(df_embedding_shift, data_manipulations)
    df_knn_tidy = tidy_knn_metrics(
        df_knn_metrics,
        data_manipulations,
        cfg_k_values,
        cfg_distance_metrics,
        cfg_diffusion_t_values,
    )
    return df_knn_tidy, df_shift_tidy, df_stats_tidy


@app.cell
def _(df_stats_tidy, plot_metric_panels):
    plot_metric_panels(
        df_stats_tidy,
        title="Embedding stats",
        spaces=("raw", "embedding"),
        metric_specs=[
            ("l2_norm", "Row L2 norm"),
            ("mean_per_dim", "Mean per dimension"),
            ("var_per_dim", "Variance per dimension"),
        ],
        layout=(1, 3),
        hue_col="state",
    )
    return


@app.cell
def _(df_shift_tidy, plot_metric_panels):
    plot_metric_panels(
        df_shift_tidy,
        title="Embedding shift",
        spaces=("raw", "embedding"),
        metric_specs=[
            ("within_ref_spread", "Within-ref pairwise L2"),
            ("within_man_spread", "Within-man pairwise L2"),
            ("shift_magnitude", "||manip − ref||"),
            ("shift_pairwise_cosine", "Pairwise shift cosine"),
        ],
        layout=(2, 2),
    )
    return


@app.cell
def _(
    cfg_diffusion_t_values,
    cfg_k_values,
    df_knn_tidy,
    plot_knn_diffusion,
    plot_knn_overlap,
):
    plot_knn_overlap(df_knn_tidy, cfg_k_values)
    plot_knn_diffusion(df_knn_tidy, cfg_k_values, cfg_diffusion_t_values)
    return


if __name__ == "__main__":
    app.run()
