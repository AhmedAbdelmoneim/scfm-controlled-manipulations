import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Embedding stats and shift exploration

    This notebook runs the stats-and-shift evaluation pass on one atlas, one
    model, and one intervention family. It loads the manipulation family,
    precomputes a reference cache once, then computes `embedding_stats` and
    `embedding_shift` against that cache and plots the results.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports
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
    from scfm_controlled_manipulations.evaluation.metrics_stats_shift import (
        compute_embedding_shift,
        compute_embedding_stats,
    )
    from scfm_controlled_manipulations.evaluation.reference_stats_shift import (
        precompute_reference_stats_shift,
    )

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
        mo,
        np,
        pd,
        plt,
        precompute_reference_stats_shift,
        sp,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configuration
    """)
    return


@app.cell
def _(Path):
    cfg_base_dir = Path("/vault/amoneim/scfm-controlled-manipulations/processed")
    cfg_atlas = "immune"
    cfg_model = "pca"
    cfg_seed = 42
    cfg_intervention_family = "local_smoothing"
    cfg_stats_shift_pairwise_cell_subsample_n = 500
    cfg_stats_shift_pairwise_max_pairs = 10_000
    return (
        cfg_atlas,
        cfg_base_dir,
        cfg_intervention_family,
        cfg_model,
        cfg_seed,
        cfg_stats_shift_pairwise_cell_subsample_n,
        cfg_stats_shift_pairwise_max_pairs,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Shared helpers

    Small utilities reused across data loading, metric computation, and the
    wide-format conversion below.
    """)
    return


@app.cell
def _(np):
    def intervention_param_columns(params_dict):
        """Flatten intervention params into DataFrame-friendly columns.

        Drops operator entries and skips long array-like values that would not
        round-trip cleanly into a tabular form.
        """
        columns = {}
        for param_key, param_value in params_dict.items():
            if param_key.startswith("operator"):
                continue
            if isinstance(param_value, (list, np.ndarray)) and len(param_value) >= 20:
                continue
            columns[f"intervention_{param_key}"] = param_value
        return columns

    return (intervention_param_columns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data loading

    Reads the reference and manipulation `h5ad` files for one atlas, model,
    and intervention family, checks `obs_names` alignment, and returns the
    matrices that the bundles downstream expect.
    """)
    return


@app.cell
def _(ad, intervention_param_columns, np, sp):
    def load_family_data(*, base_dir, atlas, model, intervention_family):
        """Load reference and manipulation matrices for one intervention family."""

        def discover_family_manipulations(manip_dir, family):
            discovered = []
            for candidate_path in sorted(manip_dir.glob(f"{family}_*.h5ad")):
                adata_backed = ad.read_h5ad(candidate_path, backed="r")
                params_dict = dict(
                    adata_backed.uns.get("scfm_intervention", {}).get(family, {})
                )
                adata_backed.file.close()
                discovered.append({
                    "manipulation_id": candidate_path.stem,
                    "intervention_params": params_dict,
                })
            if len(discovered) == 0:
                raise FileNotFoundError(
                    f"No manipulations found for family={family} in {manip_dir}"
                )

            sort_key = (
                "intervention_k" if family == "local_smoothing" else "manipulation_id"
            )

            def sort_value(entry):
                param_columns = intervention_param_columns(entry["intervention_params"])
                if sort_key in param_columns:
                    return param_columns[sort_key]
                return entry["manipulation_id"]

            discovered.sort(key=sort_value)
            return discovered

        def load_matrix_from_h5ad(h5ad_path):
            adata_obj = ad.read_h5ad(h5ad_path)
            matrix_obj = adata_obj.X
            if sp.issparse(matrix_obj):
                matrix_obj = matrix_obj.toarray()
            matrix_arr = np.asarray(matrix_obj, dtype=np.float32)
            obs_names = adata_obj.obs_names.copy()
            obs_df = adata_obj.obs.copy()
            return matrix_arr, obs_names, obs_df

        def assert_obs_names_equal(obs_names_a, obs_names_b, label_a, label_b):
            if not obs_names_a.equals(obs_names_b):
                raise ValueError(
                    f"obs_names are not aligned between {label_a} and {label_b}"
                )

        manip_dir = base_dir / atlas / "results" / "manipulations"
        emb_dir = base_dir / atlas / "embeddings" / model
        family_entries = discover_family_manipulations(manip_dir, intervention_family)

        raw_ref_mat, raw_ref_obs_names, raw_ref_obs = load_matrix_from_h5ad(
            manip_dir / "reference.h5ad"
        )
        emb_ref_mat, emb_ref_obs_names, _ = load_matrix_from_h5ad(
            emb_dir / f"{model}_reference.h5ad"
        )
        assert_obs_names_equal(
            raw_ref_obs_names, emb_ref_obs_names, "raw_ref", "emb_ref"
        )

        manipulations = []
        for family_entry in family_entries:
            manip_id = family_entry["manipulation_id"]
            raw_man_mat, raw_man_obs_names, _ = load_matrix_from_h5ad(
                manip_dir / f"{manip_id}.h5ad"
            )
            emb_man_mat, emb_man_obs_names, _ = load_matrix_from_h5ad(
                emb_dir / f"{model}_{manip_id}.h5ad"
            )
            assert_obs_names_equal(
                raw_ref_obs_names, raw_man_obs_names, "raw_ref", f"raw_{manip_id}"
            )
            assert_obs_names_equal(
                raw_ref_obs_names, emb_man_obs_names, "raw_ref", f"emb_{manip_id}"
            )
            manipulations.append({
                "manipulation_id": manip_id,
                "intervention_params": family_entry["intervention_params"],
                "raw_man_mat": raw_man_mat,
                "emb_man_mat": emb_man_mat,
            })

        summary = (
            f"Atlas: {atlas} | Model: {model} | Family: {intervention_family}\n"
            f"Variants: {len(manipulations)} | Cells: {raw_ref_mat.shape[0]}"
        )
        return manipulations, emb_ref_mat, raw_ref_mat, raw_ref_obs, summary

    return (load_family_data,)


@app.cell
def _(intervention_param_columns, pd):
    def manipulation_inventory_df(manipulations):
        """Tabular summary of intervention parameters across manipulations."""
        return pd.DataFrame([
            {
                "manipulation_id": item["manipulation_id"],
                **intervention_param_columns(item["intervention_params"]),
            }
            for item in manipulations
        ])

    return (manipulation_inventory_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reference cache

    Builds the model and dataset contexts and precomputes the reference
    statistics that every per-manipulation metric call below reuses.
    """)
    return


@app.cell
def _(
    DatasetEvaluateContext,
    ModelEvaluateContext,
    as_float_csr,
    precompute_reference_stats_shift,
):
    def build_reference_model_context(
        *,
        emb_ref_mat,
        raw_ref_mat,
        raw_ref_obs,
        seed,
        pairwise_cell_subsample_n,
        pairwise_max_pairs,
    ):
        """Construct the reference contexts and attach the precomputed cache."""
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
            f"Reference cache ready: n_sub={cache.pairwise_cell_indices.size} "
            f"raw_pairs={cache.raw_within_pairwise_l2.size} "
            f"emb_pairs={cache.emb_within_pairwise_l2.size}"
        )
        return model_ctx, summary

    return (build_reference_model_context,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Metric computation

    Applies `compute_embedding_stats` and `compute_embedding_shift` to every
    manipulation in the family. Both runners share the bundle constructor
    and the manipulation-column annotator defined first.
    """)
    return


@app.cell
def _(AlignedBundle, as_float_csr, intervention_param_columns, np):
    def make_aligned_bundle(
        raw_ref_mat, raw_man_mat, emb_ref_mat, emb_man_mat, raw_ref_obs
    ):
        return AlignedBundle(
            raw_ref=as_float_csr(raw_ref_mat),
            raw_man=as_float_csr(raw_man_mat),
            emb_ref=np.asarray(emb_ref_mat, dtype=np.float32),
            emb_man=np.asarray(emb_man_mat, dtype=np.float32),
            obs=raw_ref_obs,
        )

    def add_manipulation_columns(metrics_df, manipulation_item):
        annotated = metrics_df.copy()
        annotated["manipulation_id"] = manipulation_item["manipulation_id"]
        for column_name, column_value in intervention_param_columns(
            manipulation_item["intervention_params"]
        ).items():
            annotated[column_name] = column_value
        return annotated

    return add_manipulation_columns, make_aligned_bundle


@app.cell
def _(
    add_manipulation_columns,
    compute_embedding_stats,
    make_aligned_bundle,
    pd,
):
    def run_embedding_stats_for_family(
        *,
        manipulations,
        raw_ref_mat,
        emb_ref_mat,
        raw_ref_obs,
        ref_stats_cache,
        dataset_id,
        model,
        intervention_name,
        seed,
    ):
        output_frames = []
        for manipulation_item in manipulations:
            bundle = make_aligned_bundle(
                raw_ref_mat,
                manipulation_item["raw_man_mat"],
                emb_ref_mat,
                manipulation_item["emb_man_mat"],
                raw_ref_obs,
            )
            metrics_df = compute_embedding_stats(
                bundle=bundle,
                dataset_id=dataset_id,
                model=model,
                intervention_id=manipulation_item["manipulation_id"],
                intervention_name=intervention_name,
                seed=seed,
                ref_cache=ref_stats_cache,
            )
            output_frames.append(
                add_manipulation_columns(metrics_df, manipulation_item)
            )
        return pd.concat(output_frames, ignore_index=True)

    return (run_embedding_stats_for_family,)


@app.cell
def _(
    add_manipulation_columns,
    compute_embedding_shift,
    make_aligned_bundle,
    pd,
):
    def run_embedding_shift_for_family(
        *,
        manipulations,
        raw_ref_mat,
        emb_ref_mat,
        raw_ref_obs,
        ref_stats_cache,
        dataset_id,
        model,
        intervention_name,
        seed,
        pairwise_max_pairs,
    ):
        output_frames = []
        for manipulation_item in manipulations:
            bundle = make_aligned_bundle(
                raw_ref_mat,
                manipulation_item["raw_man_mat"],
                emb_ref_mat,
                manipulation_item["emb_man_mat"],
                raw_ref_obs,
            )
            metrics_df = compute_embedding_shift(
                bundle=bundle,
                dataset_id=dataset_id,
                model=model,
                intervention_id=manipulation_item["manipulation_id"],
                intervention_name=intervention_name,
                seed=seed,
                ref_cache=ref_stats_cache,
                pairwise_max_pairs=pairwise_max_pairs,
            )
            output_frames.append(
                add_manipulation_columns(metrics_df, manipulation_item)
            )
        return pd.concat(output_frames, ignore_index=True)

    return (run_embedding_shift_for_family,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tidy-format conversion

    Reshapes the long-format metric tables into one row per
    (manipulation, space, metric, [state]) with both `value_mean` and
    `value_std` columns, which is the form the plotters expect.
    """)
    return


@app.cell
def _(intervention_param_columns, pd):
    def lookup_metric_row(
        metrics_df,
        manipulation_id,
        space,
        metric_category,
        metric_name,
        field="value_mean",
    ):
        """Pull one scalar from the long metrics table (NaN if missing)."""
        if metrics_df is None or metrics_df.empty:
            return float("nan")
        key = str(manipulation_id)
        id_column = (
            "manipulation_id"
            if "manipulation_id" in metrics_df.columns
            else "intervention_id"
        )
        subset = metrics_df[
            (metrics_df[id_column].astype(str) == key)
            & (metrics_df["space"] == space)
            & (metrics_df["metric_category"] == metric_category)
            & (metrics_df["metric_name"] == metric_name)
        ]
        if subset.empty or field not in subset.columns:
            return float("nan")
        value = subset.iloc[0][field]
        return float("nan") if pd.isna(value) else float(value)

    def assert_embedding_stats_schema(metrics_df):
        """Fail fast when the stats table is stale or from an old package build."""
        required = {
            "col_mean_ref",
            "col_mean_man",
            "col_variance_ref",
            "col_variance_man",
            "mean_row_l2_norm_ref",
            "mean_row_l2_norm_man",
        }
        present = set(metrics_df["metric_name"].astype(str).unique())
        missing = sorted(required - present)
        if missing:
            raise RuntimeError(
                "df_embedding_stats is missing metric rows: "
                f"{missing}. Re-run imports and embedding_stats_run_cell "
                "(restart the kernel if the package was updated)."
            )

    def tidy_embedding_stats(metrics_df, manipulations):
        """Long-format stats: l2 norm, per-dim mean, per-dim variance (ref/man)."""
        assert_embedding_stats_schema(metrics_df)
        metric_specs = [
            ("l2_norm", "mean_row_l2_norm_ref", "mean_row_l2_norm_man"),
            ("mean_per_dim", "col_mean_ref", "col_mean_man"),
            ("var_per_dim", "col_variance_ref", "col_variance_man"),
        ]
        rows = []
        for manipulation_item in manipulations:
            manip_id = manipulation_item["manipulation_id"]
            param_cols = intervention_param_columns(
                manipulation_item["intervention_params"]
            )
            for space in ("raw", "embedding"):
                for metric_label, ref_name, man_name in metric_specs:
                    for state, raw_name in (("ref", ref_name), ("manip", man_name)):
                        rows.append({
                            "manipulation_id": manip_id,
                            **param_cols,
                            "space": space,
                            "metric": metric_label,
                            "state": state,
                            "value_mean": lookup_metric_row(
                                metrics_df,
                                manip_id,
                                space,
                                "embedding_stats",
                                raw_name,
                                field="value_mean",
                            ),
                            "value_std": lookup_metric_row(
                                metrics_df,
                                manip_id,
                                space,
                                "embedding_stats",
                                raw_name,
                                field="value_std",
                            ),
                        })
        return pd.DataFrame(rows)

    def tidy_embedding_shift(metrics_df, manipulations):
        """Long-format shift metrics (one row per manipulation, space, metric)."""
        metric_specs = [
            ("within_ref_spread", "within_ref_pairwise_l2"),
            ("within_man_spread", "within_man_pairwise_l2"),
            ("shift_magnitude", "paired_cell_l2_norm"),
            ("shift_pairwise_cosine", "shift_pairwise_cosine"),
        ]
        rows = []
        for manipulation_item in manipulations:
            manip_id = manipulation_item["manipulation_id"]
            param_cols = intervention_param_columns(
                manipulation_item["intervention_params"]
            )
            for space in ("raw", "embedding"):
                for metric_label, raw_name in metric_specs:
                    rows.append({
                        "manipulation_id": manip_id,
                        **param_cols,
                        "space": space,
                        "metric": metric_label,
                        "value_mean": lookup_metric_row(
                            metrics_df,
                            manip_id,
                            space,
                            "embedding_shift",
                            raw_name,
                            field="value_mean",
                        ),
                        "value_std": lookup_metric_row(
                            metrics_df,
                            manip_id,
                            space,
                            "embedding_shift",
                            raw_name,
                            field="value_std",
                        ),
                    })
        return pd.DataFrame(rows)

    return (
        assert_embedding_stats_schema,
        tidy_embedding_shift,
        tidy_embedding_stats,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting

    Two figures per evaluator, one for the raw space and one for the
    embedding space, with bars for each manipulation in the family and
    error bars taken from each metric's std.
    """)
    return


@app.cell
def _(np):
    def grouped_bars_with_errors(ax, df_subset, x_col, hue_col=None):
        """Draw bars from pre-computed `value_mean` and `value_std`.

        With `hue_col=None` draws one bar per `x_col` value. With a hue
        column draws grouped bars per hue level, side by side. NaN stds are
        treated as zero so missing error bars are simply omitted.
        """
        x_values = list(df_subset[x_col].drop_duplicates())
        x_positions = np.arange(len(x_values))

        def safe_yerr(stds):
            return np.where(np.isnan(stds), 0.0, stds)

        if hue_col is None:
            ordered = df_subset.set_index(x_col).reindex(x_values)
            ax.bar(
                x_positions,
                ordered["value_mean"].values,
                yerr=safe_yerr(ordered["value_std"].values),
                capsize=4,
            )
        else:
            hue_values = list(df_subset[hue_col].drop_duplicates())
            n_hues = len(hue_values)
            bar_width = 0.8 / max(n_hues, 1)
            for i, hue_val in enumerate(hue_values):
                ordered = (
                    df_subset[df_subset[hue_col] == hue_val]
                    .set_index(x_col)
                    .reindex(x_values)
                )
                offset = (i - (n_hues - 1) / 2.0) * bar_width
                ax.bar(
                    x_positions + offset,
                    ordered["value_mean"].values,
                    bar_width,
                    yerr=safe_yerr(ordered["value_std"].values),
                    capsize=3,
                    label=str(hue_val),
                )
            ax.legend(title=hue_col)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(v) for v in x_values], rotation=45, ha="right")
        ax.set_xlabel(x_col)

    return (grouped_bars_with_errors,)


@app.cell
def _(grouped_bars_with_errors, plt):
    def plot_embedding_stats(df_stats_tidy):
        """One figure per space, three subplots per figure.

        Subplots are L2 norm, mean per dimension, and variance per dimension.
        Each subplot groups bars by `state` (reference vs manipulated) with
        one group per manipulation. Error bars come from each metric's std.
        """
        x_col = (
            "intervention_k"
            if "intervention_k" in df_stats_tidy.columns
            else "manipulation_id"
        )
        metric_specs = [
            ("l2_norm", "Row L2 norm"),
            ("mean_per_dim", "Mean per dimension"),
            ("var_per_dim", "Variance per dimension"),
        ]
        figs = []
        for space in ("raw", "embedding"):
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
            fig.suptitle(f"Embedding stats — {space} space", y=1.02)
            for ax, (metric, title) in zip(axes, metric_specs):
                df_sub = df_stats_tidy[
                    (df_stats_tidy["space"] == space)
                    & (df_stats_tidy["metric"] == metric)
                ]
                grouped_bars_with_errors(ax, df_sub, x_col=x_col, hue_col="state")
                ax.set_title(title)
                ax.set_ylabel("value")
            plt.tight_layout()
            plt.show()
            figs.append(fig)

    return (plot_embedding_stats,)


@app.cell
def _(grouped_bars_with_errors, plt):
    def plot_embedding_shift(df_shift_tidy):
        """One figure per space, four subplots per figure.

        Subplots are within-reference pairwise spread, within-manipulated
        pairwise spread, paired shift magnitude (||manip - ref||), and
        pairwise cosine similarity between per-cell shift vectors. One bar
        per manipulation. Error bars come from each metric's std.
        """
        x_col = (
            "intervention_k"
            if "intervention_k" in df_shift_tidy.columns
            else "manipulation_id"
        )
        metric_specs = [
            ("within_ref_spread", "Within-reference pairwise L2"),
            ("within_man_spread", "Within-manipulated pairwise L2"),
            ("shift_magnitude", "Shift magnitude ||manip − ref||"),
            ("shift_pairwise_cosine", "Pairwise shift cosine"),
        ]
        figs = []
        for space in ("raw", "embedding"):
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Embedding shift — {space} space", y=1.02)
            for ax, (metric, title) in zip(axes.ravel(), metric_specs):
                df_sub = df_shift_tidy[
                    (df_shift_tidy["space"] == space)
                    & (df_shift_tidy["metric"] == metric)
                ]
                grouped_bars_with_errors(ax, df_sub, x_col=x_col, hue_col=None)
                ax.set_title(title)
                ax.set_ylabel("value")
            plt.tight_layout()
            plt.show()
            figs.append(fig)

    return (plot_embedding_shift,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load data
    """)
    return


@app.cell
def _(
    cfg_atlas,
    cfg_base_dir,
    cfg_intervention_family,
    cfg_model,
    load_family_data,
):
    data_manipulations, emb_ref_mat, raw_ref_mat, raw_ref_obs, load_summary = (
        load_family_data(
            base_dir=cfg_base_dir,
            atlas=cfg_atlas,
            model=cfg_model,
            intervention_family=cfg_intervention_family,
        )
    )
    print(load_summary)
    return data_manipulations, emb_ref_mat, raw_ref_mat, raw_ref_obs


@app.cell
def _(data_manipulations, manipulation_inventory_df):
    df_manipulation_inventory = manipulation_inventory_df(data_manipulations)
    df_manipulation_inventory
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Precompute reference cache
    """)
    return


@app.cell
def _(
    build_reference_model_context,
    cfg_seed,
    cfg_stats_shift_pairwise_cell_subsample_n,
    cfg_stats_shift_pairwise_max_pairs,
    emb_ref_mat,
    raw_ref_mat,
    raw_ref_obs,
):
    notebook_model_ctx, ref_cache_summary = build_reference_model_context(
        emb_ref_mat=emb_ref_mat,
        raw_ref_mat=raw_ref_mat,
        raw_ref_obs=raw_ref_obs,
        seed=cfg_seed,
        pairwise_cell_subsample_n=cfg_stats_shift_pairwise_cell_subsample_n,
        pairwise_max_pairs=cfg_stats_shift_pairwise_max_pairs,
    )
    print(ref_cache_summary)
    return (notebook_model_ctx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Embedding stats

    Compute the per-manipulation stats, reshape to wide form, plot.
    """)
    return


@app.cell
def _(
    cfg_atlas,
    cfg_intervention_family,
    cfg_model,
    cfg_seed,
    data_manipulations,
    emb_ref_mat,
    notebook_model_ctx,
    raw_ref_mat,
    raw_ref_obs,
    run_embedding_stats_for_family,
):
    df_embedding_stats = run_embedding_stats_for_family(
        manipulations=data_manipulations,
        raw_ref_mat=raw_ref_mat,
        emb_ref_mat=emb_ref_mat,
        raw_ref_obs=raw_ref_obs,
        ref_stats_cache=notebook_model_ctx.ref_stats_cache,
        dataset_id=cfg_atlas,
        model=cfg_model,
        intervention_name=cfg_intervention_family,
        seed=cfg_seed,
    )
    print(
        "embedding_stats metric_name:",
        sorted(df_embedding_stats["metric_name"].astype(str).unique()),
    )
    return (df_embedding_stats,)


@app.cell
def _(
    assert_embedding_stats_schema,
    data_manipulations,
    df_embedding_stats,
    tidy_embedding_stats,
):
    assert_embedding_stats_schema(df_embedding_stats)
    df_stats_tidy = tidy_embedding_stats(df_embedding_stats, data_manipulations)
    mean_nan = df_stats_tidy.loc[
        df_stats_tidy["metric"] == "mean_per_dim", "value_mean"
    ].isna().all()
    if mean_nan:
        sample = df_embedding_stats[
            df_embedding_stats["metric_name"].str.contains("col_mean", na=False)
        ][["intervention_id", "manipulation_id", "space", "metric_name", "value_mean"]]
        raise RuntimeError(
            "mean_per_dim is all NaN after tidy conversion. "
            f"Sample col_mean rows in df_embedding_stats:\n{sample.head(12)}"
        )
    df_stats_tidy
    return (df_stats_tidy,)


@app.cell
def _(df_stats_tidy, plot_embedding_stats):
    plot_embedding_stats(df_stats_tidy)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Embedding shift

    Compute the per-manipulation shifts, reshape to wide form, plot.
    """)
    return


@app.cell
def _(
    cfg_atlas,
    cfg_intervention_family,
    cfg_model,
    cfg_seed,
    cfg_stats_shift_pairwise_max_pairs,
    data_manipulations,
    emb_ref_mat,
    notebook_model_ctx,
    raw_ref_mat,
    raw_ref_obs,
    run_embedding_shift_for_family,
):
    df_embedding_shift = run_embedding_shift_for_family(
        manipulations=data_manipulations,
        raw_ref_mat=raw_ref_mat,
        emb_ref_mat=emb_ref_mat,
        raw_ref_obs=raw_ref_obs,
        ref_stats_cache=notebook_model_ctx.ref_stats_cache,
        dataset_id=cfg_atlas,
        model=cfg_model,
        intervention_name=cfg_intervention_family,
        seed=cfg_seed,
        pairwise_max_pairs=cfg_stats_shift_pairwise_max_pairs,
    )
    return (df_embedding_shift,)


@app.cell
def _(data_manipulations, df_embedding_shift, tidy_embedding_shift):
    df_shift_tidy = tidy_embedding_shift(df_embedding_shift, data_manipulations)
    df_shift_tidy
    return (df_shift_tidy,)


@app.cell
def _(df_shift_tidy, plot_embedding_shift):
    plot_embedding_shift(df_shift_tidy)
    return


if __name__ == "__main__":
    app.run()
