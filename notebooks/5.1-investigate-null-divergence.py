import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PCA and Geneformer show a lower null KL and JS divergence value - why?
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
    from matplotlib.lines import Line2D

    from scfm_controlled_manipulations.evaluation.data import dense_embedding_aligned_to_obs
    from scfm_controlled_manipulations.evaluation.metrics_knn import (
        build_weighted_knn_adjacency_from_knn,
        knn_neighbors,
        sym_kl_js_per_cell,
        transition_powers,
    )
    from scfm_controlled_manipulations.io import embedding_path

    EVALUATION_RESULTS_DIR = Path(
        "/vault/amoneim/scfm-controlled-manipulations/processed/sceval/dendritic_cells/results/evaluation"
    )
    EMBEDDINGS_DIR = Path(
        "/vault/amoneim/scfm-controlled-manipulations/processed/sceval/dendritic_cells/embeddings"
    )
    MANIPULATIONS_DIR = Path(
        "/vault/amoneim/scfm-controlled-manipulations/processed/sceval/dendritic_cells/results/manipulations"
    )

    MODEL_ORDER = [
        "pca",
        "scgpt",
        "geneformer",
        "scfoundation",
        "scimilarity",
        "scconcept",
    ]

    MODEL_COLORS = {
        "pca": "#0072B2",
        "scgpt": "#E69F00",
        "geneformer": "#009E73",
        "scfoundation": "#CC79A7",
        "scimilarity": "#D55E00",
        "scconcept": "#56B4E9",
    }

    INTERVENTION_NAME = "downsample"
    PARAM_KEY = "fraction"
    PARAM_VALUE = 0.2
    DEFAULT_K = 15
    SPACE = "embedding"
    DISTANCE_METRIC = "euclidean"
    KNN_ALPHA = 10.0
    EVAL_SEED = 42
    PLOT_DIFFUSION_T = 1
    DIFFUSION_T_VALUES = [1, 2, 4, 8, 16]
    REFERENCE_INTERVENTION_ID = "reference"
    return (
        DEFAULT_K,
        DIFFUSION_T_VALUES,
        DISTANCE_METRIC,
        EMBEDDINGS_DIR,
        EVALUATION_RESULTS_DIR,
        EVAL_SEED,
        INTERVENTION_NAME,
        KNN_ALPHA,
        Line2D,
        MANIPULATIONS_DIR,
        MODEL_COLORS,
        MODEL_ORDER,
        PARAM_KEY,
        PARAM_VALUE,
        PLOT_DIFFUSION_T,
        REFERENCE_INTERVENTION_ID,
        SPACE,
        ad,
        build_weighted_knn_adjacency_from_knn,
        dense_embedding_aligned_to_obs,
        embedding_path,
        knn_neighbors,
        mo,
        np,
        pd,
        plt,
        sym_kl_js_per_cell,
        transition_powers,
    )


@app.cell
def _(
    DEFAULT_K,
    DISTANCE_METRIC,
    EVAL_SEED,
    KNN_ALPHA,
    Line2D,
    REFERENCE_INTERVENTION_ID,
    SPACE,
    ad,
    build_weighted_knn_adjacency_from_knn,
    dense_embedding_aligned_to_obs,
    knn_neighbors,
    np,
    pd,
    plt,
    sym_kl_js_per_cell,
    transition_powers,
):
    import hashlib

    def diffusion_null_seed(
        base_seed: int, space: str, metric: str, k: int, t: int
    ) -> int:
        """Match ``metrics_knn._diffusion_null_seed`` (private; inlined for marimo)."""
        digest = hashlib.sha256(
            f"{base_seed}|diffusion|{space}|{metric}|{k}|{t}".encode()
        ).digest()
        return int.from_bytes(digest[:4], "big")

    def load_metrics_df(evaluation_dir, model_order):
        frames = []
        for path in sorted(evaluation_dir.glob("*_metrics.csv")):
            frames.append(pd.read_csv(path))
        metrics_df = pd.concat(frames, ignore_index=True)
        metrics_df["model"] = pd.Categorical(
            metrics_df["model"], categories=model_order, ordered=True
        )
        return metrics_df

    def find_intervention_id(manipulations_dir, intervention_name, param_key, param_value):
        for path in sorted(manipulations_dir.glob(f"{intervention_name}_*.h5ad")):
            adata = ad.read_h5ad(path, backed="r")
            try:
                params = adata.uns.get("scfm_intervention", {}).get(intervention_name, {})
                if params.get(param_key) == param_value:
                    return path.stem
            finally:
                adata.file.close()
        raise ValueError(
            f"No {intervention_name} manipulation with {param_key}={param_value} found"
        )

    def load_ref_and_man_embeddings(
        embeddings_dir,
        intervention_id,
        models,
        embedding_path_fn,
        *,
        reference_intervention_id=REFERENCE_INTERVENTION_ID,
    ):
        out = {}
        for model in models:
            ref_path = embedding_path_fn(embeddings_dir, model, reference_intervention_id)
            man_path = embedding_path_fn(embeddings_dir, model, intervention_id)
            if not ref_path.is_file():
                raise FileNotFoundError(f"Missing reference embedding file: {ref_path}")
            if not man_path.is_file():
                raise FileNotFoundError(f"Missing manipulated embedding file: {man_path}")
            ref_adata = ad.read_h5ad(ref_path)
            man_adata = ad.read_h5ad(man_path)
            target_obs = ref_adata.obs_names
            out[model] = (
                dense_embedding_aligned_to_obs(ref_adata, target_obs, label="emb_ref"),
                dense_embedding_aligned_to_obs(man_adata, target_obs, label="emb_man"),
            )
        return out

    def compute_per_cell_kl_distributions(
        emb_ref,
        emb_man,
        *,
        k,
        diffusion_t,
        metric=DISTANCE_METRIC,
        alpha=KNN_ALPHA,
        seed=EVAL_SEED,
        space=SPACE,
    ):
        ref_dist, ref_idx = knn_neighbors(emb_ref, k, metric)
        man_dist, man_idx = knn_neighbors(emb_man, k, metric)
        ref_adj = build_weighted_knn_adjacency_from_knn(
            ref_dist, ref_idx, alpha=alpha
        )
        man_adj = build_weighted_knn_adjacency_from_knn(
            man_dist, man_idx, alpha=alpha
        )
        p_t = transition_powers(ref_adj, [diffusion_t])[diffusion_t]
        q_t = transition_powers(man_adj, [diffusion_t])[diffusion_t]
        real_kl, _ = sym_kl_js_per_cell(p_t, q_t)
        null_rng = np.random.default_rng(
            diffusion_null_seed(seed, space, metric, k, diffusion_t)
        )
        perm = null_rng.permutation(emb_ref.shape[0])
        null_kl, _ = sym_kl_js_per_cell(p_t, q_t[perm])
        return real_kl, null_kl

    def compute_per_cell_kl_distributions_multi_t(
        emb_ref,
        emb_man,
        *,
        k,
        diffusion_t_values,
        metric=DISTANCE_METRIC,
        alpha=KNN_ALPHA,
        seed=EVAL_SEED,
        space=SPACE,
    ):
        t_list = [int(t) for t in diffusion_t_values]
        ref_dist, ref_idx = knn_neighbors(emb_ref, k, metric)
        man_dist, man_idx = knn_neighbors(emb_man, k, metric)
        ref_adj = build_weighted_knn_adjacency_from_knn(
            ref_dist, ref_idx, alpha=alpha
        )
        man_adj = build_weighted_knn_adjacency_from_knn(
            man_dist, man_idx, alpha=alpha
        )
        ref_powers = transition_powers(ref_adj, t_list)
        man_powers = transition_powers(man_adj, t_list)

        real_by_t = {}
        null_by_t = {}
        for diffusion_t in t_list:
            p_t = ref_powers[diffusion_t]
            q_t = man_powers[diffusion_t]
            real_kl, _ = sym_kl_js_per_cell(p_t, q_t)
            null_rng = np.random.default_rng(
                diffusion_null_seed(seed, space, metric, k, diffusion_t)
            )
            perm = null_rng.permutation(emb_ref.shape[0])
            null_kl, _ = sym_kl_js_per_cell(p_t, q_t[perm])
            real_by_t[diffusion_t] = real_kl
            null_by_t[diffusion_t] = null_kl
        return real_by_t, null_by_t

    def compute_all_model_kl_distributions(
        embedding_pairs,
        *,
        model_order,
        k,
        diffusion_t,
    ):
        out = {}
        for model in model_order:
            emb_ref, emb_man = embedding_pairs[model]
            real_kl, null_kl = compute_per_cell_kl_distributions(
                emb_ref, emb_man, k=k, diffusion_t=diffusion_t
            )
            out[model] = {"real": real_kl, "null": null_kl}
        return out

    def build_kl_violin_dataframe(
        embedding_pairs,
        *,
        model_order,
        k,
        diffusion_t_values,
    ):
        rows = []
        for model in model_order:
            emb_ref, emb_man = embedding_pairs[model]
            real_by_t, null_by_t = compute_per_cell_kl_distributions_multi_t(
                emb_ref,
                emb_man,
                k=k,
                diffusion_t_values=diffusion_t_values,
            )
            for diffusion_t in diffusion_t_values:
                for value in real_by_t[int(diffusion_t)]:
                    rows.append(
                        {
                            "model": model,
                            "diffusion_t": int(diffusion_t),
                            "kind": "Observed",
                            "kl": float(value),
                        }
                    )
                for value in null_by_t[int(diffusion_t)]:
                    rows.append(
                        {
                            "model": model,
                            "diffusion_t": int(diffusion_t),
                            "kind": "Null",
                            "kl": float(value),
                        }
                    )
        out = pd.DataFrame(rows)
        out["model"] = pd.Categorical(
            out["model"], categories=model_order, ordered=True
        )
        return out

    def plot_kl_violin_across_t(
        violin_df,
        *,
        model_order,
        model_colors,
        k,
    ):
        t_values = sorted(violin_df["diffusion_t"].unique())
        n_models = len(model_order)
        fig, axes = plt.subplots(
            n_models, 2, figsize=(12, 2.6 * n_models), squeeze=False, sharey="row"
        )
        col_titles = ("Observed (paired)", "Permuted null")

        for row_idx, model in enumerate(model_order):
            model_df = violin_df[violin_df["model"].astype(str) == model]
            color = model_colors[model]
            for col_idx, kind in enumerate(("Observed", "Null")):
                ax = axes[row_idx, col_idx]
                kind_df = model_df[model_df["kind"] == kind]
                datasets = [
                    kind_df.loc[kind_df["diffusion_t"] == t, "kl"].to_numpy()
                    for t in t_values
                ]
                positions = np.arange(len(t_values))
                parts = ax.violinplot(
                    datasets,
                    positions=positions,
                    widths=0.8,
                    showmeans=True,
                    showmedians=True,
                )
                for body in parts["bodies"]:
                    body.set_facecolor(color)
                    body.set_edgecolor(color)
                    body.set_alpha(0.65)
                for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
                    if key in parts:
                        parts[key].set_color("#222222")
                        parts[key].set_linewidth(1.0)
                ax.set_xticks(positions)
                ax.set_xticklabels([str(int(t)) for t in t_values])
                if row_idx == 0:
                    ax.set_title(col_titles[col_idx])
                if col_idx == 0:
                    ax.set_ylabel(model)
                if row_idx == n_models - 1:
                    ax.set_xlabel("Diffusion time t")

        fig.suptitle(
            f"Per-cell symmetric KL across diffusion times — downsample fraction=0.2, k={k}",
            y=1.01,
            fontsize=12,
        )
        fig.tight_layout()
        return fig

    def plot_kl_distribution_grid(
        embedding_pairs,
        *,
        model_order,
        model_colors,
        diffusion_t,
        k,
        bins=40,
    ):
        kl_distributions = compute_all_model_kl_distributions(
            embedding_pairs,
            model_order=model_order,
            k=k,
            diffusion_t=diffusion_t,
        )
        n_models = len(model_order)
        fig, axes = plt.subplots(
            n_models, 2, figsize=(10, 2.4 * n_models), squeeze=False, sharex="col"
        )
        col_titles = ("Observed (paired)", "Permuted null")

        for row_idx, model in enumerate(model_order):
            dists = kl_distributions[model]
            color = model_colors[model]
            for col_idx, key in enumerate(("real", "null")):
                ax = axes[row_idx, col_idx]
                values = dists[key]
                ax.hist(
                    values,
                    bins=bins,
                    density=True,
                    alpha=0.8,
                    color=color,
                    edgecolor="white",
                    linewidth=0.4,
                )
                ax.axvline(
                    float(np.mean(values)),
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                    label=f"mean={np.mean(values):.3f}",
                )
                if row_idx == 0:
                    ax.set_title(col_titles[col_idx])
                if col_idx == 0:
                    ax.set_ylabel(model)
                if row_idx == n_models - 1:
                    ax.set_xlabel("Symmetric KL divergence")
                ax.legend(loc="upper right", fontsize=8, frameon=False)

        fig.suptitle(
            f"Per-cell symmetric KL — downsample fraction=0.2, k={k}, t={diffusion_t}",
            y=1.01,
            fontsize=12,
        )
        fig.tight_layout()
        return fig

    def compute_model_pca_umap_facets(
        emb_ref,
        emb_man,
        *,
        seed=EVAL_SEED,
        umap_n_neighbors=15,
    ):
        from sklearn.decomposition import PCA
        import umap

        pca_ref = PCA(n_components=2, random_state=seed).fit_transform(emb_ref)
        pca_man = PCA(n_components=2, random_state=seed).fit_transform(emb_man)
        umap_ref = umap.UMAP(
            n_components=2,
            random_state=seed,
            n_neighbors=umap_n_neighbors,
            min_dist=0.1,
        ).fit_transform(emb_ref)
        umap_man = umap.UMAP(
            n_components=2,
            random_state=seed,
            n_neighbors=umap_n_neighbors,
            min_dist=0.1,
        ).fit_transform(emb_man)
        return pca_ref, pca_man, umap_ref, umap_man

    def plot_embedding_pca_umap_grid(
        embedding_pairs,
        *,
        model_order,
        seed=EVAL_SEED,
    ):
        n_models = len(model_order)
        fig, axes = plt.subplots(
            n_models, 4, figsize=(14, 3.0 * n_models), squeeze=False
        )
        panel_specs = (
            ("pca_ref", "PCA (reference)", "#404040"),
            ("pca_man", "PCA (manipulated)", "#D55E00"),
            ("umap_ref", "UMAP (reference)", "#404040"),
            ("umap_man", "UMAP (manipulated)", "#D55E00"),
        )

        for row_idx, model in enumerate(model_order):
            emb_ref, emb_man = embedding_pairs[model]
            coords_by_key = dict(
                zip(
                    ("pca_ref", "pca_man", "umap_ref", "umap_man"),
                    compute_model_pca_umap_facets(emb_ref, emb_man, seed=seed),
                )
            )
            for col_idx, (key, title, color) in enumerate(panel_specs):
                ax = axes[row_idx, col_idx]
                coords = coords_by_key[key]
                ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    s=10,
                    alpha=0.6,
                    c=color,
                    linewidths=0,
                )
                if row_idx == 0:
                    ax.set_title(title, fontsize=10)
                if col_idx == 0:
                    ax.set_ylabel(model)
                ax.set_xticks([])
                ax.set_yticks([])

        fig.suptitle(
            "Embedding geometry per model — PCA and UMAP fit separately on reference and manipulated",
            y=1.02,
            fontsize=12,
        )
        fig.tight_layout()
        return fig

    def filter_divergence_metrics(
        metrics_df,
        *,
        intervention_id,
        space,
        k,
        metric_names,
    ):
        sub = metrics_df[
            (metrics_df["intervention_id"] == intervention_id)
            & (metrics_df["metric_category"] == "knn_metrics")
            & (metrics_df["metric_name"].isin(metric_names))
            & (metrics_df["space"] == space)
            & (metrics_df["k"] == k)
        ].copy()
        sub = sub.dropna(subset=["value_mean", "diffusion_t"])
        sub = sub.sort_values(["metric_name", "model", "diffusion_t"])
        return sub

    STAT_COLUMNS = [
        "value_mean",
        "value_median",
        "value_std",
        "value_min",
        "value_max",
        "value_q05",
        "value_q25",
        "value_q75",
        "value_q95",
    ]

    STAT_LABELS = {
        "value_mean": "mean",
        "value_median": "median",
        "value_std": "std",
        "value_min": "min",
        "value_max": "max",
        "value_q05": "q05",
        "value_q25": "q25",
        "value_q75": "q75",
        "value_q95": "q95",
        "null_value": "null_mean",
    }

    def build_divergence_stats_table(divergence_df, metric_name, model_order):
        sub = divergence_df[divergence_df["metric_name"] == metric_name].copy()
        sub["model"] = pd.Categorical(sub["model"], categories=model_order, ordered=True)
        sub = sub.sort_values(["model", "diffusion_t"])
        table = sub[["model", "diffusion_t", *STAT_COLUMNS, "null_value"]].rename(
            columns=STAT_LABELS
        )
        numeric_cols = [STAT_LABELS[c] for c in STAT_COLUMNS] + [STAT_LABELS["null_value"]]
        table[numeric_cols] = table[numeric_cols].round(4)
        return table

    def plot_kl_js_divergence(
        divergence_df,
        *,
        model_order,
        model_colors,
        title_suffix,
    ):
        metric_panels = [
            ("diffusion_sym_kl", "Symmetric KL divergence"),
            ("diffusion_js", "Jensen–Shannon divergence"),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

        for ax, (metric_name, ylabel) in zip(axes, metric_panels):
            panel = divergence_df[divergence_df["metric_name"] == metric_name].copy()
            for model in model_order:
                mdf = panel[panel["model"].astype(str) == model].sort_values("diffusion_t")
                if mdf.empty:
                    continue
                color = model_colors[model]
                ax.plot(
                    mdf["diffusion_t"],
                    mdf["value_mean"],
                    color=color,
                    marker="o",
                    linewidth=2,
                    label=model,
                )
                if mdf["null_value"].notna().any():
                    ax.plot(
                        mdf["diffusion_t"],
                        mdf["null_value"],
                        color=color,
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.55,
                    )
            ax.set_xlabel("Diffusion time t")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)

        model_handles = [
            Line2D([0], [0], color=model_colors[m], marker="o", linewidth=2, label=m)
            for m in model_order
            if m in divergence_df["model"].astype(str).unique()
        ]
        null_handle = Line2D(
            [0], [0], color="gray", linestyle="--", linewidth=1.2, label="null"
        )
        fig.legend(
            handles=[*model_handles, null_handle],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=min(len(model_handles) + 1, 7),
            frameon=False,
        )
        fig.suptitle(
            f"Downsample ({title_suffix}) — embedding space, k={DEFAULT_K}",
            y=1.14,
            fontsize=12,
        )
        fig.tight_layout()
        return fig

    return (
        build_divergence_stats_table,
        build_kl_violin_dataframe,
        filter_divergence_metrics,
        find_intervention_id,
        load_metrics_df,
        load_ref_and_man_embeddings,
        plot_embedding_pca_umap_grid,
        plot_kl_distribution_grid,
        plot_kl_js_divergence,
        plot_kl_violin_across_t,
    )


@app.cell
def _(
    EVALUATION_RESULTS_DIR,
    INTERVENTION_NAME,
    MANIPULATIONS_DIR,
    MODEL_ORDER,
    PARAM_KEY,
    PARAM_VALUE,
    find_intervention_id,
    load_metrics_df,
):
    metrics_df = load_metrics_df(EVALUATION_RESULTS_DIR, MODEL_ORDER)
    intervention_id = find_intervention_id(
        MANIPULATIONS_DIR, INTERVENTION_NAME, PARAM_KEY, PARAM_VALUE
    )
    return intervention_id, metrics_df


@app.cell
def _(
    DEFAULT_K,
    EMBEDDINGS_DIR,
    MODEL_ORDER,
    SPACE,
    embedding_path,
    filter_divergence_metrics,
    intervention_id,
    load_ref_and_man_embeddings,
    metrics_df,
):
    divergence_df = filter_divergence_metrics(
        metrics_df,
        intervention_id=intervention_id,
        space=SPACE,
        k=DEFAULT_K,
        metric_names=["diffusion_sym_kl", "diffusion_js"],
    )
    embedding_pairs = load_ref_and_man_embeddings(
        EMBEDDINGS_DIR,
        intervention_id,
        MODEL_ORDER,
        embedding_path,
    )
    return divergence_df, embedding_pairs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Recreate plots

    Two panels side by side (symmetric KL and JS divergence) for **downsample** at **fraction = 0.2**.
    Solid lines are per-model divergence means; dashed lines are the corresponding permutation null means.
    """)
    return


@app.cell
def _(
    MODEL_COLORS,
    MODEL_ORDER,
    PARAM_KEY,
    PARAM_VALUE,
    divergence_df,
    plot_kl_js_divergence,
    plt,
):
    title_suffix = f"{PARAM_KEY}={PARAM_VALUE}"
    plot_kl_js_divergence(
        divergence_df,
        model_order=MODEL_ORDER,
        model_colors=MODEL_COLORS,
        title_suffix=title_suffix,
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Distribution stats (observed vs null)

    Per-model, per-diffusion-time summary of the **per-cell** divergence distribution (observed columns)
    alongside the **permutation null mean** (`null_mean`). One row per model and `diffusion_t`.
    """)
    return


@app.cell
def _(MODEL_ORDER, build_divergence_stats_table, divergence_df, mo):
    kl_stats_table = build_divergence_stats_table(
        divergence_df, "diffusion_sym_kl", MODEL_ORDER
    )
    mo.vstack(
        [
            mo.md("**Symmetric KL divergence**"),
            mo.ui.table(kl_stats_table),
        ]
    )
    return


@app.cell
def _(MODEL_ORDER, build_divergence_stats_table, divergence_df, mo):
    js_stats_table = build_divergence_stats_table(
        divergence_df, "diffusion_js", MODEL_ORDER
    )
    mo.vstack(
        [
            mo.md("**Jensen–Shannon divergence**"),
            mo.ui.table(js_stats_table),
        ]
    )
    return


@app.cell(hide_code=True)
def _(DEFAULT_K, PLOT_DIFFUSION_T, mo):
    mo.md(
        f"""
    ### Per-cell KL distributions (recalculated)

    Recomputes symmetric KL per cell using the evaluation pipeline
    (`knn_neighbors` → PHATE-style adjacency → diffusion powers → `sym_kl_js_per_cell`).
    Left column: **paired** ref vs manipulated transitions. Right column: same with
    manipulated rows **permuted** (one shuffle, matching evaluation `null_value`).

    Shown for **k = {DEFAULT_K}**, **t = {PLOT_DIFFUSION_T}**. Change `PLOT_DIFFUSION_T` in the imports cell to explore other diffusion times.
    """
    )
    return


@app.cell
def _(
    DEFAULT_K,
    MODEL_COLORS,
    MODEL_ORDER,
    embedding_pairs,
    plot_kl_distribution_grid,
    plt,
):
    plot_kl_distribution_grid(
        embedding_pairs,
        model_order=MODEL_ORDER,
        model_colors=MODEL_COLORS,
        diffusion_t=1,
        k=DEFAULT_K,
    )
    plt.gcf()
    return


@app.cell
def _(
    DEFAULT_K,
    MODEL_COLORS,
    MODEL_ORDER,
    embedding_pairs,
    plot_kl_distribution_grid,
    plt,
):
    plot_kl_distribution_grid(
        embedding_pairs,
        model_order=MODEL_ORDER,
        model_colors=MODEL_COLORS,
        diffusion_t=2,
        k=DEFAULT_K,
    )
    plt.gcf()
    return


@app.cell
def _(
    DEFAULT_K,
    MODEL_COLORS,
    MODEL_ORDER,
    embedding_pairs,
    plot_kl_distribution_grid,
    plt,
):
    plot_kl_distribution_grid(
        embedding_pairs,
        model_order=MODEL_ORDER,
        model_colors=MODEL_COLORS,
        diffusion_t=4,
        k=DEFAULT_K,
    )
    plt.gcf()
    return


@app.cell
def _(
    DEFAULT_K,
    MODEL_COLORS,
    MODEL_ORDER,
    embedding_pairs,
    plot_kl_distribution_grid,
    plt,
):
    plot_kl_distribution_grid(
        embedding_pairs,
        model_order=MODEL_ORDER,
        model_colors=MODEL_COLORS,
        diffusion_t=8,
        k=DEFAULT_K,
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(DEFAULT_K, mo):
    mo.md(
        f"""
    ### KL violin plots across diffusion times

    Summarizes the per-cell symmetric KL distributions at every diffusion time step.
    **n_models × 2** layout: observed (left) and permuted null (right); x-axis = `t`.
    Black markers on each violin = mean and median. Recomputed with **k = {DEFAULT_K}**.
    """
    )
    return


@app.cell
def _(
    DEFAULT_K,
    DIFFUSION_T_VALUES,
    MODEL_ORDER,
    build_kl_violin_dataframe,
    embedding_pairs,
):
    kl_violin_df = build_kl_violin_dataframe(
        embedding_pairs,
        model_order=MODEL_ORDER,
        k=DEFAULT_K,
        diffusion_t_values=DIFFUSION_T_VALUES,
    )
    return (kl_violin_df,)


@app.cell
def _(
    DEFAULT_K,
    MODEL_COLORS,
    MODEL_ORDER,
    kl_violin_df,
    plot_kl_violin_across_t,
    plt,
):
    plot_kl_violin_across_t(
        kl_violin_df,
        model_order=MODEL_ORDER,
        model_colors=MODEL_COLORS,
        k=DEFAULT_K,
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Embedding geometry (PCA and UMAP per model)

    Four columns per model row: **PCA (reference)**, **PCA (manipulated)**,
    **UMAP (reference)**, **UMAP (manipulated)**. PCA and UMAP are each fit
    separately on the reference or manipulated embedding matrix for that model.
    """)
    return


@app.cell
def _(MODEL_ORDER, embedding_pairs, plot_embedding_pca_umap_grid, plt):
    plot_embedding_pca_umap_grid(
        embedding_pairs,
        model_order=MODEL_ORDER,
    )
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
