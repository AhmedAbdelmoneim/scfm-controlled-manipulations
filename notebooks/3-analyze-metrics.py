import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Analyze SCEval metrics — dendritic_cells

    Loads per-model evaluation CSVs and plots **all models together** (`hue=model`).
    Each row summarizes a per-cell distribution via `value_mean` (and related columns from evaluation).
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
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.figsize"] = (10, 5)

    DATASET_ID = "dendritic_cells"
    RESULTS_DIR = Path(
        "/vault/amoneim/scfm-controlled-manipulations/processed/sceval/dendritic_cells/results"
    )
    EVAL_DIR = RESULTS_DIR / "evaluation"
    MANIP_DIR = RESULTS_DIR / "manipulations"

    MODEL_ORDER = [
        "pca",
        "scgpt",
        "geneformer",
        "scfoundation",
        "scimilarity",
        "scconcept",
    ]

    PARAM_KEYS = {
        "downsample": "fraction",
        "gene_dropout": "dropout_rate",
        "local_smoothing": "k",
        "poisson_resampling": "iterations",
        "gene_shuffle": "variant",
    }
    return (
        EVAL_DIR,
        MANIP_DIR,
        MODEL_ORDER,
        PARAM_KEYS,
        ad,
        mo,
        np,
        pd,
        plt,
        sns,
    )


@app.cell
def _(EVAL_DIR, MANIP_DIR, MODEL_ORDER, PARAM_KEYS, ad, np, pd):
    metric_frames = []
    for _path in sorted(EVAL_DIR.glob("*_metrics.csv")):
        _df = pd.read_csv(_path)
        metric_frames.append(_df)
    metrics_df = pd.concat(metric_frames, ignore_index=True)
    metrics_df["model"] = pd.Categorical(
        metrics_df["model"], categories=MODEL_ORDER, ordered=True
    )

    intervention_ids = metrics_df["intervention_id"].drop_duplicates().tolist()
    param_rows = []
    for _iid in intervention_ids:
        _h5ad = MANIP_DIR / f"{_iid}.h5ad"
        if not _h5ad.is_file():
            continue
        _adata = ad.read_h5ad(_h5ad, backed="r")
        _name = metrics_df.loc[metrics_df["intervention_id"] == _iid, "intervention_name"].iloc[0]
        _params = _adata.uns.get("scfm_intervention", {}).get(_name, {})
        _key = PARAM_KEYS.get(_name)
        _value = _params.get(_key) if _key else None
        if _value is None and _params:
            _scalar = {
                _k: _v
                for _k, _v in _params.items()
                if not isinstance(_v, (list, np.ndarray))
                or (hasattr(_v, "__len__") and len(_v) < 10)
            }
            if len(_scalar) == 1:
                _key, _value = next(iter(_scalar.items()))
            else:
                _key, _value = "intervention_id", _iid
        param_rows.append(
            {
                "intervention_id": _iid,
                "intervention_name": _name,
                "param_key": _key,
                "param_value": _value,
            }
        )
        _adata.file.close()

    params_df = pd.DataFrame(param_rows)
    metrics_df = metrics_df.merge(params_df, on=["intervention_id", "intervention_name"], how="left")

    overview = (
        metrics_df.groupby(["model", "intervention_name", "metric_category"], observed=True)
        .size()
        .reset_index(name="n_rows")
    )
    return metrics_df, overview


@app.cell
def _(mo):
    mo.md(r"""
    ### Overview
    """)
    return


@app.cell
def _(metrics_df, mo, overview):
    mo.vstack(
        [
            mo.md(
                f"**{metrics_df['model'].nunique()}** models, "
                f"**{metrics_df['intervention_id'].nunique()}** interventions, "
                f"**{len(metrics_df):,}** metric rows."
            ),
            overview,
        ]
    )
    return


@app.cell
def _(MODEL_ORDER, np, pd, plt, sns):
    def _sort_plot_df(df: pd.DataFrame, x: str = "param_value") -> pd.DataFrame:
        out = df.copy()
        out["model"] = pd.Categorical(out["model"], categories=MODEL_ORDER, ordered=True)
        if x not in out.columns:
            return out
        if pd.api.types.is_numeric_dtype(out[x]):
            return out.sort_values(x)
        return out.sort_values(x, key=lambda s: s.astype(str))

    def filter_metrics(
        metrics_df: pd.DataFrame,
        *,
        metric_category: str,
        metric_names: list[str] | None = None,
        spaces: list[str] | None = None,
        dropna: bool = True,
    ) -> pd.DataFrame:
        sub = metrics_df[metrics_df["metric_category"] == metric_category].copy()
        if metric_names is not None:
            sub = sub[sub["metric_name"].isin(metric_names)]
        if spaces is not None:
            sub = sub[sub["space"].isin(spaces)]
        if dropna:
            sub = sub.dropna(subset=["value_mean"])
        return sub

    _EMBEDDING_STATS_PANELS = [
        ("mean_row_l2_norm_ref", "mean_row_l2_norm_man", "L2 norm (per cell)"),
        ("col_mean_ref", "col_mean_man", "Mean per dimension"),
        ("col_variance_ref", "col_variance_man", "Variance per dimension"),
    ]

    def _inv_subset(
        df: pd.DataFrame, intervention_name: str, space: str
    ) -> pd.DataFrame:
        return df[
            (df["intervention_name"] == intervention_name) & (df["space"] == space)
        ].copy()

    def _xlabel_from_inv(inv_df: pd.DataFrame, x_col: str = "param_value") -> str:
        if "param_key" in inv_df.columns and inv_df["param_key"].notna().any():
            return str(inv_df["param_key"].dropna().iloc[0])
        return x_col

    def _sort_param_values(values) -> list:
        vals = list(pd.Series(values).dropna().unique())
        try:
            return sorted(vals, key=lambda v: float(v))
        except (TypeError, ValueError):
            return sorted(vals, key=str)

    def _format_param_value(pval, param_key: str | None) -> str:
        if param_key:
            return f"{param_key}={pval}"
        return str(pval)

    def _add_model_legend(fig, inv_df: pd.DataFrame, *, hue: str = "model") -> None:
        from matplotlib.lines import Line2D

        palette = sns.color_palette("tab10", n_colors=len(MODEL_ORDER))
        present = [m for m in MODEL_ORDER if m in inv_df[hue].astype(str).unique()]
        handles = [
            Line2D(
                [0],
                [0],
                color=palette[MODEL_ORDER.index(m)],
                marker="o",
                linewidth=2,
                label=m,
            )
            for m in present
        ]
        if handles:
            fig.legend(
                handles=handles,
                title="Model",
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=min(len(present), 6),
                frameon=False,
            )

    def plot_metric_grid_intervention(
        df: pd.DataFrame,
        intervention_name: str,
        space: str,
        panel_specs: list[dict],
        nrows: int,
        ncols: int,
        figure_title: str,
        *,
        y: str = "value_mean",
        hue: str = "model",
        x_col: str = "param_value",
        figsize: tuple[float, float] | None = None,
    ):
        """One figure per intervention: each panel is one metric (optional extra filters), all models."""
        inv_df = _inv_subset(df, intervention_name, space)
        if inv_df.empty:
            _fig, _ax = plt.subplots(figsize=(6, 3))
            _ax.text(0.5, 0.5, "No data", ha="center", va="center")
            _ax.set_axis_off()
            return _fig

        if figsize is None:
            figsize = (4.5 * ncols, 3.2 * nrows)
        xlabel = _xlabel_from_inv(inv_df, x_col)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
        axes = np.atleast_2d(axes)
        if nrows == 1 and ncols == 1:
            axes_flat = [axes[0, 0]]
        else:
            axes_flat = axes.flatten()

        for ax, spec in zip(axes_flat, panel_specs):
            panel = inv_df[inv_df["metric_name"] == spec["metric"]].copy()
            for key, val in spec.get("filter", {}).items():
                panel = panel[panel[key] == val]
            panel = _sort_plot_df(panel, x=x_col)
            if panel.empty:
                ax.set_visible(False)
                continue
            sns.lineplot(
                data=panel,
                x=x_col,
                y=y,
                hue=hue,
                ax=ax,
                marker="o",
                linewidth=2,
                palette="tab10",
                legend=False,
            )
            ax.set_title(spec.get("title", spec["metric"]))
            ax.tick_params(axis="x", rotation=25)

        for ax in axes_flat[len(panel_specs) :]:
            ax.set_visible(False)

        for ax in axes_flat[: len(panel_specs)]:
            if ax.get_visible():
                ax.set_xlabel(xlabel)

        _add_model_legend(fig, inv_df, hue=hue)
        param_hint = f" · x = {xlabel}" if inv_df[x_col].notna().any() else ""
        fig.suptitle(f"{figure_title}{param_hint}", y=1.06, fontsize=12)
        fig.tight_layout()
        return fig

    def plot_embedding_stats_intervention(
        df: pd.DataFrame,
        intervention_name: str,
        space: str,
        *,
        y: str = "value_mean",
        hue: str = "model",
    ):
        """3x2: L2 / mean per dim / var per dim x ref vs man."""
        inv_df = _inv_subset(df, intervention_name, space)
        if inv_df.empty:
            _fig, _ax = plt.subplots(figsize=(8, 3))
            _ax.text(0.5, 0.5, "No data", ha="center", va="center")
            _ax.set_axis_off()
            return _fig

        x_col = "param_value"
        xlabel = _xlabel_from_inv(inv_df, x_col)
        fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
        col_titles = ("Reference", "Manipulation")

        for row_idx, (ref_metric, man_metric, row_label) in enumerate(_EMBEDDING_STATS_PANELS):
            for col_idx, metric_name in enumerate((ref_metric, man_metric)):
                ax = axes[row_idx, col_idx]
                panel = _sort_plot_df(
                    inv_df[inv_df["metric_name"] == metric_name], x=x_col
                )
                if panel.empty:
                    ax.set_visible(False)
                    continue
                sns.lineplot(
                    data=panel,
                    x=x_col,
                    y=y,
                    hue=hue,
                    ax=ax,
                    marker="o",
                    linewidth=2,
                    palette="tab10",
                    legend=False,
                )
                ax.set_title(col_titles[col_idx] if row_idx == 0 else "")
                ax.set_ylabel(row_label)
                if row_idx == 2:
                    ax.set_xlabel(xlabel)
                else:
                    ax.tick_params(labelbottom=False)

        _add_model_legend(fig, inv_df, hue=hue)
        param_hint = f" · x = {xlabel}" if inv_df[x_col].notna().any() else ""
        fig.suptitle(
            f"{intervention_name} — embedding stats ({space}){param_hint}",
            y=1.06,
            fontsize=12,
        )
        fig.tight_layout()
        return fig

    _SHIFT_METRICS = [
        ("paired_cell_l2_norm", "Paired cell L2"),
        ("shift_pairwise_cosine", "Shift pairwise cosine"),
        ("within_ref_pairwise_l2", "Within-ref pairwise L2"),
        ("within_man_pairwise_l2", "Within-man pairwise L2"),
    ]

    def plot_embedding_shift_intervention(
        df: pd.DataFrame, intervention_name: str, space: str, *, y: str = "value_mean"
    ):
        specs = [
            {"metric": m, "title": t} for m, t in _SHIFT_METRICS
        ]
        return plot_metric_grid_intervention(
            df,
            intervention_name,
            space,
            specs,
            2,
            2,
            f"{intervention_name} — embedding shift ({space})",
            y=y,
        )

    def plot_clustering_intervention(
        df: pd.DataFrame, intervention_name: str, *, y: str = "value_mean"
    ):
        return plot_metric_grid_intervention(
            df,
            intervention_name,
            "embedding",
            [{"metric": "leiden_ari", "title": "Leiden ARI"}],
            1,
            1,
            f"{intervention_name} — clustering (embedding)",
            y=y,
            x_col="resolution",
            figsize=(7, 4),
        )

    INTERVENTIONS = [
        "downsample",
        "gene_dropout",
        "gene_shuffle",
        "local_smoothing",
        "poisson_resampling",
    ]
    return (
        filter_metrics,
        plot_clustering_intervention,
        plot_embedding_shift_intervention,
        plot_embedding_stats_intervention,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Embedding stats

    **One figure per intervention** (and per `raw` / `embedding` space). Each figure is a 3×2 panel:

    | Row | Left (reference) | Right (manipulation) |
    |-----|------------------|----------------------|
    | 1 | L2 norm per cell | L2 norm per cell |
    | 2 | Mean per dimension | Mean per dimension |
    | 3 | Variance per dimension | Variance per dimension |

    All models appear on every panel (`hue=model`). Reference rows should align across models; manipulation rows show model-specific embedding geometry.
    """)
    return


@app.cell
def _(filter_metrics, metrics_df):
    _stats_names = [
        "mean_row_l2_norm_ref",
        "mean_row_l2_norm_man",
        "col_mean_ref",
        "col_mean_man",
        "col_variance_ref",
        "col_variance_man",
    ]
    embedding_stats_df = filter_metrics(
        metrics_df,
        metric_category="embedding_stats",
        metric_names=_stats_names,
        spaces=["embedding"],
    )
    return (embedding_stats_df,)


@app.cell
def _(embedding_stats_df, plot_embedding_stats_intervention, plt):
    for _space in ["embedding"]:
        plot_embedding_stats_intervention(embedding_stats_df, "downsample", _space)
    plt.gcf()
    return


@app.cell
def _(embedding_stats_df, plot_embedding_stats_intervention, plt):
    for _space in ["embedding"]:
        plot_embedding_stats_intervention(embedding_stats_df, "gene_dropout", _space)
    plt.gcf()
    return


@app.cell
def _(embedding_stats_df, plot_embedding_stats_intervention, plt):
    for _space in ["embedding"]:
        plot_embedding_stats_intervention(embedding_stats_df, "gene_shuffle", _space)
    plt.gcf()
    return


@app.cell
def _(embedding_stats_df, plot_embedding_stats_intervention, plt):
    for _space in ["embedding"]:
        plot_embedding_stats_intervention(embedding_stats_df, "local_smoothing", _space)
    plt.gcf()
    return


@app.cell
def _(embedding_stats_df, plot_embedding_stats_intervention, plt):
    for _space in ["embedding"]:
        plot_embedding_stats_intervention(
            embedding_stats_df, "poisson_resampling", _space
        )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Embedding shift

    **One figure per intervention** (`embedding`). Each figure is a 2×2 panel: paired cell L2, shift pairwise cosine, within-ref pairwise L2, within-man pairwise L2. All models on every panel.
    """)
    return


@app.cell
def _(filter_metrics, metrics_df):
    _shift_names = [
        "paired_cell_l2_norm",
        "shift_pairwise_cosine",
        "within_ref_pairwise_l2",
        "within_man_pairwise_l2",
    ]
    embedding_shift_df = filter_metrics(
        metrics_df,
        metric_category="embedding_shift",
        metric_names=_shift_names,
        spaces=["embedding"],
    )
    return (embedding_shift_df,)


@app.cell
def _(embedding_shift_df, plot_embedding_shift_intervention, plt):
    for _space in ["embedding"]:
        plot_embedding_shift_intervention(embedding_shift_df, "downsample", _space)
    plt.gcf()
    return


@app.cell
def _(embedding_shift_df, plot_embedding_shift_intervention, plt):
    for _space in ["embedding"]:
        plot_embedding_shift_intervention(embedding_shift_df, "gene_dropout", _space)
    plt.gcf()
    return


@app.cell
def _(embedding_shift_df, plot_embedding_shift_intervention, plt):
    for _space in ["embedding"]:
        plot_embedding_shift_intervention(embedding_shift_df, "gene_shuffle", _space)
    plt.gcf()
    return


@app.cell
def _(embedding_shift_df, plot_embedding_shift_intervention, plt):
    for _space in ["embedding"]:
        plot_embedding_shift_intervention(embedding_shift_df, "local_smoothing", _space)
    plt.gcf()
    return


@app.cell
def _(embedding_shift_df, plot_embedding_shift_intervention, plt):
    for _space in ["embedding"]:
        plot_embedding_shift_intervention(
            embedding_shift_df, "poisson_resampling", _space
        )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Clustering metrics

    **One figure per intervention:** Leiden ARI vs resolution (`embedding` space), all models.
    """)
    return


@app.cell
def _(filter_metrics, metrics_df):
    clustering_df = filter_metrics(
        metrics_df,
        metric_category="clustering_metrics",
        metric_names=["leiden_ari"],
        spaces=["embedding"],
    )
    return (clustering_df,)


@app.cell
def _(clustering_df, plot_clustering_intervention, plt):
    plot_clustering_intervention(clustering_df, "downsample")
    plt.gcf()
    return


@app.cell
def _(clustering_df, plot_clustering_intervention, plt):
    plot_clustering_intervention(clustering_df, "gene_dropout")
    plt.gcf()
    return


@app.cell
def _(clustering_df, plot_clustering_intervention, plt):
    plot_clustering_intervention(clustering_df, "gene_shuffle")
    plt.gcf()
    return


@app.cell
def _(clustering_df, plot_clustering_intervention, plt):
    plot_clustering_intervention(clustering_df, "local_smoothing")
    plt.gcf()
    return


@app.cell
def _(clustering_df, plot_clustering_intervention, plt):
    plot_clustering_intervention(clustering_df, "poisson_resampling")
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cell type and batch metrics

    `silhouette_label`, `graph_connectivity`, and `ilisi_knn` via [scib-metrics](https://scib-metrics.readthedocs.io/). NaN rows usually mean missing `obs` columns or a failed metric (see eval logs).
    """)
    return


@app.cell
def _(metrics_df, mo):
    _batch = metrics_df[metrics_df["metric_category"] == "bio_conservation_metrics"]
    _n_null = _batch["value_mean"].isna().sum()
    mo.vstack(
        [
            mo.callout(
                f"{_n_null} / {len(_batch)} cell_type/batch metric rows have null `value_mean`. "
                "Check `evaluation.cell_type_col` / `batch_col`, reference `obs`, and eval logs; re-run `make evaluate` after fixes.",
                kind="warn",
            ),
            _batch[
                ["model", "intervention_name", "space", "value_mean", "null_value"]
            ].drop_duplicates(),
        ]
    )

    _batch
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Diagnose lower nulls
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(clustering_df):
    # filter to intervention_name is gene_dropout and param_value is 0.2

    subs = clustering_df[
        (clustering_df["intervention_name"] == "gene_dropout") &
        (clustering_df["param_value"] == 0.2)
    ]
    return (subs,)


@app.cell
def _(subs):
    subs
    return


@app.cell
def _(plt, sns, subs):
    def plot_di(subs):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.barplot(
            data=subs,
            x="model",
            y="n_ref_clusters",
            ax=axes[0]
        )
        axes[0].set_title("Number of Reference Clusters")
        axes[0].set_xlabel("Model")
        axes[0].set_ylabel("n_ref_clusters")

        sns.barplot(
            data=subs,
            x="model",
            y="n_manip_clusters",
            ax=axes[1]
        )
        axes[1].set_title("Number of Manipulated Clusters")
        axes[1].set_xlabel("Model")
        axes[1].set_ylabel("n_manip_clusters")

        plt.tight_layout()
        plt.show()

    plot_di(subs)
    return


@app.cell
def _(embedding_shift_df):
    embedding_shift_df
    return


@app.cell
def _(embedding_shift_df):
    embedding_shift_subs = embedding_shift_df[
        (embedding_shift_df["model"].isin(["geneformer", "pca", "scimilarity"])) &
        (embedding_shift_df["intervention_name"] == "gene_dropout") &
        (embedding_shift_df["metric_name"].isin(["within_ref_pairwise_l2", "within_man_pairwise_l2"])) &
        (embedding_shift_df["space"] == "embedding") &
        (embedding_shift_df["param_value"] == 0.2)
    ]
    embedding_shift_subs
    return


if __name__ == "__main__":
    app.run()
