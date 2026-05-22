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
        DATASET_ID,
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
    return metrics_df, overview, params_df


@app.cell
def _(mo, overview):
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
def _(MODEL_ORDER, pd, plt, sns):
    def _sort_plot_df(df: pd.DataFrame, x: str = "param_value") -> pd.DataFrame:
        out = df.copy()
        out["model"] = pd.Categorical(out["model"], categories=MODEL_ORDER, ordered=True)
        if x not in out.columns:
            return out
        if pd.api.types.is_numeric_dtype(out[x]):
            return out.sort_values(x)
        return out.sort_values(x, key=lambda s: s.astype(str))

    def plot_models_line(
        df: pd.DataFrame,
        *,
        x: str = "param_value",
        y: str = "value_mean",
        hue: str = "model",
        col: str | None = "intervention_name",
        row: str | None = None,
        title: str | None = None,
        height: float = 3.2,
        aspect: float = 1.15,
    ):
        if df.empty:
            _fig, _ax = plt.subplots(figsize=(6, 3))
            _ax.text(0.5, 0.5, "No plottable rows", ha="center", va="center")
            _ax.set_axis_off()
            return _fig

        plot_df = _sort_plot_df(df, x=x)
        kwargs: dict = {
            "data": plot_df,
            "x": x,
            "y": y,
            "hue": hue,
            "kind": "line",
            "marker": "o",
            "linewidth": 2,
            "palette": "tab10",
            "height": height,
            "aspect": aspect,
            "facet_kws": {"sharey": False},
        }
        if col is not None:
            kwargs["col"] = col
        if row is not None:
            kwargs["row"] = row

        g = sns.relplot(**kwargs)
        g.set(xlabel=plot_df["param_key"].dropna().iloc[0] if "param_key" in plot_df.columns and plot_df["param_key"].notna().any() else x)
        for _ax in g.axes.flatten():
            _ax.tick_params(axis="x", rotation=25)
        if title:
            g.fig.suptitle(title, y=1.03)
        plt.tight_layout()
        return g.fig

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

    return filter_metrics, plot_models_line


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Embedding stats

    Global embedding geometry summaries (`mean_row_l2_norm`, column means/variances), faceted by **raw** vs **embedding** space.
    Reference-side metrics should match across models; manipulation-side metrics reflect model-specific embeddings.
    """)
    return


@app.cell
def _(filter_metrics, metrics_df, plot_models_line, plt):
    _stats_names = [
        "mean_row_l2_norm_ref",
        "mean_row_l2_norm_man",
        "col_mean_ref",
        "col_mean_man",
        "col_variance_ref",
        "col_variance_man",
    ]
    for _space in ["raw", "embedding"]:
        _sub = filter_metrics(
            metrics_df,
            metric_category="embedding_stats",
            metric_names=_stats_names,
            spaces=[_space],
        )
        plot_models_line(
            _sub,
            row="metric_name",
            col="intervention_name",
            title=f"embedding_stats — space={_space}",
        )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Embedding shift

    Paired geometric shift between reference and manipulation (per-cell L2, pairwise cosines/L2 within ref/man).
    Some `shift_pairwise_cosine` rows are NaN and are dropped before plotting.
    """)
    return


@app.cell
def _(filter_metrics, metrics_df, plot_models_line, plt):
    _shift_names = [
        "paired_cell_l2_norm",
        "shift_pairwise_cosine",
        "within_ref_pairwise_l2",
        "within_man_pairwise_l2",
    ]
    for _space in ["raw", "embedding"]:
        _sub = filter_metrics(
            metrics_df,
            metric_category="embedding_shift",
            metric_names=_shift_names,
            spaces=[_space],
        )
        plot_models_line(
            _sub,
            row="metric_name",
            col="intervention_name",
            title=f"embedding_shift — space={_space}",
        )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Embedding shift gain

    **embedding minus raw** (`space=embedding_minus_raw`): positive values mean the model embedding amplifies the intervention signal vs raw counts.
    """)
    return


@app.cell
def _(filter_metrics, metrics_df, plot_models_line, plt):
    _gain_names = [
        "paired_cell_l2_norm",
        "shift_pairwise_cosine",
        "within_ref_pairwise_l2",
        "within_man_pairwise_l2",
    ]
    _sub = filter_metrics(
        metrics_df,
        metric_category="embedding_shift_gain",
        metric_names=_gain_names,
        spaces=["embedding_minus_raw"],
    )
    plot_models_line(
        _sub,
        row="metric_name",
        col="intervention_name",
        title="embedding_shift_gain",
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### KNN metrics

    Neighborhood preservation (`knn_recall`) and diffusion divergence (`diffusion_js`, `diffusion_sym_kl`), faceted by **k** or **diffusion_t** and **space**.
    """)
    return


@app.cell
def _(filter_metrics, metrics_df, plot_models_line, plt):
    for _space in ["raw", "embedding"]:
        for _k in sorted(metrics_df["k"].dropna().unique()):
            _sub = filter_metrics(
                metrics_df,
                metric_category="knn_metrics",
                metric_names=["knn_recall"],
                spaces=[_space],
            )
            _sub = _sub[_sub["k"] == _k]
            plot_models_line(
                _sub,
                col="intervention_name",
                title=f"knn_recall — space={_space}, k={int(_k)}",
            )
    plt.gcf()
    return


@app.cell
def _(filter_metrics, metrics_df, plot_models_line, plt):
    for _metric in ["diffusion_js", "diffusion_sym_kl"]:
        for _space in ["raw", "embedding"]:
            for _t in sorted(metrics_df["diffusion_t"].dropna().unique()):
                _sub = filter_metrics(
                    metrics_df,
                    metric_category="knn_metrics",
                    metric_names=[_metric],
                    spaces=[_space],
                )
                _sub = _sub[_sub["diffusion_t"] == _t]
                plot_models_line(
                    _sub,
                    col="intervention_name",
                    title=f"{_metric} — space={_space}, diffusion_t={int(_t)}",
                )
    plt.gcf()
    return


@app.cell
def _(MODEL_ORDER, filter_metrics, metrics_df, plt, sns):
    _sub = filter_metrics(
        metrics_df,
        metric_category="knn_metrics",
        metric_names=["knn_recall"],
        spaces=["embedding"],
    )
    _sub = _sub[_sub["k"] == 15]
    if not _sub.empty:
        _pivot = _sub.pivot_table(
            index="model",
            columns="intervention_id",
            values="value_mean",
            aggfunc="first",
            observed=True,
        )
        _pivot = _pivot.reindex(MODEL_ORDER)
        _fig, _ax = plt.subplots(figsize=(14, 4))
        sns.heatmap(_pivot, ax=_ax, cmap="viridis", annot=False)
        _ax.set_title("knn_recall (embedding, k=15) — model x intervention")
        plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### KNN metrics gain

    `knn_recall` gain (embedding minus raw) across intervention sweeps.
    """)
    return


@app.cell
def _(filter_metrics, metrics_df, plot_models_line, plt):
    for _k in sorted(metrics_df["k"].dropna().unique()):
        _sub = filter_metrics(
            metrics_df,
            metric_category="knn_metrics_gain",
            metric_names=["knn_recall"],
            spaces=["embedding_minus_raw"],
        )
        _sub = _sub[_sub["k"] == _k]
        plot_models_line(
            _sub,
            col="intervention_name",
            title=f"knn_recall gain — k={int(_k)}",
        )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Clustering metrics

    Leiden ARI between reference and manipulation clusterings (`space=embedding`), vs Leiden resolution.
    """)
    return


@app.cell
def _(filter_metrics, metrics_df, plot_models_line, plt):
    _sub = filter_metrics(
        metrics_df,
        metric_category="clustering_metrics",
        metric_names=["leiden_ari"],
        spaces=["embedding"],
    )
    _sub = _sub.copy()
    _sub["param_value"] = _sub["resolution"]
    _sub["param_key"] = "resolution"
    plot_models_line(
        _sub,
        x="resolution",
        col="intervention_name",
        title="leiden_ari vs resolution (embedding)",
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cell type and batch metrics

    `batch_ilisi` (scIB iLISI) — currently all null for this dataset (likely missing `batch` in obs).
    """)
    return


@app.cell
def _(metrics_df, mo):
    _batch = metrics_df[metrics_df["metric_category"] == "cell_type_and_batch_metrics"]
    _n_null = _batch["value_mean"].isna().sum()
    mo.vstack(
        [
            mo.callout(
                f"All {_n_null} / {len(_batch)} `batch_ilisi` rows have null `value_mean`. "
                "No comparison plots until batch labels exist in the dataset obs.",
                kind="warn",
            ),
            _batch[
                ["model", "intervention_name", "space", "value_mean", "null_value"]
            ].drop_duplicates(),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
