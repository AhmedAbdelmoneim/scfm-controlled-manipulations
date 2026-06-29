import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model patterns — are some models consistently better?

    Compares models across the generated datasets under the project's label-free
    perturbation metrics and under reference scIB scores. The question is *consistency*:
    not which model has the highest dataset-averaged value, but which model is reliably
    ranked above the others **on the same dataset**, across datasets.

    **Why within-dataset ranks.** Datasets differ in difficulty. A model that is always
    2nd can have a lower average value than one that is 1st on easy datasets and last on
    hard ones, yet the former is far more consistent. So the primitive here is the
    **rank of each model within each dataset**, and a model's consistency is the
    *distribution* of those ranks (central rank **and** spread), not a single mean.

    **Design choices**

    - **Per metric, not collapsed.** Each project metric (DistCorr, Local/Global RNX,
      Leiden ARI) is ranked separately, because whether the metrics *agree* on the
      ordering is itself part of "consistently better." A collapsed summary is shown
      second, not first.
    - **scIB bio and batch kept separate.** The project metrics relate to scIB
      bio-conservation positively and to batch-correction negatively, so the 0.6/0.4
      total blends two opposing axes. Ranking by total would manufacture apparent
      disagreement; bio, batch, and total are reported side by side.
    - **Consistency gets a statistic.** Friedman test (is any model reliably better when
      blocking by dataset) and Kendall's W (how strongly datasets agree on the ordering,
      0–1) quantify "consistently."

    **Caveat.** With ~12 datasets and a handful of models, rank statistics are
    low-powered — Kendall's W and the agreement correlations have wide uncertainty, and
    a strong linear baseline (PCA) can rank well on label-free structure metrics. Read
    the patterns, not any single coefficient.
    """)
    return


@app.cell
def _():
    import hashlib
    import json
    from pathlib import Path
    import warnings

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import yaml
    from scipy.stats import friedmanchisquare, spearmanr

    warnings.filterwarnings("ignore", category=FutureWarning)
    sns.set_theme(style="whitegrid", context="notebook")

    REPO_ROOT = Path(__file__).resolve().parents[1]
    CONFIG_DIR = REPO_ROOT / "configs" / "generated" / "normalized_datasets"
    DATASET_PREFIXES = ("atlases__", "sceval__")

    SCIB_METRIC_CATEGORIES = ("bio_conservation_metrics", "batch_correction_metrics")
    SCIB_SPACE = "embedding_reference"
    SCIB_BIO_WEIGHT = 0.6
    EXCLUDED_SCIB_METRICS = (
        "nmi_ari_cluster_labels_kmeans_ari",
        "nmi_ari_cluster_labels_kmeans_nmi",
    )

    PROJECT_SPACE = "embedding"
    EXCLUDE_INTERVENTIONS = ("reference",)
    EXCLUDED_MANIPULATION_PARAMETERS = (
        ("gene_shuffle", {"variant": "chromosome"}),
        ("gene_shuffle", {"variant": "chromosome_control"}),
        ("gene_shuffle", {"variant": "random"}),
        ("downsample", {"fraction": 0.2}),
        ("gene_dropout", {"dropout_rate": 0.9}),
    )
    EXCLUDED_MODELS = ()
    INTERVENTION_COL = "intervention_name"
    CANONICAL_METRICS = {
        "distcorr": "DistCorr",
        "viscore_local_sp": "Local RNX",
        "viscore_global_sp": "Global RNX",
        "leiden_ari": "Leiden ARI",
    }
    CANONICAL_MEDIAN_REDUCE = ("Leiden ARI",)
    COMPOSITE_LABELS = {
        "scib_bio": "scIB bio-conservation",
        "scib_batch": "scIB batch-correction",
        "scib_total": "scIB total",
    }
    return (
        CANONICAL_MEDIAN_REDUCE,
        CANONICAL_METRICS,
        COMPOSITE_LABELS,
        CONFIG_DIR,
        DATASET_PREFIXES,
        EXCLUDED_MANIPULATION_PARAMETERS,
        EXCLUDED_MODELS,
        EXCLUDED_SCIB_METRICS,
        EXCLUDE_INTERVENTIONS,
        INTERVENTION_COL,
        PROJECT_SPACE,
        Path,
        SCIB_BIO_WEIGHT,
        SCIB_METRIC_CATEGORIES,
        SCIB_SPACE,
        friedmanchisquare,
        hashlib,
        json,
        mo,
        np,
        pd,
        plt,
        sns,
        yaml,
    )


@app.cell
def _(
    CONFIG_DIR,
    DATASET_PREFIXES,
    EXCLUDED_MANIPULATION_PARAMETERS,
    EXCLUDED_MODELS,
    EXCLUDED_SCIB_METRICS,
    Path,
    hashlib,
    json,
    pd,
    yaml,
):
    def discover_model_configs() -> pd.DataFrame:
        rows = []
        for cfg_path in sorted(CONFIG_DIR.glob("*.yaml")):
            if not cfg_path.stem.startswith(DATASET_PREFIXES):
                continue
            cfg = yaml.safe_load(cfg_path.read_text())
            dataset_group, dataset_name = cfg_path.stem.split("__", 1)
            evaluation = cfg.get("evaluation") or {}
            rows.append({
                "dataset_key": cfg_path.stem,
                "dataset_group": dataset_group,
                "dataset_name": dataset_name,
                "dataset_id": evaluation.get("dataset_id", f"{dataset_group}/{dataset_name}"),
                "config_path": cfg_path,
                "results_dir": Path(cfg["results_dir"]),
            })
        return pd.DataFrame(rows)

    def _excluded_intervention_ids() -> set[str]:
        ids = set()
        for name, kwargs in EXCLUDED_MANIPULATION_PARAMETERS:
            payload = json.dumps(kwargs, sort_keys=True, default=str)
            digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
            ids.add(f"{name}_{digest}")
        return ids

    def _read_csv(path, *, dataset_key, dataset_id) -> pd.DataFrame:
        df = pd.read_csv(path)
        if df.empty:
            return df
        df["dataset_key"] = dataset_key
        if "dataset_id" not in df.columns or df["dataset_id"].isna().all():
            df["dataset_id"] = dataset_id
        df["source_file"] = path.name
        return df

    def load_model_metric_tables(configs: pd.DataFrame):
        main_frames, scib_frames, rows = [], [], []
        for row in configs.itertuples(index=False):
            evaluation_dir = row.results_dir / "evaluation"
            main_paths = sorted(
                p for p in evaluation_dir.glob("*_metrics.csv") if "_scib_metrics" not in p.stem)
            scib_paths = sorted(evaluation_dir.glob("*_scib_metrics.csv"))
            for path in main_paths:
                main_frames.append(_read_csv(path, dataset_key=row.dataset_key, dataset_id=row.dataset_id))
            for path in scib_paths:
                scib_frames.append(_read_csv(path, dataset_key=row.dataset_key, dataset_id=row.dataset_id))
            rows.append({"dataset_key": row.dataset_key, "dataset_id": row.dataset_id,
                         "main_csvs": len(main_paths), "scib_csvs": len(scib_paths)})
        main_df = pd.concat(main_frames, ignore_index=True) if main_frames else pd.DataFrame()
        scib_df = pd.concat(scib_frames, ignore_index=True) if scib_frames else pd.DataFrame()

        excl_ids = _excluded_intervention_ids()
        if not main_df.empty and "intervention_id" in main_df.columns:
            main_df = main_df[~main_df["intervention_id"].astype(str).isin(excl_ids)].reset_index(drop=True)
        if EXCLUDED_MODELS:
            if not main_df.empty and "model" in main_df.columns:
                main_df = main_df[~main_df["model"].astype(str).isin(EXCLUDED_MODELS)].reset_index(drop=True)
            if not scib_df.empty and "model" in scib_df.columns:
                scib_df = scib_df[~scib_df["model"].astype(str).isin(EXCLUDED_MODELS)].reset_index(drop=True)
        if not scib_df.empty and EXCLUDED_SCIB_METRICS:
            scib_df = scib_df[~scib_df["metric_name"].astype(str).isin(EXCLUDED_SCIB_METRICS)].reset_index(drop=True)
        return main_df, scib_df, pd.DataFrame(rows)

    model_configs = discover_model_configs()
    model_main_raw, model_scib_raw, model_load_summary = load_model_metric_tables(model_configs)
    return model_configs, model_load_summary, model_main_raw, model_scib_raw


@app.cell(hide_code=True)
def _(mo, model_configs, model_load_summary, model_main_raw, model_scib_raw):
    mo.vstack([
        mo.md(
            f"""
            ## Loaded outputs

            - Configs: **{len(model_configs)}** · main rows: **{len(model_main_raw):,}** ·
              scIB rows: **{len(model_scib_raw):,}**
            """
        ),
        model_load_summary,
    ])
    return


@app.cell
def _(
    CANONICAL_MEDIAN_REDUCE,
    CANONICAL_METRICS,
    EXCLUDE_INTERVENTIONS,
    INTERVENTION_COL,
    PROJECT_SPACE,
    SCIB_BIO_WEIGHT,
    SCIB_METRIC_CATEGORIES,
    SCIB_SPACE,
    model_main_raw,
    model_scib_raw,
    pd,
):
    def model_project_values(main_long: pd.DataFrame) -> pd.DataFrame:
        if main_long.empty:
            return pd.DataFrame()
        df = main_long[main_long["space"].astype(str).eq(PROJECT_SPACE)].copy()
        df = df[df["metric_name"].isin(CANONICAL_METRICS)]
        if INTERVENTION_COL in df.columns and EXCLUDE_INTERVENTIONS:
            df = df[~df[INTERVENTION_COL].astype(str).isin(EXCLUDE_INTERVENTIONS)]
        if df.empty:
            return pd.DataFrame()
        df["canonical"] = df["metric_name"].map(CANONICAL_METRICS)
        df["value_mean"] = pd.to_numeric(df["value_mean"], errors="coerce")
        df = df.dropna(subset=["value_mean"])
        has_manip = INTERVENTION_COL in df.columns
        grp = ["dataset_key", "dataset_id", "model", "canonical"] + ([INTERVENTION_COL] if has_manip else [])
        is_median = df["canonical"].isin(CANONICAL_MEDIAN_REDUCE)
        per_manip = pd.concat([
            df[is_median].groupby(grp, observed=True)["value_mean"].median().reset_index(name="project_value"),
            df[~is_median].groupby(grp, observed=True)["value_mean"].mean().reset_index(name="project_value"),
        ], ignore_index=True)
        return (per_manip.groupby(["dataset_key", "dataset_id", "model", "canonical"], observed=True)
                ["project_value"].mean().reset_index())

    def model_scib_composites(scib_long: pd.DataFrame) -> pd.DataFrame:
        if scib_long.empty:
            return pd.DataFrame()
        df = scib_long.copy()
        df["value_mean"] = pd.to_numeric(df["value_mean"], errors="coerce")
        df = df.dropna(subset=["value_mean"])
        df = df[df["metric_category"].isin(SCIB_METRIC_CATEGORIES)]
        df = df[df["space"].astype(str).eq(SCIB_SPACE)]
        if df.empty:
            return pd.DataFrame()
        cat_mean = (df.groupby(["dataset_key", "dataset_id", "model", "metric_category"], observed=True)
                    ["value_mean"].mean().reset_index())
        wide = cat_mean.pivot_table(index=["dataset_key", "dataset_id", "model"],
                                    columns="metric_category", values="value_mean").reset_index()
        bio = wide.get("bio_conservation_metrics")
        batch = wide.get("batch_correction_metrics")
        wide["scib_bio"] = bio
        wide["scib_batch"] = batch
        wide["scib_total"] = (SCIB_BIO_WEIGHT * bio + (1.0 - SCIB_BIO_WEIGHT) * batch
                              if bio is not None and batch is not None
                              else bio if batch is None else batch)
        return wide.melt(id_vars=["dataset_key", "dataset_id", "model"],
                         value_vars=[c for c in ("scib_bio", "scib_batch", "scib_total") if c in wide.columns],
                         var_name="composite", value_name="scib_value").dropna(subset=["scib_value"])

    model_project_overall = model_project_values(model_main_raw)
    model_scib_composite_values = model_scib_composites(model_scib_raw)
    return model_project_overall, model_scib_composite_values


@app.cell
def _(
    friedmanchisquare,
    model_project_overall,
    model_scib_composite_values,
    np,
    pd,
):
    # ---- ranking + consistency core ----
    def within_dataset_ranks(table, *, value_col, metric_col):
        """Rank models within each (dataset, metric); rank 1 = best (highest value)."""
        if table.empty:
            return table.assign(rank=pd.Series(dtype=float))
        df = table.copy()
        df["rank"] = (df.groupby(["dataset_key", metric_col], observed=True)[value_col]
                      .rank(ascending=False, method="average"))
        return df

    def rank_distribution(ranked, *, metric_col):
        """Per (metric, model): central rank and spread across datasets."""
        if ranked.empty:
            return pd.DataFrame()
        g = ranked.groupby([metric_col, "model"], observed=True)["rank"]
        return g.agg(median_rank="median", mean_rank="mean",
                     q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75),
                     rank_std=lambda s: float(s.std(ddof=0)), n_datasets="count").reset_index()

    def friedman_kendall(table, *, value_col, metric_col):
        """Per metric: Friedman test + Kendall's W over complete dataset×model blocks."""
        rows = []
        for metric, g in table.groupby(metric_col, observed=True):
            wide = g.pivot_table(index="dataset_key", columns="model", values=value_col).dropna()
            n, k = wide.shape
            if n < 2 or k < 3:
                rows.append({metric_col: metric, "n_datasets": int(n), "n_models": int(k),
                             "friedman_chi2": np.nan, "friedman_p": np.nan, "kendall_w": np.nan})
                continue
            stat, p = friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
            rows.append({metric_col: metric, "n_datasets": int(n), "n_models": int(k),
                         "friedman_chi2": float(stat), "friedman_p": float(p),
                         "kendall_w": float(stat / (n * (k - 1)))})
        return pd.DataFrame(rows)

    project_ranked = within_dataset_ranks(
        model_project_overall, value_col="project_value", metric_col="canonical")
    scib_ranked = within_dataset_ranks(
        model_scib_composite_values, value_col="scib_value", metric_col="composite")

    project_rank_dist = rank_distribution(project_ranked, metric_col="canonical")
    scib_rank_dist = rank_distribution(scib_ranked, metric_col="composite")

    project_consistency = friedman_kendall(
        model_project_overall, value_col="project_value", metric_col="canonical")
    scib_consistency = friedman_kendall(
        model_scib_composite_values, value_col="scib_value", metric_col="composite")
    return project_consistency, project_ranked, scib_consistency, scib_ranked


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Per-metric model rank distributions

    Each panel is one metric. For every model, the distribution of its within-dataset
    rank across datasets (1 = best). **Low and tight = consistently better.** A box
    centred low but tall is a model that is sometimes best and sometimes not — high
    average but not consistent. Project metrics and scIB composites are shown in the
    same rank space so the orderings are directly comparable.
    """)
    return


@app.cell
def _(pd, plt, project_ranked, scib_ranked, sns):
    def _draw_rank_box_panels(
        axes_row,
        ranked,
        metric_col,
        metric_order,
        *,
        n_models,
        model_order,
        color,
    ):
        metrics = [m for m in metric_order if m in set(ranked[metric_col])] if metric_order else \
                  sorted(ranked[metric_col].unique())
        if not metrics:
            axes_row[0].text(0.5, 0.5, "No data", ha="center", va="center")
            axes_row[0].set_axis_off()
            for ax in axes_row[1:]:
                ax.set_axis_off()
            return
        for ax, metric in zip(axes_row, metrics):
            sub = ranked[ranked[metric_col] == metric]
            sns.boxplot(data=sub, y="model", x="rank", order=model_order, ax=ax,
                        color=color, fliersize=2, width=0.6)
            sns.stripplot(data=sub, y="model", x="rank", order=model_order, ax=ax,
                          color="black", size=3, alpha=0.5, jitter=0.18)
            ax.set_title(metric, fontsize=11)
            ax.set_xlabel("rank within dataset")
            ax.set_ylabel("")
            ax.set_xlim(0.5, n_models + 0.5)
            ax.invert_xaxis()
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes_row[len(metrics):]:
            ax.set_axis_off()

    _pn = max(project_ranked["model"].nunique() if not project_ranked.empty else 1,
              scib_ranked["model"].nunique() if not scib_ranked.empty else 1)
    rank_box_shared_order = pd.concat(
        [
            project_ranked[["model", "rank"]] if not project_ranked.empty else pd.DataFrame(),
            scib_ranked[["model", "rank"]] if not scib_ranked.empty else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    rank_box_model_order = (
        rank_box_shared_order.groupby("model", observed=True)["rank"].mean().sort_values().index.tolist()
        if not rank_box_shared_order.empty
        else []
    )
    rank_box_cols = max(4, 3)
    rank_box_fig, rank_box_axes = plt.subplots(
        2,
        rank_box_cols,
        figsize=(3.4 * rank_box_cols + 1, 0.9 * _pn + 4.2),
        sharey=True,
        squeeze=False,
    )
    palette = sns.color_palette("colorblind")
    _draw_rank_box_panels(
        rank_box_axes[0],
        project_ranked,
        "canonical",
        ["DistCorr", "Local RNX", "Global RNX", "Leiden ARI"],
        n_models=_pn,
        model_order=rank_box_model_order,
        color=palette[0],
    )
    _draw_rank_box_panels(
        rank_box_axes[1],
        scib_ranked,
        "composite",
        ["scib_bio", "scib_batch", "scib_total"],
        n_models=_pn,
        model_order=rank_box_model_order,
        color=palette[1],
    )
    rank_box_axes[0, 0].set_ylabel("Project metrics", fontsize=11)
    rank_box_axes[1, 0].set_ylabel("scIB composites", fontsize=11)
    rank_box_fig.tight_layout()
    rank_box_fig
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Consistency statistics

    **Friedman p**: is *any* model reliably ranked above others when blocking by dataset
    (small p = yes). **Kendall's W** (0–1): how strongly the datasets agree on the model
    ordering — the direct measure of "consistently." W near 1 = the same ordering on
    nearly every dataset; W near 0 = the ordering is essentially random across datasets.
    At ~12 datasets these are indicative, not decisive.
    """)
    return


@app.cell
def _(COMPOSITE_LABELS, mo, project_consistency, scib_consistency):
    def _fmt(df, name_col, label_map=None):
        if df.empty:
            return df
        out = df.copy()
        if label_map:
            out[name_col] = out[name_col].map(lambda v: label_map.get(v, v))
        for c in ("friedman_chi2", "kendall_w"):
            if c in out.columns:
                out[c] = out[c].round(3)
        if "friedman_p" in out.columns:
            out["friedman_p"] = out["friedman_p"].map(
                lambda v: f"{v:.1e}" if (v == v) else "n/a")
        return out

    mo.vstack([
        mo.md("**Project metrics:**"),
        _fmt(project_consistency, "canonical"),
        mo.md("**scIB composites:**"),
        _fmt(scib_consistency, "composite", COMPOSITE_LABELS),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Model × metric rank map

    Mean within-dataset rank per model and metric (lower = better, green). Project
    metrics and scIB composites side by side, bio/batch/total kept separate. Reading
    down a column shows the ordering under that metric; reading across a row shows
    whether a model is consistently placed across metrics or swings (e.g. strong on
    bio, weak on batch).
    """)
    return


@app.cell
def _(COMPOSITE_LABELS, pd, plt, project_ranked, scib_ranked, sns):
    def _rank_map(
        ax,
        ranked,
        metric_col,
        metric_order,
        col_labels_map,
        title,
        *,
        model_order,
        vmax,
        show_colorbar,
        show_model_labels,
    ):
        if ranked.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center"); ax.set_axis_off(); return
        mat = (ranked.groupby(["model", metric_col], observed=True)["rank"].mean()
               .reset_index().pivot(index="model", columns=metric_col, values="rank"))
        cols = [c for c in metric_order if c in mat.columns] if metric_order else list(mat.columns)
        mat = mat.reindex(columns=cols)
        mat = mat.reindex(index=model_order)
        sns.heatmap(mat, ax=ax, cmap="RdYlGn_r", vmin=1, vmax=vmax, annot=True, fmt=".1f",
                    linewidths=0.4, cbar=show_colorbar,
                    cbar_kws={"label": "mean rank (1 = best)"} if show_colorbar else None)
        if col_labels_map:
            ax.set_xticklabels([col_labels_map.get(c, c) for c in mat.columns], rotation=20, ha="right")
        if not show_model_labels:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.set_title(title); ax.set_xlabel(""); ax.set_ylabel("")

    shared_rank_order = pd.concat(
        [
            project_ranked[["model", "rank"]] if not project_ranked.empty else pd.DataFrame(),
            scib_ranked[["model", "rank"]] if not scib_ranked.empty else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    model_order = (
        shared_rank_order.groupby("model", observed=True)["rank"].mean().sort_values().index.tolist()
        if not shared_rank_order.empty
        else []
    )
    vmax = max(
        project_ranked["model"].nunique() if not project_ranked.empty else 1,
        scib_ranked["model"].nunique() if not scib_ranked.empty else 1,
    )
    rank_map_fig_obj, rank_map_axes = plt.subplots(1, 2, figsize=(15, 0.5 * max(
        len(model_order) if model_order else 1, 1) + 3),
        gridspec_kw={"width_ratios": [4, 3]})
    _rank_map(rank_map_axes[0], project_ranked, "canonical",
              ["DistCorr", "Local RNX", "Global RNX", "Leiden ARI"], None,
              "Project metrics — mean rank",
              model_order=model_order, vmax=vmax, show_colorbar=False, show_model_labels=True)
    _rank_map(rank_map_axes[1], scib_ranked, "composite",
              [c for c in COMPOSITE_LABELS], COMPOSITE_LABELS, "scIB composites — mean rank",
              model_order=model_order, vmax=vmax, show_colorbar=True, show_model_labels=False)
    rank_map_fig_obj.tight_layout()
    rank_map_fig = rank_map_fig_obj
    rank_map_fig
    return


if __name__ == "__main__":
    app.run()
