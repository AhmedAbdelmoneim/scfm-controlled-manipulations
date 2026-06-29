import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dataset patterns

    Summarize trends across **datasets**, aggregating over models. The goal is to see
    whether some datasets are consistently easier/higher-scoring or harder/lower-scoring
    under the project's perturbation metrics and reference scIB scores.
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
        hashlib,
        json,
        mo,
        np,
        pd,
        plt,
        sns,
        yaml,
    )


@app.cell(hide_code=True)
def _(EXCLUDED_MODELS, EXCLUDED_SCIB_METRICS, mo):
    mo.md(
        f"""
        ## Configuration

        - Excluded scIB metrics: `{", ".join(EXCLUDED_SCIB_METRICS)}`
        - Excluded models: `{", ".join(EXCLUDED_MODELS) if EXCLUDED_MODELS else "none"}`
        """
    )
    return


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
    def discover_dataset_configs() -> pd.DataFrame:
        rows = []
        for cfg_path in sorted(CONFIG_DIR.glob("*.yaml")):
            if not cfg_path.stem.startswith(DATASET_PREFIXES):
                continue
            cfg = yaml.safe_load(cfg_path.read_text())
            dataset_group, dataset_name = cfg_path.stem.split("__", 1)
            evaluation = cfg.get("evaluation") or {}
            rows.append(
                {
                    "dataset_key": cfg_path.stem,
                    "dataset_group": dataset_group,
                    "dataset_name": dataset_name,
                    "dataset_id": evaluation.get("dataset_id", f"{dataset_group}/{dataset_name}"),
                    "config_path": cfg_path,
                    "results_dir": Path(cfg["results_dir"]),
                }
            )
        return pd.DataFrame(rows)

    def excluded_intervention_ids_for_datasets() -> set[str]:
        ids = set()
        for name, kwargs in EXCLUDED_MANIPULATION_PARAMETERS:
            payload = json.dumps(kwargs, sort_keys=True, default=str)
            digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
            ids.add(f"{name}_{digest}")
        return ids

    def read_metric_csv_for_datasets(path, *, dataset_key, dataset_id) -> pd.DataFrame:
        df = pd.read_csv(path)
        if df.empty:
            return df
        df["dataset_key"] = dataset_key
        if "dataset_id" not in df.columns or df["dataset_id"].isna().all():
            df["dataset_id"] = dataset_id
        df["source_file"] = path.name
        return df

    def load_dataset_metric_tables(configs: pd.DataFrame):
        main_frames, scib_frames, rows = [], [], []
        for row in configs.itertuples(index=False):
            evaluation_dir = row.results_dir / "evaluation"
            main_paths = sorted(
                p for p in evaluation_dir.glob("*_metrics.csv") if "_scib_metrics" not in p.stem
            )
            scib_paths = sorted(evaluation_dir.glob("*_scib_metrics.csv"))
            for path in main_paths:
                main_frames.append(
                    read_metric_csv_for_datasets(
                        path, dataset_key=row.dataset_key, dataset_id=row.dataset_id
                    )
                )
            for path in scib_paths:
                scib_frames.append(
                    read_metric_csv_for_datasets(
                        path, dataset_key=row.dataset_key, dataset_id=row.dataset_id
                    )
                )
            rows.append(
                {
                    "dataset_key": row.dataset_key,
                    "dataset_id": row.dataset_id,
                    "main_csvs": len(main_paths),
                    "scib_csvs": len(scib_paths),
                }
            )
        main_df = pd.concat(main_frames, ignore_index=True) if main_frames else pd.DataFrame()
        scib_df = pd.concat(scib_frames, ignore_index=True) if scib_frames else pd.DataFrame()

        excluded_intervention_ids = excluded_intervention_ids_for_datasets()
        if not main_df.empty and "intervention_id" in main_df.columns:
            main_df = main_df[
                ~main_df["intervention_id"].astype(str).isin(excluded_intervention_ids)
            ].reset_index(drop=True)
        if not main_df.empty and EXCLUDED_MODELS and "model" in main_df.columns:
            main_df = main_df[~main_df["model"].astype(str).isin(EXCLUDED_MODELS)].reset_index(
                drop=True
            )
        if not scib_df.empty and EXCLUDED_MODELS and "model" in scib_df.columns:
            scib_df = scib_df[~scib_df["model"].astype(str).isin(EXCLUDED_MODELS)].reset_index(
                drop=True
            )
        if not scib_df.empty and EXCLUDED_SCIB_METRICS:
            scib_df = scib_df[
                ~scib_df["metric_name"].astype(str).isin(EXCLUDED_SCIB_METRICS)
            ].reset_index(drop=True)
        return main_df, scib_df, pd.DataFrame(rows)

    dataset_configs = discover_dataset_configs()
    dataset_main_raw, dataset_scib_raw, dataset_load_summary = load_dataset_metric_tables(
        dataset_configs
    )
    return dataset_configs, dataset_load_summary, dataset_main_raw, dataset_scib_raw


@app.cell(hide_code=True)
def _(dataset_configs, dataset_load_summary, dataset_main_raw, dataset_scib_raw, mo):
    mo.vstack([
        mo.md(
            f"""
            ## Loaded generated outputs

            - Configs discovered: **{len(dataset_configs)}**
            - Main metric rows: **{len(dataset_main_raw):,}**
            - scIB metric rows: **{len(dataset_scib_raw):,}**
            """
        ),
        dataset_load_summary,
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
    dataset_main_raw,
    dataset_scib_raw,
    pd,
):
    def dataset_project_values(main_long: pd.DataFrame) -> pd.DataFrame:
        if main_long.empty:
            return pd.DataFrame()
        df = main_long.copy()
        df = df[df["space"].astype(str).eq(PROJECT_SPACE)]
        df = df[df["metric_name"].isin(CANONICAL_METRICS)]
        if INTERVENTION_COL in df.columns and EXCLUDE_INTERVENTIONS:
            df = df[~df[INTERVENTION_COL].astype(str).isin(EXCLUDE_INTERVENTIONS)]
        if df.empty:
            return pd.DataFrame()
        df["canonical"] = df["metric_name"].map(CANONICAL_METRICS)
        df["value_mean"] = pd.to_numeric(df["value_mean"], errors="coerce")
        df = df.dropna(subset=["value_mean"])

        has_manip = INTERVENTION_COL in df.columns
        group_cols = ["dataset_key", "dataset_id", "model", "canonical"]
        group_cols += [INTERVENTION_COL] if has_manip else []
        is_median = df["canonical"].isin(CANONICAL_MEDIAN_REDUCE)
        per_manip = pd.concat(
            [
                df[is_median]
                .groupby(group_cols, observed=True)["value_mean"]
                .median()
                .reset_index(name="project_value"),
                df[~is_median]
                .groupby(group_cols, observed=True)["value_mean"]
                .mean()
                .reset_index(name="project_value"),
            ],
            ignore_index=True,
        )
        if not has_manip:
            per_manip[INTERVENTION_COL] = "all"
        return (
            per_manip.groupby(
                ["dataset_key", "dataset_id", "model", "canonical"], observed=True
            )["project_value"]
            .mean()
            .reset_index()
        )

    def dataset_scib_composites(scib_long: pd.DataFrame) -> pd.DataFrame:
        if scib_long.empty:
            return pd.DataFrame()
        df = scib_long.copy()
        df["value_mean"] = pd.to_numeric(df["value_mean"], errors="coerce")
        df = df.dropna(subset=["value_mean"])
        df = df[df["metric_category"].isin(SCIB_METRIC_CATEGORIES)]
        df = df[df["space"].astype(str).eq(SCIB_SPACE)]
        if df.empty:
            return pd.DataFrame()
        cat_mean = (
            df.groupby(["dataset_key", "dataset_id", "model", "metric_category"], observed=True)[
                "value_mean"
            ]
            .mean()
            .reset_index()
        )
        wide = cat_mean.pivot_table(
            index=["dataset_key", "dataset_id", "model"],
            columns="metric_category",
            values="value_mean",
        ).reset_index()
        bio = wide.get("bio_conservation_metrics")
        batch = wide.get("batch_correction_metrics")
        wide["scib_bio"] = bio
        wide["scib_batch"] = batch
        wide["scib_total"] = (
            SCIB_BIO_WEIGHT * bio + (1.0 - SCIB_BIO_WEIGHT) * batch
            if bio is not None and batch is not None
            else bio if batch is None else batch
        )
        return wide.melt(
            id_vars=["dataset_key", "dataset_id", "model"],
            value_vars=[c for c in ("scib_bio", "scib_batch", "scib_total") if c in wide.columns],
            var_name="composite",
            value_name="scib_value",
        ).dropna(subset=["scib_value"])

    dataset_project_overall = dataset_project_values(dataset_main_raw)
    dataset_scib_composite_values = dataset_scib_composites(dataset_scib_raw)
    return dataset_project_overall, dataset_scib_composite_values


@app.cell
def _(COMPOSITE_LABELS, dataset_project_overall, dataset_scib_composite_values, pd):
    def dataset_mean_and_rank(table, *, value_col, metric_col):
        if table.empty:
            return pd.DataFrame(), pd.DataFrame()
        means = (
            table.groupby(["dataset_key", "dataset_id", metric_col], observed=True)[value_col]
            .mean()
            .reset_index(name="mean_value")
        )
        means["rank_percentile"] = means.groupby(metric_col, observed=True)["mean_value"].rank(
            pct=True, ascending=True
        )
        ranking = (
            means.groupby(["dataset_key", "dataset_id"], observed=True)
            .agg(
                ranking_score=("rank_percentile", "mean"),
                mean_value=("mean_value", "mean"),
                n_metrics=(metric_col, "nunique"),
            )
            .reset_index()
            .sort_values("ranking_score", ascending=False)
        )
        ranking["dataset_group"] = ranking["dataset_key"].astype(str).str.split("__").str[0]
        ranking["rank"] = range(1, len(ranking) + 1)
        return means, ranking

    project_dataset_means, project_dataset_ranking = dataset_mean_and_rank(
        dataset_project_overall,
        value_col="project_value",
        metric_col="canonical",
    )
    scib_dataset_means, scib_dataset_ranking = dataset_mean_and_rank(
        dataset_scib_composite_values,
        value_col="scib_value",
        metric_col="composite",
    )
    if not scib_dataset_means.empty:
        scib_dataset_means["metric_label"] = scib_dataset_means["composite"].map(COMPOSITE_LABELS)
    return project_dataset_means, project_dataset_ranking, scib_dataset_means, scib_dataset_ranking


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset rankings

    Ranking scores are average percentile ranks across metrics after aggregating over
    models. Higher-ranking datasets score higher more consistently across metrics.
    """)
    return


@app.cell
def _(
    COMPOSITE_LABELS,
    mo,
    project_dataset_means,
    project_dataset_ranking,
    scib_dataset_means,
    scib_dataset_ranking,
    plt,
    sns,
):
    def plot_dataset_rankings(project_rank, scib_rank):
        fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.38 * max(len(project_rank), len(scib_rank)) + 2)))
        palette = {"atlases": sns.color_palette("colorblind")[0], "sceval": sns.color_palette("colorblind")[1]}
        panels = [
            (axes[0], project_rank, "Project metric dataset ranking"),
            (axes[1], scib_rank, "scIB composite dataset ranking"),
        ]
        for ax, ranking, title in panels:
            if ranking.empty:
                ax.text(0.5, 0.5, "No ranking data", ha="center", va="center")
                ax.set_axis_off()
                continue
            data = ranking.sort_values("ranking_score", ascending=True)
            colors = [palette.get(group, "0.5") for group in data["dataset_group"]]
            ax.barh(data["dataset_id"], data["ranking_score"], color=colors)
            ax.set_xlim(0, 1.05)
            ax.set_xlabel("mean percentile rank")
            ax.set_title(title)
            ax.spines[["top", "right"]].set_visible(False)
        handles = [
            plt.Line2D([0], [0], color=color, lw=6, label=label)
            for label, color in palette.items()
        ]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
        fig.tight_layout()
        fig.subplots_adjust(top=0.90)
        return fig

    def plot_dataset_metric_heatmaps(project_means, scib_means):
        fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.38 * max(project_means["dataset_id"].nunique() if not project_means.empty else 1, scib_means["dataset_id"].nunique() if not scib_means.empty else 1) + 2)))
        if project_means.empty:
            axes[0].text(0.5, 0.5, "No project metric means", ha="center", va="center")
            axes[0].set_axis_off()
        else:
            project_mat = project_means.pivot(index="dataset_id", columns="canonical", values="mean_value")
            sns.heatmap(project_mat, ax=axes[0], cmap="viridis", annot=True, fmt=".2f")
            axes[0].set_title("Project metric means by dataset")
            axes[0].set_xlabel("")
            axes[0].set_ylabel("")
        if scib_means.empty:
            axes[1].text(0.5, 0.5, "No scIB metric means", ha="center", va="center")
            axes[1].set_axis_off()
        else:
            scib_mat = scib_means.pivot(index="dataset_id", columns="composite", values="mean_value")
            scib_mat = scib_mat.reindex(columns=[c for c in COMPOSITE_LABELS if c in scib_mat.columns])
            sns.heatmap(scib_mat, ax=axes[1], cmap="viridis", vmin=0, vmax=1, annot=True, fmt=".2f")
            axes[1].set_xticklabels([COMPOSITE_LABELS.get(c, c) for c in scib_mat.columns], rotation=20, ha="right")
            axes[1].set_title("scIB composite means by dataset")
            axes[1].set_xlabel("")
            axes[1].set_ylabel("")
        fig.tight_layout()
        return fig

    dataset_ranking_fig = plot_dataset_rankings(project_dataset_ranking, scib_dataset_ranking)
    dataset_metric_heatmap_fig = plot_dataset_metric_heatmaps(
        project_dataset_means, scib_dataset_means
    )
    mo.vstack([dataset_ranking_fig, dataset_metric_heatmap_fig])
    return dataset_metric_heatmap_fig, dataset_ranking_fig


@app.cell(hide_code=True)
def _(mo, project_dataset_ranking, scib_dataset_ranking):
    mo.vstack([
        mo.md("### Ranking tables"),
        mo.md("**Project metrics:**"),
        project_dataset_ranking,
        mo.md("**scIB composites:**"),
        scib_dataset_ranking,
    ])
    return


if __name__ == "__main__":
    app.run()
