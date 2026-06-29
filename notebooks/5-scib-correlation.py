import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # scIB correlation across datasets — convergent validity & divergence

    Exploratory check of how the project's label-free structure-preservation metrics
    relate to scIB reference-embedding scores across the generated `atlases` and
    `sceval` datasets. There is no single right answer here: the goal is to see both
    where the metrics **agree** with scIB (convergent validity) and where they
    **diverge** (potential value-add — structure scIB's bio/batch scores miss).

    **Design**

    - **Unit:** one value per *dataset × model* per metric. Aggregation is computed per
      model and also pooled across models (with the non-independence caveat noted).
    - **scIB composites are computed here** from the sub-metrics (the CSVs hold only
      sub-metrics): a **bio-conservation** mean, a **batch-correction** mean, and a
      **total** = 0.6·bio + 0.4·batch (the standard scIB weighting). No cross-method
      min–max scaling is applied, so each dataset×model score is self-contained and
      appropriate for a correlation across datasets.
    - **Project metrics are collapsed to canonical forms** (Leiden ARI as the
      resolution-median; ViScore Sl/Sg, DistCorr as single values) and **averaged
      across manipulations** to a per-dataset "perturbation-robustness" summary, with a
      per-manipulation drill-down kept separately. The trivial `reference` intervention
      is excluded.
    - **Statistics:** Pearson or Spearman correlations are reported as point estimates.
      At *n*≈12 per model, no single coefficient should be over-read.
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
    from scipy import stats
    import seaborn as sns
    import yaml

    warnings.filterwarnings("ignore", category=FutureWarning)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.figsize"] = (12, 8)

    REPO_ROOT = Path(__file__).resolve().parents[1]
    CONFIG_DIR = REPO_ROOT / "configs" / "generated" / "normalized_datasets"
    DATASET_PREFIXES = ("atlases__", "sceval__")

    # ---- correlation settings ----
    CORRELATION_METHOD = "pearson"          # robust to outliers at low n
    ALLOWED_CORRELATION_METHODS = {"pearson", "spearman"}
    MIN_DATASETS_FOR_CORRELATION = 4
    TOP_N_DISAGREEMENTS = 12

    # ---- scIB composite settings ----
    SCIB_METRIC_CATEGORIES = ("bio_conservation_metrics", "batch_correction_metrics")
    SCIB_SPACE = "embedding_reference"
    SCIB_BIO_WEIGHT = 0.6                    # total = bio_weight*bio + (1-bio_weight)*batch
    SCIB_COMPOSITES = ("scib_bio", "scib_batch", "scib_total")
    EXCLUDED_SCIB_METRICS = (
        "nmi_ari_cluster_labels_kmeans_ari",
        "nmi_ari_cluster_labels_kmeans_nmi",
    )

    # ---- project metric settings ----
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
    # metric_name -> canonical label. Add/rename to match what is in the CSVs.
    CANONICAL_METRICS = {
        "distcorr": "DistCorr",
        "viscore_local_sp": "Local RNX",
        "viscore_global_sp": "Global RNX",
        "leiden_ari": "Leiden ARI",
    }
    # canonical labels reduced by MEDIAN over their parameter rows (e.g. resolution);
    # everything else reduced by MEAN.
    CANONICAL_MEDIAN_REDUCE = ("Leiden ARI",)

    if CORRELATION_METHOD not in ALLOWED_CORRELATION_METHODS:
        raise ValueError(
            f"CORRELATION_METHOD must be one of {sorted(ALLOWED_CORRELATION_METHODS)}"
        )

    COMPOSITE_LABELS = {
        "scib_bio": "scIB bio-conservation",
        "scib_batch": "scIB batch-correction",
        "scib_total": "scIB total (0.6·bio+0.4·batch)",
    }
    return (
        CANONICAL_MEDIAN_REDUCE,
        CANONICAL_METRICS,
        COMPOSITE_LABELS,
        CONFIG_DIR,
        CORRELATION_METHOD,
        DATASET_PREFIXES,
        EXCLUDED_MANIPULATION_PARAMETERS,
        EXCLUDED_MODELS,
        EXCLUDED_SCIB_METRICS,
        EXCLUDE_INTERVENTIONS,
        INTERVENTION_COL,
        MIN_DATASETS_FOR_CORRELATION,
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
        stats,
        yaml,
    )


@app.cell(hide_code=True)
def _(
    CORRELATION_METHOD,
    EXCLUDED_MANIPULATION_PARAMETERS,
    EXCLUDED_MODELS,
    EXCLUDED_SCIB_METRICS,
    MIN_DATASETS_FOR_CORRELATION,
    mo,
):
    mo.md(
        f"""
        ## Configuration

        - Correlation method: `{CORRELATION_METHOD}`
        - Minimum datasets per correlation: `{MIN_DATASETS_FOR_CORRELATION}`
        - scIB total weighting: `0.6·bio + 0.4·batch`
        - Excluded scIB metrics: `{", ".join(EXCLUDED_SCIB_METRICS)}`
        - Excluded manipulations: `{", ".join(f"{name} {kwargs}" for name, kwargs in EXCLUDED_MANIPULATION_PARAMETERS)}`
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
    def discover_generated_configs() -> pd.DataFrame:
        rows = []
        for cfg_path in sorted(CONFIG_DIR.glob("*.yaml")):
            if not cfg_path.stem.startswith(DATASET_PREFIXES):
                continue
            cfg = yaml.safe_load(cfg_path.read_text())
            dataset_group = cfg_path.stem.split("__", 1)[0]
            dataset_name = cfg_path.stem.split("__", 1)[1]
            evaluation = cfg.get("evaluation") or {}
            rows.append(
                {
                    "dataset_key": cfg_path.stem,
                    "dataset_group": dataset_group,
                    "dataset_name": dataset_name,
                    "dataset_id": evaluation.get("dataset_id", f"{dataset_group}/{dataset_name}"),
                    "config_path": cfg_path,
                    "results_dir": Path(cfg["results_dir"]),
                    "models": tuple(cfg.get("models") or ()),
                }
            )
        return pd.DataFrame(rows)

    def _read_metric_csv(path: Path, *, dataset_key: str, dataset_id: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if df.empty:
            return df
        df["dataset_key"] = dataset_key
        if "dataset_id" not in df.columns or df["dataset_id"].isna().all():
            df["dataset_id"] = dataset_id
        df["source_file"] = path.name
        return df

    def _load_excluded_intervention_ids() -> set[str]:
        ids = set()
        for name, kwargs in EXCLUDED_MANIPULATION_PARAMETERS:
            payload = json.dumps(kwargs, sort_keys=True, default=str)
            digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
            ids.add(f"{name}_{digest}")
        return ids

    def load_generated_metric_tables(
        configs: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        main_frames, scib_frames, load_rows = [], [], []
        for row in configs.itertuples(index=False):
            evaluation_dir = row.results_dir / "evaluation"
            main_paths = sorted(
                p for p in evaluation_dir.glob("*_metrics.csv") if "_scib_metrics" not in p.stem
            )
            scib_paths = sorted(evaluation_dir.glob("*_scib_metrics.csv"))
            for path in main_paths:
                main_frames.append(
                    _read_metric_csv(path, dataset_key=row.dataset_key, dataset_id=row.dataset_id)
                )
            for path in scib_paths:
                scib_frames.append(
                    _read_metric_csv(path, dataset_key=row.dataset_key, dataset_id=row.dataset_id)
                )
            load_rows.append(
                {
                    "dataset_key": row.dataset_key,
                    "dataset_id": row.dataset_id,
                    "main_csvs": len(main_paths),
                    "scib_csvs": len(scib_paths),
                }
            )
        main_df = pd.concat(main_frames, ignore_index=True) if main_frames else pd.DataFrame()
        scib_df = pd.concat(scib_frames, ignore_index=True) if scib_frames else pd.DataFrame()
        excluded_intervention_ids = _load_excluded_intervention_ids()
        if (
            not main_df.empty
            and excluded_intervention_ids
            and "intervention_id" in main_df.columns
        ):
            main_df = main_df[
                ~main_df["intervention_id"].astype(str).isin(excluded_intervention_ids)
            ].reset_index(drop=True)
        if not main_df.empty and EXCLUDED_MODELS and "model" in main_df.columns:
            main_df = main_df[
                ~main_df["model"].astype(str).isin(EXCLUDED_MODELS)
            ].reset_index(drop=True)
        if not scib_df.empty and EXCLUDED_MODELS and "model" in scib_df.columns:
            scib_df = scib_df[
                ~scib_df["model"].astype(str).isin(EXCLUDED_MODELS)
            ].reset_index(drop=True)
        if not scib_df.empty and EXCLUDED_SCIB_METRICS:
            scib_df = scib_df[
                ~scib_df["metric_name"].astype(str).isin(EXCLUDED_SCIB_METRICS)
            ].reset_index(drop=True)
        return main_df, scib_df, pd.DataFrame(load_rows)

    configs = discover_generated_configs()
    main_raw, scib_raw, load_summary = load_generated_metric_tables(configs)
    return configs, load_summary, main_raw, scib_raw


@app.cell(hide_code=True)
def _(configs, load_summary, main_raw, mo, scib_raw):
    mo.vstack([
        mo.md(
            f"""
            ## Loaded generated outputs

            - Configs discovered: **{len(configs)}**
            - Main metric rows: **{len(main_raw):,}** · scIB metric rows: **{len(scib_raw):,}**
            """
        ),
        load_summary,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## scIB composites

    Sub-metrics are averaged within their category to a bio-conservation and a
    batch-correction score, with total = 0.6·bio + 0.4·batch. One row per
    dataset × model × composite.
    """)
    return


@app.cell
def _(SCIB_BIO_WEIGHT, SCIB_METRIC_CATEGORIES, SCIB_SPACE, pd, scib_raw):
    def compute_scib_composites(
        scib_long: pd.DataFrame, *, bio_weight: float = SCIB_BIO_WEIGHT
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if scib_long.empty:
            return pd.DataFrame(), pd.DataFrame()
        out = scib_long.copy()
        out["value_mean"] = pd.to_numeric(out["value_mean"], errors="coerce")
        out = out[out["value_mean"].notna()]
        out = out[out["metric_category"].isin(SCIB_METRIC_CATEGORIES)]
        out = out[out["space"].astype(str).eq(SCIB_SPACE)]
        if out.empty:
            return pd.DataFrame(), pd.DataFrame()

        # which sub-metrics fed each composite (for transparency)
        coverage = (
            out.groupby(["metric_category"], observed=True)["metric_name"]
            .agg(lambda s: ", ".join(sorted(set(s))))
            .reset_index(name="sub_metrics")
        )

        cat_mean = (
            out.groupby(["dataset_key", "dataset_id", "model", "metric_category"], observed=True)[
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
        if bio is not None and batch is not None:
            wide["scib_total"] = bio_weight * bio + (1.0 - bio_weight) * batch
        else:
            wide["scib_total"] = bio if batch is None else batch
        long = wide.melt(
            id_vars=["dataset_key", "dataset_id", "model"],
            value_vars=[c for c in ("scib_bio", "scib_batch", "scib_total") if c in wide.columns],
            var_name="composite",
            value_name="scib_value",
        ).dropna(subset=["scib_value"])
        return long, coverage

    scib_composites, scib_coverage = compute_scib_composites(scib_raw)
    return scib_composites, scib_coverage


@app.cell(hide_code=True)
def _(mo, scib_composites, scib_coverage):
    mo.vstack([
        mo.md(f"**scIB composite rows:** {len(scib_composites):,}"),
        mo.md("**Sub-metrics feeding each composite:**"),
        scib_coverage,
    ])
    return


@app.cell(hide_code=True)
def _(
    COMPOSITE_LABELS,
    SCIB_METRIC_CATEGORIES,
    SCIB_SPACE,
    mo,
    pd,
    plt,
    scib_composites,
    scib_raw,
    sns,
):
    scib_scores = pd.DataFrame()
    if not scib_raw.empty:
        scib_scores = scib_raw.copy()
        scib_scores["value_mean"] = pd.to_numeric(scib_scores["value_mean"], errors="coerce")
        scib_scores = scib_scores[
            scib_scores["metric_category"].isin(SCIB_METRIC_CATEGORIES)
            & scib_scores["space"].astype(str).eq(SCIB_SPACE)
        ]
        score_cols = [
            c
            for c in (
                "dataset_id",
                "dataset_key",
                "model",
                "metric_category",
                "metric_name",
                "value_mean",
            )
            if c in scib_scores.columns
        ]
        scib_scores = (
            scib_scores[score_cols]
            .dropna(subset=["value_mean"])
            .sort_values(["dataset_id", "model", "metric_category", "metric_name"])
            .reset_index(drop=True)
        )
        scib_scores["value_mean"] = scib_scores["value_mean"].round(4)

    composite_scores = pd.DataFrame()
    if not scib_composites.empty:
        composite_scores = (
            scib_composites.pivot_table(
                index=["dataset_id", "dataset_key", "model"],
                columns="composite",
                values="scib_value",
            )
            .reset_index()
            .sort_values(["dataset_id", "model"])
        )
        composite_cols = [c for c in COMPOSITE_LABELS if c in composite_scores.columns]
        composite_scores = composite_scores[
            ["dataset_id", "dataset_key", "model", *composite_cols]
        ].rename(columns=COMPOSITE_LABELS)
        value_cols = [COMPOSITE_LABELS[c] for c in composite_cols]
        composite_scores[value_cols] = composite_scores[value_cols].round(4)

    def _empty_fig(message: str):
        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.text(0.5, 0.5, message, ha="center", va="center")
        ax.set_axis_off()
        return fig

    def plot_raw_scib_scores(scores: pd.DataFrame):
        if scores.empty:
            return _empty_fig("No raw scIB scores available")

        plot_df = scores.copy()
        plot_df["dataset_model"] = (
            plot_df["dataset_id"].astype(str) + " · " + plot_df["model"].astype(str)
        )
        row_order = (
            plot_df[["dataset_id", "model", "dataset_model"]]
            .drop_duplicates()
            .sort_values(["dataset_id", "model"])["dataset_model"]
            .tolist()
        )
        categories = [
            c for c in SCIB_METRIC_CATEGORIES if c in set(plot_df["metric_category"])
        ]
        categories += sorted(set(plot_df["metric_category"]) - set(categories))
        width_ratios = [
            max(3.5, 0.5 * plot_df[plot_df["metric_category"].eq(cat)]["metric_name"].nunique())
            for cat in categories
        ]
        fig, axes = plt.subplots(
            1,
            len(categories),
            figsize=(sum(width_ratios) + 1.5, max(4, 0.32 * len(row_order) + 1.8)),
            width_ratios=width_ratios,
            squeeze=False,
            sharey=True,
        )
        for ax, category in zip(axes[0], categories):
            sub = plot_df[plot_df["metric_category"].eq(category)]
            mat = (
                sub.pivot_table(
                    index="dataset_model", columns="metric_name", values="value_mean"
                )
                .reindex(row_order)
                .sort_index(axis=1)
            )
            sns.heatmap(
                mat,
                ax=ax,
                cmap="viridis",
                vmin=0,
                vmax=1,
                linewidths=0.25,
                cbar=ax is axes[0][-1],
                cbar_kws={"label": "scIB score"},
            )
            ax.set_title(category.replace("_", " "))
            ax.set_xlabel("")
            ax.set_ylabel("dataset · model" if ax is axes[0][0] else "")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        fig.suptitle("Raw scIB sub-metric scores", y=1.02)
        fig.tight_layout()
        return fig

    def plot_composite_scores(composites: pd.DataFrame):
        if composites.empty:
            return _empty_fig("No scIB composite scores available")

        plot_df = composites.copy()
        plot_df["dataset_model"] = (
            plot_df["dataset_id"].astype(str) + " · " + plot_df["model"].astype(str)
        )
        composite_cols = [
            label for label in COMPOSITE_LABELS.values() if label in plot_df.columns
        ]
        if not composite_cols:
            return _empty_fig("No scIB composite score columns available")
        mat = (
            plot_df.sort_values(["dataset_id", "model"])
            .set_index("dataset_model")[composite_cols]
        )
        fig, ax = plt.subplots(
            figsize=(
                max(6, 1.8 * len(composite_cols)),
                max(4, 0.32 * len(mat) + 1.8),
            )
        )
        sns.heatmap(
            mat,
            ax=ax,
            cmap="viridis",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            linewidths=0.25,
            cbar_kws={"label": "composite score"},
        )
        ax.set_title("Computed scIB composite scores")
        ax.set_xlabel("")
        ax.set_ylabel("dataset · model")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        fig.tight_layout()
        return fig

    raw_scib_scores_fig = plot_raw_scib_scores(scib_scores)
    composite_scores_fig = plot_composite_scores(composite_scores)

    mo.vstack([
        mo.md(r"""
        ## scIB scores by dataset

        Raw scIB sub-metric scores used for the configured reference embedding space,
        followed by the dataset × model composites computed above.
        """),
        raw_scib_scores_fig,
        mo.md("**Raw scIB sub-metric scores:**"),
        scib_scores,
        composite_scores_fig,
        mo.md("**Computed scIB composite scores:**"),
        composite_scores,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Canonical project metrics

    Each project metric is collapsed to a canonical form (Leiden ARI → median over
    resolution; others → mean over any parameter rows), reduced per manipulation, then
    averaged across manipulations to a per-dataset robustness summary. The
    per-manipulation table is kept for the drill-down below.
    """)
    return


@app.cell
def _(
    CANONICAL_MEDIAN_REDUCE,
    CANONICAL_METRICS,
    EXCLUDE_INTERVENTIONS,
    INTERVENTION_COL,
    PROJECT_SPACE,
    main_raw,
    pd,
):
    def canonical_project_values(
        main_long: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if main_long.empty:
            return pd.DataFrame(), pd.DataFrame()
        df = main_long.copy()
        df = df[df["space"].astype(str).eq(PROJECT_SPACE)]
        df = df[df["metric_name"].isin(CANONICAL_METRICS)]
        if INTERVENTION_COL in df.columns and EXCLUDE_INTERVENTIONS:
            df = df[~df[INTERVENTION_COL].astype(str).isin(EXCLUDE_INTERVENTIONS)]
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        df["canonical"] = df["metric_name"].map(CANONICAL_METRICS)
        df["value_mean"] = pd.to_numeric(df["value_mean"], errors="coerce")
        df = df.dropna(subset=["value_mean"])

        has_manip = INTERVENTION_COL in df.columns
        grp = ["dataset_key", "dataset_id", "model", "canonical"]
        grp += [INTERVENTION_COL] if has_manip else []

        is_median = df["canonical"].isin(CANONICAL_MEDIAN_REDUCE)
        red_median = (
            df[is_median].groupby(grp, observed=True)["value_mean"].median()
            .reset_index(name="proj_value")
        )
        red_mean = (
            df[~is_median].groupby(grp, observed=True)["value_mean"].mean()
            .reset_index(name="proj_value")
        )
        per_manip = pd.concat([red_median, red_mean], ignore_index=True)
        if not has_manip:
            per_manip[INTERVENTION_COL] = "all"

        overall = (
            per_manip.groupby(["dataset_key", "dataset_id", "model", "canonical"], observed=True)[
                "proj_value"
            ]
            .mean()
            .reset_index()
        )
        return per_manip, overall

    project_per_manip, project_overall = canonical_project_values(main_raw)

    project_availability = (
        project_overall.groupby(["model", "canonical"], observed=True)["dataset_key"]
        .nunique()
        .reset_index(name="n_datasets")
        .sort_values(["model", "canonical"])
    )
    return project_availability, project_overall, project_per_manip


@app.cell(hide_code=True)
def _(mo, project_availability, project_overall, scib_composites):
    mo.vstack([
        mo.md(
            f"""
            ### Aggregated tables

            - Project per-dataset rows: **{len(project_overall):,}**
            - scIB composite rows: **{len(scib_composites):,}**
            """
        ),
        mo.md("**Project metric availability (datasets per model × metric):**"),
        project_availability,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correlations across datasets

    For each canonical metric × scIB composite: Spearman correlation over datasets,
    computed **per model** and **pooled** (all models, dataset×model as points).
    Pooled has more power but the points are not independent (each dataset recurs
    across models); treat it as a summary, with per-model consistency as the check.
    """)
    return


@app.cell
def _(
    CORRELATION_METHOD,
    MIN_DATASETS_FOR_CORRELATION,
    np,
    pd,
    project_overall,
    scib_composites,
    stats,
):
    def _corr(x, y, method):
        x = np.asarray(x, float); y = np.asarray(y, float)
        if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
            return np.nan, np.nan
        res = stats.spearmanr(x, y) if method == "spearman" else stats.pearsonr(x, y)
        return float(res.statistic), float(res.pvalue)

    def compute_correlations(
        overall, scib_comp, *, method, min_datasets
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if overall.empty or scib_comp.empty:
            return pd.DataFrame(), pd.DataFrame()
        rows, point_frames = [], []
        canon = sorted(overall["canonical"].unique())
        comps = sorted(scib_comp["composite"].unique())
        shared = sorted(set(overall["model"].astype(str)) & set(scib_comp["model"].astype(str)))
        model_sets = [("__pooled__", None)] + [(m, m) for m in shared]
        for mlabel, mfilt in model_sets:
            po = overall if mfilt is None else overall[overall["model"].astype(str).eq(mfilt)]
            sc = scib_comp if mfilt is None else scib_comp[scib_comp["model"].astype(str).eq(mfilt)]
            join_keys = ["dataset_key", "model"] if mfilt is None else ["dataset_key"]
            for cn in canon:
                pcn = po[po["canonical"].eq(cn)]
                for cp in comps:
                    scp = sc[sc["composite"].eq(cp)]
                    j = pcn.merge(scp, on=join_keys, how="inner", suffixes=("_p", "_s"))
                    j = j.dropna(subset=["proj_value", "scib_value"])
                    n = int(len(j))
                    if n < min_datasets:
                        r, p = np.nan, np.nan
                    else:
                        r, p = _corr(j["proj_value"], j["scib_value"], method)
                    rows.append({
                        "model": mlabel, "canonical": cn, "composite": cp, "method": method,
                        "r": r, "abs_r": abs(r) if np.isfinite(r) else np.nan,
                        "p_value": p, "n_datasets": n,
                    })
                    if n:
                        jj = j.copy()
                        jj["model_label"] = mlabel; jj["canonical"] = cn; jj["composite"] = cp
                        point_frames.append(jj)
        corr = pd.DataFrame(rows)
        points = pd.concat(point_frames, ignore_index=True) if point_frames else pd.DataFrame()
        return corr, points

    correlations, correlation_points = compute_correlations(
        project_overall, scib_composites,
        method=CORRELATION_METHOD, min_datasets=MIN_DATASETS_FOR_CORRELATION,
    )
    return correlation_points, correlations


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Headline — pooled correlation grid

    Canonical metric × scIB composite, pooled across models. Colour and annotation show
    signed correlation *r*. Positive under bio-conservation = the metric **captures**
    what scIB rewards; near-zero (especially under batch) = the metric is measuring
    something scIB's score does not.
    """)
    return


@app.cell
def _(COMPOSITE_LABELS, correlations, np, plt, sns):
    def _annot(r):
        if not np.isfinite(r):
            return ""
        return f"{r:.2f}"

    def plot_grid(corr, *, model_label, title):
        sub = corr[corr["model"].astype(str).eq(model_label)].copy()
        if sub.empty or sub["r"].notna().sum() == 0:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.text(0.5, 0.5, f"No correlations for {model_label}", ha="center", va="center")
            ax.set_axis_off()
            return fig
        rmat = sub.pivot(index="canonical", columns="composite", values="r")
        amat = sub.copy()
        amat["txt"] = [_annot(r) for r in amat["r"]]
        tmat = amat.pivot(index="canonical", columns="composite", values="txt")
        rmat = rmat.reindex(columns=[c for c in COMPOSITE_LABELS if c in rmat.columns])
        tmat = tmat.reindex(columns=rmat.columns)
        fig, ax = plt.subplots(figsize=(1.9 * len(rmat.columns) + 3, 0.7 * len(rmat) + 2.5))
        sns.heatmap(rmat, ax=ax, cmap="vlag", vmin=-1, vmax=1, center=0,
                    annot=tmat.values, fmt="", linewidths=0.4,
                    cbar_kws={"label": "Pearson r"})
        ax.set_xticklabels([COMPOSITE_LABELS.get(c, c) for c in rmat.columns], rotation=20, ha="right")
        ax.set_title(title)
        ax.set_xlabel(""); ax.set_ylabel("")
        fig.tight_layout()
        return fig

    pooled_grid_fig = plot_grid(correlations, model_label="__pooled__", title="")
    pooled_grid_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Relationship scatter — pooled, coloured by model

    The actual data behind the headline grid: each point is one dataset×model. Shows
    whether a correlation is a clean trend or driven by a few points, and whether
    models separate. Rows = canonical metrics, columns = scIB composites.
    """)
    return


@app.cell
def _(COMPOSITE_LABELS, correlation_points, correlations, np, plt, sns):
    def plot_scatter_grid(points):
        sub = points[points["model_label"].astype(str).eq("__pooled__")].copy()
        if sub.empty:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.text(0.5, 0.5, "No pooled points", ha="center", va="center")
            ax.set_axis_off()
            return fig
        canon = sorted(sub["canonical"].unique())
        comps = [c for c in COMPOSITE_LABELS if c in set(sub["composite"])]
        models = sorted(sub["model"].unique())
        palette = dict(zip(models, sns.color_palette("colorblind", n_colors=max(len(models), 3))))
        y_limits = {
            "DistCorr": (0, 1),
            "Global RNX": (0, 1),
            "Local RNX": (0, 1),
            "Leiden ARI": (-1, 1),
        }
        def _format_p_value(p_value):
            if not np.isfinite(p_value):
                return "NA"
            return f"{p_value:.1e}" if p_value < 0.001 else f"{p_value:.3f}"

        fig, axes = plt.subplots(len(canon), len(comps),
                                 figsize=(3.6 * len(comps), 3.0 * len(canon)), squeeze=False,
                                 sharex=True)
        for i, cn in enumerate(canon):
            for j, cp in enumerate(comps):
                ax = axes[i, j]
                d = sub[(sub["canonical"].eq(cn)) & (sub["composite"].eq(cp))]
                for model in models:
                    dm = d[d["model"].eq(model)]
                    ax.scatter(dm["scib_value"], dm["proj_value"], s=30, alpha=0.8,
                               color=palette[model], label=model if (i == 0 and j == 0) else None)
                if len(d) >= 2 and d["scib_value"].nunique() >= 2:
                    coef = np.polyfit(d["scib_value"], d["proj_value"], 1)
                    xs = np.linspace(d["scib_value"].min(), d["scib_value"].max(), 50)
                    ax.plot(xs, coef[0] * xs + coef[1], color="black", lw=1.2, alpha=0.6)
                stat = correlations[
                    correlations["model"].astype(str).eq("__pooled__")
                    & correlations["canonical"].eq(cn)
                    & correlations["composite"].eq(cp)
                ]
                if not stat.empty and np.isfinite(stat.iloc[0]["r"]):
                    ax.text(
                        0.04,
                        0.94,
                        f"r = {stat.iloc[0]['r']:.2f}\np = {_format_p_value(stat.iloc[0]['p_value'])}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8,
                        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                    )
                if i == 0:
                    ax.set_title(COMPOSITE_LABELS.get(cp, cp), fontsize=10)
                if j == 0:
                    ax.set_ylabel(cn, fontsize=9)
                if i == len(canon) - 1:
                    ax.set_xlabel("scIB composite score", fontsize=9)
                else:
                    ax.set_xlabel("")
                ax.set_xlim(0, 1)
                ax.set_ylim(*y_limits.get(cn, (0, 1)))
                ax.spines[["top", "right"]].set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center",
                       bbox_to_anchor=(0.5, 0.94), ncol=min(len(models), 6), frameon=False)
        fig.tight_layout()
        fig.subplots_adjust(top=0.86)
        return fig

    scatter_grid_fig = plot_scatter_grid(correlation_points)
    scatter_grid_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Manipulation × parameter correlations vs individual scIB metrics

    This keeps each manipulation parameter setting separate instead of averaging
    across parameter values. For every canonical project metric × manipulation ×
    parameter combination, we correlate the per-dataset/model project value against
    each individual scIB metric.
    """)
    return


@app.cell
def _(
    manipulation_parameter_scib_correlation_heatmaps,
    manipulation_parameter_scib_correlation_table,
    mo,
):
    mo.vstack([
        *manipulation_parameter_scib_correlation_heatmaps,
        mo.md("**Correlation table:**"),
        manipulation_parameter_scib_correlation_table,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Manipulation × parameter correlations vs individual scIB metrics — sceval only

    Same heatmap layout as above, but correlations are recomputed using only the
    `sceval` datasets.
    """)
    return


@app.cell
def _(mo, sceval_manipulation_parameter_scib_correlation_heatmaps):
    mo.vstack(sceval_manipulation_parameter_scib_correlation_heatmaps)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Relationship scatter — individual scIB metrics

    The composite view above averages scIB sub-metrics into bio, batch, and total
    scores. This expands the same pooled scatter check to each individual scIB metric:
    rows are canonical project metrics, columns are scIB metrics, and points are
    coloured by dataset family (`atlas` vs `sceval`).
    """)
    return


@app.cell
def _(
    CORRELATION_METHOD,
    MIN_DATASETS_FOR_CORRELATION,
    SCIB_METRIC_CATEGORIES,
    SCIB_SPACE,
    np,
    pd,
    project_overall,
    scib_raw,
    stats,
):
    def prepare_scib_metric_values(scib_long: pd.DataFrame) -> pd.DataFrame:
        if scib_long.empty:
            return pd.DataFrame()

        df = scib_long.copy()
        df["scib_value"] = pd.to_numeric(df["value_mean"], errors="coerce")
        df = df[
            df["metric_category"].isin(SCIB_METRIC_CATEGORIES)
            & df["space"].astype(str).eq(SCIB_SPACE)
        ].dropna(subset=["scib_value"])
        if df.empty:
            return pd.DataFrame()

        return (
            df.groupby(
                [
                    "dataset_key",
                    "dataset_id",
                    "model",
                    "metric_category",
                    "metric_name",
                ],
                observed=True,
            )["scib_value"]
            .mean()
            .reset_index()
            .rename(columns={"metric_name": "scib_metric"})
        )

    def _corr(x, y, method):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
            return np.nan, np.nan
        res = stats.spearmanr(x, y) if method == "spearman" else stats.pearsonr(x, y)
        return float(res.statistic), float(res.pvalue)

    def compute_scib_metric_correlations(
        overall: pd.DataFrame,
        scib_metrics: pd.DataFrame,
        *,
        method: str,
        min_datasets: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if overall.empty or scib_metrics.empty:
            return pd.DataFrame(), pd.DataFrame()

        rows, point_frames = [], []
        canon = sorted(overall["canonical"].unique())
        metrics = (
            scib_metrics[["metric_category", "scib_metric"]]
            .drop_duplicates()
            .sort_values(["metric_category", "scib_metric"])
        )
        shared = sorted(
            set(overall["model"].astype(str)) & set(scib_metrics["model"].astype(str))
        )
        model_sets = [("__pooled__", None)] + [(m, m) for m in shared]

        for mlabel, mfilt in model_sets:
            po = overall if mfilt is None else overall[overall["model"].astype(str).eq(mfilt)]
            sc = (
                scib_metrics
                if mfilt is None
                else scib_metrics[scib_metrics["model"].astype(str).eq(mfilt)]
            )
            join_keys = ["dataset_key", "model"] if mfilt is None else ["dataset_key"]
            for cn in canon:
                pcn = po[po["canonical"].eq(cn)]
                for metric in metrics.itertuples(index=False):
                    scp = sc[sc["scib_metric"].eq(metric.scib_metric)]
                    j = pcn.merge(scp, on=join_keys, how="inner", suffixes=("_p", "_s"))
                    j = j.dropna(subset=["proj_value", "scib_value"])
                    n = int(len(j))
                    if n < min_datasets:
                        r, p = np.nan, np.nan
                    else:
                        r, p = _corr(j["proj_value"], j["scib_value"], method)
                    rows.append(
                        {
                            "model": mlabel,
                            "canonical": cn,
                            "metric_category": metric.metric_category,
                            "scib_metric": metric.scib_metric,
                            "method": method,
                            "r": r,
                            "abs_r": abs(r) if np.isfinite(r) else np.nan,
                            "p_value": p,
                            "n_datasets": n,
                        }
                    )
                    if n:
                        jj = j.copy()
                        jj["model_label"] = mlabel
                        jj["canonical"] = cn
                        jj["metric_category"] = metric.metric_category
                        jj["scib_metric"] = metric.scib_metric
                        point_frames.append(jj)

        corr = pd.DataFrame(rows)
        points = pd.concat(point_frames, ignore_index=True) if point_frames else pd.DataFrame()
        return corr, points

    scib_metric_values = prepare_scib_metric_values(scib_raw)
    scib_metric_correlations, scib_metric_points = compute_scib_metric_correlations(
        project_overall,
        scib_metric_values,
        method=CORRELATION_METHOD,
        min_datasets=MIN_DATASETS_FOR_CORRELATION,
    )
    return (scib_metric_points,)


@app.cell
def _(CORRELATION_METHOD, mo, np, plt, scib_metric_points, sns, stats):
    def _empty_fig(message: str):
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, message, ha="center", va="center")
        ax.set_axis_off()
        return fig

    def _pretty_metric_name(name: str) -> str:
        return str(name).replace("_", " ")

    def _dataset_family(dataset_key) -> str:
        prefix = str(dataset_key).split("__", 1)[0]
        return {"atlases": "atlas", "sceval": "sceval"}.get(prefix, prefix)

    def _corr_label(d):
        n = int(len(d))
        if n < 3 or d["proj_value"].nunique() < 2 or d["scib_value"].nunique() < 2:
            return f"r = NA, n = {n}"
        res = (
            stats.spearmanr(d["proj_value"], d["scib_value"])
            if CORRELATION_METHOD == "spearman"
            else stats.pearsonr(d["proj_value"], d["scib_value"])
        )
        return f"r = {float(res.statistic):.2f}, n = {n}"

    def plot_scib_metric_scatter_grid(
        points, *, dataset_family_filter=None, title_suffix="all datasets"
    ):
        sub = points[points["model_label"].astype(str).eq("__pooled__")].copy()
        if sub.empty:
            return _empty_fig("No individual scIB metric correlation points")

        sub["dataset_family"] = sub["dataset_key"].map(_dataset_family)
        if dataset_family_filter is not None:
            sub = sub[sub["dataset_family"].isin(dataset_family_filter)]
        if sub.empty:
            return _empty_fig(f"No individual scIB metric points for {title_suffix}")

        canon = sorted(sub["canonical"].unique())
        metrics = (
            sub[["metric_category", "scib_metric"]]
            .drop_duplicates()
            .sort_values(["metric_category", "scib_metric"])
        )
        families = [
            family
            for family in ("atlas", "sceval")
            if family in set(sub["dataset_family"])
        ]
        families += sorted(set(sub["dataset_family"]) - set(families))
        palette = dict(
            zip(families, sns.color_palette("colorblind", n_colors=max(len(families), 2)))
        )
        fig, axes = plt.subplots(
            len(canon),
            len(metrics),
            figsize=(3.1 * len(metrics), 2.7 * len(canon)),
            squeeze=False,
            sharey=True,
        )
        for i, cn in enumerate(canon):
            for j, metric in enumerate(metrics.itertuples(index=False)):
                ax = axes[i, j]
                d = sub[
                    sub["canonical"].eq(cn)
                    & sub["metric_category"].eq(metric.metric_category)
                    & sub["scib_metric"].eq(metric.scib_metric)
                ]
                for family in families:
                    dfam = d[d["dataset_family"].eq(family)]
                    ax.scatter(
                        dfam["proj_value"],
                        dfam["scib_value"],
                        s=30,
                        alpha=0.8,
                        color=palette[family],
                        label=family if (i == 0 and j == 0) else None,
                    )
                if len(d) >= 2 and d["proj_value"].nunique() >= 2:
                    coef = np.polyfit(d["proj_value"], d["scib_value"], 1)
                    xs = np.linspace(d["proj_value"].min(), d["proj_value"].max(), 50)
                    ax.plot(xs, coef[0] * xs + coef[1], color="black", lw=1.1, alpha=0.6)

                stat_label = _corr_label(d)
                ax.text(
                    0.03,
                    0.95,
                    stat_label,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
                )
                if i == 0:
                    ax.set_title(
                        f"{_pretty_metric_name(metric.scib_metric)}\n"
                        f"{metric.metric_category.replace('_', ' ')}",
                        fontsize=9,
                    )
                if j == 0:
                    ax.set_ylabel(f"{cn}\nscIB score", fontsize=9)
                if i == len(canon) - 1:
                    ax.set_xlabel("project metric value", fontsize=9)
                else:
                    ax.set_xlabel("")
                ax.spines[["top", "right"]].set_visible(False)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                title="dataset family",
                loc="upper center",
                bbox_to_anchor=(0.5, 1.005),
                ncol=len(labels),
                frameon=False,
            )
        fig.suptitle(
            "Project metrics vs individual scIB metrics"
            f" · {title_suffix} · {CORRELATION_METHOD} correlation",
            y=1.01,
        )
        fig.tight_layout()
        return fig

    scib_metric_scatter_grid_fig = plot_scib_metric_scatter_grid(
        scib_metric_points, title_suffix="atlas + sceval"
    )
    scib_metric_scatter_grid_sceval_fig = plot_scib_metric_scatter_grid(
        scib_metric_points,
        dataset_family_filter=("sceval",),
        title_suffix="sceval only",
    )

    mo.vstack([
        scib_metric_scatter_grid_fig,
        scib_metric_scatter_grid_sceval_fig,
    ])
    return


@app.cell(hide_code=True)
def _():
    return


@app.cell
def _(
    CANONICAL_METRICS,
    CORRELATION_METHOD,
    EXCLUDE_INTERVENTIONS,
    INTERVENTION_COL,
    MIN_DATASETS_FOR_CORRELATION,
    PROJECT_SPACE,
    SCIB_METRIC_CATEGORIES,
    SCIB_SPACE,
    configs,
    hashlib,
    json,
    main_raw,
    np,
    pd,
    plt,
    scib_raw,
    sns,
    stats,
    yaml,
):
    def _format_param_value(value) -> str:
        if pd.isna(value):
            return ""
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.notna(numeric):
            return f"{float(numeric):g}"
        return str(value)

    def _expand_parameter_kwargs(kwargs: dict) -> list[dict]:
        if not kwargs:
            return [{}]
        keys = list(kwargs)
        values = [
            value if isinstance(value, list) else [value]
            for value in (kwargs[key] for key in keys)
        ]
        expanded = []
        for combo in __import__("itertools").product(*values):
            expanded.append(dict(zip(keys, combo, strict=True)))
        return expanded

    def _intervention_id_from_parameter_kwargs(name: str, kwargs: dict) -> str:
        payload = json.dumps(kwargs, sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
        return f"{name}_{digest}"

    def _parameter_label_from_kwargs(kwargs: dict) -> str:
        if not kwargs:
            return "default"
        return ", ".join(f"{key}={_format_param_value(kwargs[key])}" for key in sorted(kwargs))

    def build_intervention_parameter_lookup(config_table: pd.DataFrame) -> pd.DataFrame:
        if config_table.empty or "config_path" not in config_table.columns:
            return pd.DataFrame()

        rows = []
        for cfg_path in config_table["config_path"]:
            cfg = yaml.safe_load(cfg_path.read_text())
            for spec in cfg.get("interventions") or ():
                name = spec["name"]
                kwargs = dict(spec.get("kwargs") or {})
                kwargs.update(dict(spec.get("sweep") or {}))
                for expanded_kwargs in _expand_parameter_kwargs(kwargs):
                    rows.append(
                        {
                            "dataset_key": cfg_path.stem,
                            "intervention_id": _intervention_id_from_parameter_kwargs(
                                name, expanded_kwargs
                            ),
                            "intervention_parameter": _parameter_label_from_kwargs(
                                expanded_kwargs
                            ),
                        }
                    )
        return pd.DataFrame(rows).drop_duplicates()

    def prepare_project_parameter_values(main_long: pd.DataFrame) -> pd.DataFrame:
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
        df["proj_value"] = pd.to_numeric(df["value_mean"], errors="coerce")
        df = df.dropna(subset=["proj_value"])
        if INTERVENTION_COL not in df.columns:
            df[INTERVENTION_COL] = "all"

        intervention_lookup = build_intervention_parameter_lookup(configs)
        if (
            not intervention_lookup.empty
            and "intervention_id" in df.columns
            and "dataset_key" in df.columns
        ):
            df = df.merge(
                intervention_lookup,
                on=["dataset_key", "intervention_id"],
                how="left",
            )
        else:
            df["intervention_parameter"] = "default"

        df["intervention_parameter"] = df["intervention_parameter"].fillna("default")
        df["parameter"] = df["intervention_parameter"]

        group_cols = [
            "dataset_key",
            "dataset_id",
            "model",
            "canonical",
            INTERVENTION_COL,
            "parameter",
        ]
        return (
            df.groupby(group_cols, dropna=False, observed=True)["proj_value"]
            .mean()
            .reset_index()
            .rename(columns={INTERVENTION_COL: "manipulation"})
        )

    def prepare_scib_metric_values_for_parameter_corr(scib_long: pd.DataFrame) -> pd.DataFrame:
        if scib_long.empty:
            return pd.DataFrame()

        df = scib_long.copy()
        df["scib_value"] = pd.to_numeric(df["value_mean"], errors="coerce")
        df = df[
            df["metric_category"].isin(SCIB_METRIC_CATEGORIES)
            & df["space"].astype(str).eq(SCIB_SPACE)
        ].dropna(subset=["scib_value"])
        if df.empty:
            return pd.DataFrame()

        return (
            df.groupby(
                ["dataset_key", "dataset_id", "model", "metric_category", "metric_name"],
                observed=True,
            )["scib_value"]
            .mean()
            .reset_index()
            .rename(columns={"metric_name": "scib_metric"})
        )

    def _corr(x, y, method):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
            return np.nan, np.nan
        res = stats.spearmanr(x, y) if method == "spearman" else stats.pearsonr(x, y)
        return float(res.statistic), float(res.pvalue)

    def compute_parameter_correlations(project_params, scib_metrics) -> pd.DataFrame:
        if project_params.empty or scib_metrics.empty:
            return pd.DataFrame()

        rows = []
        project_keys = (
            project_params[["canonical", "manipulation", "parameter"]]
            .drop_duplicates()
            .sort_values(["canonical", "manipulation", "parameter"])
        )
        scib_keys = (
            scib_metrics[["metric_category", "scib_metric"]]
            .drop_duplicates()
            .sort_values(["metric_category", "scib_metric"])
        )
        for proj_key in project_keys.itertuples(index=False):
            psub = project_params[
                project_params["canonical"].eq(proj_key.canonical)
                & project_params["manipulation"].astype(str).eq(str(proj_key.manipulation))
                & project_params["parameter"].astype(str).eq(str(proj_key.parameter))
            ]
            for scib_key in scib_keys.itertuples(index=False):
                ssub = scib_metrics[
                    scib_metrics["metric_category"].eq(scib_key.metric_category)
                    & scib_metrics["scib_metric"].eq(scib_key.scib_metric)
                ]
                joined = psub.merge(
                    ssub,
                    on=["dataset_key", "model"],
                    how="inner",
                    suffixes=("_project", "_scib"),
                ).dropna(subset=["proj_value", "scib_value"])
                n = int(len(joined))
                if n < MIN_DATASETS_FOR_CORRELATION:
                    r, p_value = np.nan, np.nan
                else:
                    r, p_value = _corr(
                        joined["proj_value"], joined["scib_value"], CORRELATION_METHOD
                    )
                rows.append(
                    {
                        "canonical": proj_key.canonical,
                        "manipulation": proj_key.manipulation,
                        "parameter": proj_key.parameter,
                        "metric_category": scib_key.metric_category,
                        "scib_metric": scib_key.scib_metric,
                        "method": CORRELATION_METHOD,
                        "r": r,
                        "abs_r": abs(r) if np.isfinite(r) else np.nan,
                        "p_value": p_value,
                        "n_points": n,
                    }
                )
        return pd.DataFrame(rows)

    def plot_parameter_correlation_heatmaps(corr_table: pd.DataFrame):
        if corr_table.empty or corr_table["r"].notna().sum() == 0:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.text(
                0.5,
                0.5,
                "No manipulation-parameter correlations available",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
            return [fig]

        plot_df = corr_table.copy()
        plot_df["column_label"] = (
            plot_df["manipulation"].astype(str) + "\n" + plot_df["parameter"].astype(str)
        )
        plot_df["scib_metric_label"] = (
            plot_df["scib_metric"].astype(str).str.replace("_", " ", regex=False)
        )
        scib_rows = (
            plot_df[["metric_category", "scib_metric", "scib_metric_label"]]
            .drop_duplicates()
            .sort_values(["metric_category", "scib_metric"])
        )
        scib_order = scib_rows["scib_metric_label"]
        figures = []
        for canonical in sorted(plot_df["canonical"].dropna().unique()):
            sub = plot_df[plot_df["canonical"].eq(canonical)].copy()
            column_order = (
                sub[["manipulation", "parameter", "column_label"]]
                .drop_duplicates()
                .sort_values(["manipulation", "parameter"])["column_label"]
            )
            mat = sub.pivot_table(index="scib_metric_label", columns="column_label", values="r")
            mat = mat.reindex(index=scib_order, columns=column_order)

            fig, ax = plt.subplots(
                figsize=(
                    max(10, 0.48 * len(mat.columns) + 3.5),
                    max(6, 0.45 * len(mat) + 2.5),
                )
            )
            sns.heatmap(
                mat,
                ax=ax,
                cmap="vlag",
                vmin=-1,
                vmax=1,
                center=0,
                linewidths=0.15,
                cbar_kws={"label": f"{CORRELATION_METHOD} r"},
            )
            ax.set_title(
                f"{canonical}: manipulation × parameter correlations vs individual scIB metrics"
            )
            ax.set_xlabel("manipulation × parameter")
            ax.set_ylabel("")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=7)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
            category_start = 0
            category_labels = {
                "bio_conservation_metrics": "Bio-conservation metrics",
                "batch_correction_metrics": "Batch-correction metrics",
            }
            for category, rows in scib_rows.groupby("metric_category", sort=False, observed=True):
                category_size = len(rows)
                category_center = category_start + category_size / 2
                ax.text(
                    -0.36,
                    category_center,
                    category_labels.get(category, str(category).replace("_", " ")),
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=9,
                    fontweight="bold",
                )
                if category_start:
                    ax.axhline(category_start, color="black", lw=0.8)
                category_start += category_size
            fig.tight_layout()
            fig.subplots_adjust(left=0.28)
            figures.append(fig)
        return figures

    project_parameter_values = prepare_project_parameter_values(main_raw)
    scib_metric_values_for_parameters = prepare_scib_metric_values_for_parameter_corr(scib_raw)
    manipulation_parameter_scib_correlations = compute_parameter_correlations(
        project_parameter_values, scib_metric_values_for_parameters
    )
    manipulation_parameter_scib_correlation_heatmaps = plot_parameter_correlation_heatmaps(
        manipulation_parameter_scib_correlations
    )

    table = manipulation_parameter_scib_correlations.copy()
    if not table.empty:
        table = table.sort_values(
            ["canonical", "manipulation", "parameter", "metric_category", "scib_metric"]
        ).reset_index(drop=True)
        for col in ("r", "abs_r", "p_value"):
            table[col] = table[col].round(4)

    manipulation_parameter_scib_correlation_table = table
    return (
        manipulation_parameter_scib_correlation_heatmaps,
        manipulation_parameter_scib_correlation_table,
        project_parameter_values,
        scib_metric_values_for_parameters,
    )


@app.cell(hide_code=True)
def _():
    return


@app.cell
def _(
    CORRELATION_METHOD,
    MIN_DATASETS_FOR_CORRELATION,
    np,
    pd,
    plt,
    project_parameter_values,
    scib_metric_values_for_parameters,
    sns,
    stats,
):
    def _corr_for_sceval_parameter_heatmaps(x, y, method):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
            return np.nan, np.nan
        res = stats.spearmanr(x, y) if method == "spearman" else stats.pearsonr(x, y)
        return float(res.statistic), float(res.pvalue)

    def compute_sceval_parameter_correlations(project_params, scib_metrics) -> pd.DataFrame:
        project_params = project_params[
            project_params["dataset_key"].astype(str).str.startswith("sceval__")
        ].copy()
        scib_metrics = scib_metrics[
            scib_metrics["dataset_key"].astype(str).str.startswith("sceval__")
        ].copy()
        if project_params.empty or scib_metrics.empty:
            return pd.DataFrame()

        rows = []
        project_keys = (
            project_params[["canonical", "manipulation", "parameter"]]
            .drop_duplicates()
            .sort_values(["canonical", "manipulation", "parameter"])
        )
        scib_keys = (
            scib_metrics[["metric_category", "scib_metric"]]
            .drop_duplicates()
            .sort_values(["metric_category", "scib_metric"])
        )
        for proj_key in project_keys.itertuples(index=False):
            psub = project_params[
                project_params["canonical"].eq(proj_key.canonical)
                & project_params["manipulation"].astype(str).eq(str(proj_key.manipulation))
                & project_params["parameter"].astype(str).eq(str(proj_key.parameter))
            ]
            for scib_key in scib_keys.itertuples(index=False):
                ssub = scib_metrics[
                    scib_metrics["metric_category"].eq(scib_key.metric_category)
                    & scib_metrics["scib_metric"].eq(scib_key.scib_metric)
                ]
                joined = psub.merge(
                    ssub,
                    on=["dataset_key", "model"],
                    how="inner",
                    suffixes=("_project", "_scib"),
                ).dropna(subset=["proj_value", "scib_value"])
                n = int(len(joined))
                if n < MIN_DATASETS_FOR_CORRELATION:
                    r, p_value = np.nan, np.nan
                else:
                    r, p_value = _corr_for_sceval_parameter_heatmaps(
                        joined["proj_value"], joined["scib_value"], CORRELATION_METHOD
                    )
                rows.append(
                    {
                        "canonical": proj_key.canonical,
                        "manipulation": proj_key.manipulation,
                        "parameter": proj_key.parameter,
                        "metric_category": scib_key.metric_category,
                        "scib_metric": scib_key.scib_metric,
                        "method": CORRELATION_METHOD,
                        "r": r,
                        "abs_r": abs(r) if np.isfinite(r) else np.nan,
                        "p_value": p_value,
                        "n_points": n,
                    }
                )
        return pd.DataFrame(rows)

    def plot_sceval_parameter_correlation_heatmaps(corr_table: pd.DataFrame):
        if corr_table.empty or corr_table["r"].notna().sum() == 0:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.text(
                0.5,
                0.5,
                "No sceval manipulation-parameter correlations available",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
            return [fig]

        plot_df = corr_table.copy()
        plot_df["column_label"] = (
            plot_df["manipulation"].astype(str) + "\n" + plot_df["parameter"].astype(str)
        )
        plot_df["scib_metric_label"] = (
            plot_df["scib_metric"].astype(str).str.replace("_", " ", regex=False)
        )
        scib_rows = (
            plot_df[["metric_category", "scib_metric", "scib_metric_label"]]
            .drop_duplicates()
            .sort_values(["metric_category", "scib_metric"])
        )
        scib_order = scib_rows["scib_metric_label"]
        figures = []
        for canonical in sorted(plot_df["canonical"].dropna().unique()):
            sub = plot_df[plot_df["canonical"].eq(canonical)].copy()
            column_order = (
                sub[["manipulation", "parameter", "column_label"]]
                .drop_duplicates()
                .sort_values(["manipulation", "parameter"])["column_label"]
            )
            mat = sub.pivot_table(index="scib_metric_label", columns="column_label", values="r")
            mat = mat.reindex(index=scib_order, columns=column_order)

            fig, ax = plt.subplots(
                figsize=(
                    max(10, 0.48 * len(mat.columns) + 3.5),
                    max(6, 0.45 * len(mat) + 2.5),
                )
            )
            sns.heatmap(
                mat,
                ax=ax,
                cmap="vlag",
                vmin=-1,
                vmax=1,
                center=0,
                linewidths=0.15,
                cbar_kws={"label": f"{CORRELATION_METHOD} r"},
            )
            ax.set_title(
                f"{canonical}: sceval-only manipulation × parameter correlations vs individual scIB metrics"
            )
            ax.set_xlabel("manipulation × parameter")
            ax.set_ylabel("")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", fontsize=7)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
            category_start = 0
            category_labels = {
                "bio_conservation_metrics": "Bio-conservation metrics",
                "batch_correction_metrics": "Batch-correction metrics",
            }
            for category, rows in scib_rows.groupby("metric_category", sort=False, observed=True):
                category_size = len(rows)
                category_center = category_start + category_size / 2
                ax.text(
                    -0.36,
                    category_center,
                    category_labels.get(category, str(category).replace("_", " ")),
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=9,
                    fontweight="bold",
                )
                if category_start:
                    ax.axhline(category_start, color="black", lw=0.8)
                category_start += category_size
            fig.tight_layout()
            fig.subplots_adjust(left=0.28)
            figures.append(fig)
        return figures

    sceval_manipulation_parameter_scib_correlations = compute_sceval_parameter_correlations(
        project_parameter_values, scib_metric_values_for_parameters
    )
    sceval_manipulation_parameter_scib_correlation_heatmaps = (
        plot_sceval_parameter_correlation_heatmaps(
            sceval_manipulation_parameter_scib_correlations
        )
    )
    return (sceval_manipulation_parameter_scib_correlation_heatmaps,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Per-manipulation drill-down

    The headline averages over manipulations. This breaks the correlation out by
    manipulation (against scIB bio-conservation): which perturbations produce a
    preservation readout that tracks scIB, and which do not. Rows = metric ×
    manipulation, pooled across models.
    """)
    return


@app.cell
def _(
    CORRELATION_METHOD,
    INTERVENTION_COL,
    MIN_DATASETS_FOR_CORRELATION,
    np,
    pd,
    plt,
    project_per_manip,
    scib_composites,
    sns,
    stats,
):
    def per_manipulation_corr(per_manip, scib_comp, *, composite="scib_bio",
                              method=CORRELATION_METHOD, min_datasets=MIN_DATASETS_FOR_CORRELATION):
        if per_manip.empty or scib_comp.empty or INTERVENTION_COL not in per_manip.columns:
            return pd.DataFrame()
        sc = scib_comp[scib_comp["composite"].eq(composite)][["dataset_key", "model", "scib_value"]]
        rows = []
        canon = sorted(per_manip["canonical"].unique())
        manips = sorted(per_manip[INTERVENTION_COL].astype(str).unique())
        for cn in canon:
            for mp in manips:
                d = per_manip[(per_manip["canonical"].eq(cn))
                              & (per_manip[INTERVENTION_COL].astype(str).eq(mp))]
                j = d.merge(sc, on=["dataset_key", "model"], how="inner").dropna(
                    subset=["proj_value", "scib_value"])
                n = len(j)
                if n < min_datasets or j["proj_value"].nunique() < 2 or j["scib_value"].nunique() < 2:
                    r = np.nan
                else:
                    res = (stats.spearmanr(j["proj_value"], j["scib_value"]) if method == "spearman"
                           else stats.pearsonr(j["proj_value"], j["scib_value"]))
                    r = float(res.statistic)
                rows.append({"canonical": cn, "manipulation": mp, "r": r, "n_datasets": n})
        return pd.DataFrame(rows)

    def plot_per_manip(corr_mp):
        if corr_mp.empty or corr_mp["r"].notna().sum() == 0:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.text(0.5, 0.5, "No per-manipulation correlations", ha="center", va="center")
            ax.set_axis_off()
            return fig
        mat = corr_mp.pivot(index="canonical", columns="manipulation", values="r")
        fig, ax = plt.subplots(figsize=(1.6 * len(mat.columns) + 3, 0.7 * len(mat) + 2.5))
        sns.heatmap(mat, ax=ax, cmap="vlag", vmin=-1, vmax=1, center=0, annot=True, fmt=".2f",
                    linewidths=0.4, cbar_kws={"label": "pearson r"})
        ax.set_title("Per-manipulation correlation vs scIB bio-conservation (pooled)")
        ax.set_xlabel("manipulation"); ax.set_ylabel("project metric")
        ax.set_xticklabels([t.get_text().replace("_", " ") for t in ax.get_xticklabels()],
                           rotation=25, ha="right")
        fig.tight_layout()
        return fig

    per_manip_corr = per_manipulation_corr(project_per_manip, scib_composites)
    per_manip_fig = plot_per_manip(per_manip_corr)
    per_manip_fig
    return


if __name__ == "__main__":
    app.run()
