import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Metric stability vs dataset size

    **Section 1** — one manipulation (constants): metric vs subsample size, one line per atlas.

    **Section 2** — all manipulations: CV envelopes and step-diff **convergence n** (min *n* where
    subsequent changes stay below per-metric tolerance; `inf` if never).

    **Section 3** — snapshot at fixed *n* and seed: metric vs intervention strength, one line per atlas;
    rows = metrics, columns = manipulations.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import anndata as ad
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "lib"))
    from cellcount_sweep import (
        CV_THRESHOLD,
        DEFAULT_ATLASES,
        DEFAULT_CELL_COUNTS,
        discover_runs_from_processed,
    )

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.figsize"] = (12, 9)

    VAULT = Path("/vault/amoneim/scfm-controlled-manipulations")
    SIZE_SWEEP_ROOT = VAULT / "processed_size_sweep"

    # --- sweep grid ---
    SWEEP_ATLASES = DEFAULT_ATLASES
    CELL_COUNTS = DEFAULT_CELL_COUNTS
    SWEEP_MODEL = "scimilarity"
    METRIC_SPACE = "embedding"

    MANIPULATION_ORDER = (
        "downsample",
        "gene_dropout",
        "poisson_resampling",
        "local_smoothing",
        "gene_shuffle",
    )

    # --- single-manipulation stability figure (section 1) ---
    INTERVENTION_NAME = "poisson_resampling"
    # fraction for downsample; dropout_rate for gene_dropout; k for local_smoothing; etc.
    # Set to None to include all param values for the chosen intervention.
    INTERVENTION_PARAM_VALUE = "2"

    # --- snapshot grid (section 3) ---
    SNAPSHOT_CELL_COUNT = 10000
    SNAPSHOT_SEED = 0

    PARAM_KEYS = {
        "downsample": "fraction",
        "gene_dropout": "dropout_rate",
        "local_smoothing": "k",
        "poisson_resampling": "iterations",
        "gene_shuffle": "variant",
    }

    STABILITY_PANELS = (
        {
            "metric_name": "distcorr",
            "metric_category": "structure_metrics",
            "label": "Distance correlation",
            "ylim": (0.0, 1.0),
        },
        {
            "metric_name": "viscore_local_sp",
            "metric_category": "structure_metrics",
            "label": "Local SP",
            "ylim": (0.0, 1.0),
        },
        {
            "metric_name": "viscore_global_sp",
            "metric_category": "structure_metrics",
            "label": "Global SP",
            "ylim": (0.0, 1.0),
        },
        {
            "metric_name": "leiden_ari",
            "metric_category": "clustering_metrics",
            "label": "Leiden ARI",
            "ylim": (-1.0, 1.0),
            "avg_resolutions": True,
        },
    )

    # Max |m(n_{i+1}) - m(n_i)| allowed once converged (absolute step size; edit per metric).
    METRIC_CONVERGENCE_TOLERANCE = {
        "distcorr": 0.05,
        "viscore_local_sp": 0.05,
        "viscore_global_sp": 0.05,
        "leiden_ari": 0.05,
    }
    return (
        CELL_COUNTS,
        CV_THRESHOLD,
        INTERVENTION_NAME,
        INTERVENTION_PARAM_VALUE,
        MANIPULATION_ORDER,
        METRIC_CONVERGENCE_TOLERANCE,
        METRIC_SPACE,
        PARAM_KEYS,
        Path,
        SIZE_SWEEP_ROOT,
        SNAPSHOT_CELL_COUNT,
        SNAPSHOT_SEED,
        STABILITY_PANELS,
        SWEEP_ATLASES,
        SWEEP_MODEL,
        ad,
        discover_runs_from_processed,
        mo,
        np,
        pd,
        plt,
        sns,
    )


@app.cell
def _(
    CELL_COUNTS,
    INTERVENTION_NAME,
    INTERVENTION_PARAM_VALUE,
    PARAM_KEYS,
    Path,
    SIZE_SWEEP_ROOT,
    SWEEP_ATLASES,
    SWEEP_MODEL,
    ad,
    discover_runs_from_processed,
    pd,
):
    def _read_eval_csv(results_dir: Path, model: str) -> pd.DataFrame:
        path = results_dir / "evaluation" / f"{model}_metrics.csv"
        return pd.read_csv(path) if path.is_file() else pd.DataFrame()

    def _enrich_intervention_params(metrics: pd.DataFrame, runs: list[dict]) -> pd.DataFrame:
        manip_dirs = {Path(r["results_dir"]) / "manipulations" for r in runs}
        param_rows = []
        for iid in metrics["intervention_id"].drop_duplicates():
            h5ad_path = None
            for manip_dir in manip_dirs:
                candidate = manip_dir / f"{iid}.h5ad"
                if candidate.is_file():
                    h5ad_path = candidate
                    break
            if h5ad_path is None:
                continue
            name = metrics.loc[metrics["intervention_id"] == iid, "intervention_name"].iloc[0]
            adata = ad.read_h5ad(h5ad_path, backed="r")
            params = adata.uns.get("scfm_intervention", {}).get(name, {})
            key = PARAM_KEYS.get(name)
            value = params.get(key) if key else None
            param_rows.append(
                {
                    "intervention_id": iid,
                    "intervention_name": name,
                    "param_key": key,
                    "param_value": value,
                }
            )
            adata.file.close()
        if not param_rows:
            return metrics
        return metrics.merge(
            pd.DataFrame(param_rows), on=["intervention_id", "intervention_name"], how="left"
        )

    def load_size_sweep_metrics(
        *,
        sweep_root,
        atlases,
        cell_counts,
        model,
        intervention_name=None,
        param_value=None,
        exclude_reference: bool = True,
    ) -> tuple[pd.DataFrame, list[dict]]:
        allowed_atlases = set(atlases)
        allowed_counts = {int(n) for n in cell_counts}
        runs = discover_runs_from_processed(sweep_root)
        runs = [
            r
            for r in runs
            if r.get("atlas") in allowed_atlases and int(r["cell_count"]) in allowed_counts
        ]

        frames = []
        for run in runs:
            results_dir = Path(run["results_dir"])
            df = _read_eval_csv(results_dir, model)
            if df.empty:
                continue
            df["atlas"] = run["atlas"]
            df["cell_count"] = int(run["cell_count"])
            df["sweep_seed"] = int(run["seed"])
            df["run_id"] = run.get("run_id", "")
            frames.append(df)

        if not frames:
            return pd.DataFrame(), runs

        metrics = pd.concat(frames, ignore_index=True)
        metrics = _enrich_intervention_params(metrics, runs)
        if exclude_reference:
            metrics = metrics[metrics["intervention_name"] != "reference"].copy()
        if intervention_name is not None:
            metrics = metrics[metrics["intervention_name"] == intervention_name].copy()
        if param_value is not None and "param_value" in metrics.columns:
            if isinstance(param_value, (int, float)):
                metrics = metrics[
                    pd.to_numeric(metrics["param_value"], errors="coerce") == float(param_value)
                ]
            else:
                metrics = metrics[metrics["param_value"].astype(str) == str(param_value)]
        metrics = metrics[metrics["model"].astype(str) == model].copy()
        return metrics, runs

    metrics_df, _sweep_runs = load_size_sweep_metrics(
        sweep_root=SIZE_SWEEP_ROOT,
        atlases=SWEEP_ATLASES,
        cell_counts=CELL_COUNTS,
        model=SWEEP_MODEL,
        intervention_name=INTERVENTION_NAME,
        param_value=INTERVENTION_PARAM_VALUE,
    )

    metrics_all_df, sweep_runs = load_size_sweep_metrics(
        sweep_root=SIZE_SWEEP_ROOT,
        atlases=SWEEP_ATLASES,
        cell_counts=CELL_COUNTS,
        model=SWEEP_MODEL,
        intervention_name=None,
        param_value=None,
    )
    return metrics_all_df, metrics_df


@app.cell
def _(METRIC_SPACE, STABILITY_PANELS, np, pd):
    def per_seed_scalar(
        metrics_df: pd.DataFrame,
        panel: dict,
        *,
        extra_group_cols: tuple[str, ...] = (),
    ) -> pd.DataFrame:
        """One scalar per seed for a metric (optionally grouped by intervention/param)."""
        sub = metrics_df[
            (metrics_df["metric_category"] == panel["metric_category"])
            & (metrics_df["metric_name"] == panel["metric_name"])
            & (metrics_df["space"] == METRIC_SPACE)
        ].copy()
        if sub.empty:
            return sub

        sub["value_mean"] = pd.to_numeric(sub["value_mean"], errors="coerce")
        sub = sub.dropna(subset=["value_mean"])
        if sub.empty:
            return sub

        seed_cols = ["atlas", "cell_count", "sweep_seed", *extra_group_cols]
        seed_cols = [c for c in seed_cols if c in sub.columns]
        if panel.get("avg_resolutions"):
            out = (
                sub.groupby(seed_cols, observed=True)["value_mean"]
                .mean()
                .reset_index()
                .assign(metric_name=panel["metric_name"])
            )
            return out

        return (
            sub.groupby(seed_cols, observed=True)["value_mean"]
            .first()
            .reset_index()
            .assign(metric_name=panel["metric_name"])
        )

    def aggregate_metric_curves(
        metrics_df: pd.DataFrame,
        *,
        extra_group_cols: tuple[str, ...] = (),
    ) -> pd.DataFrame:
        """Mean ± std across seeds for each atlas × cell_count × metric (± intervention)."""
        parts = []
        group_cols = ["atlas", "cell_count", *extra_group_cols]
        for panel in STABILITY_PANELS:
            per_seed = per_seed_scalar(metrics_df, panel, extra_group_cols=extra_group_cols)
            if per_seed.empty:
                continue
            agg = (
                per_seed.groupby(group_cols, observed=True)["value_mean"]
                .agg(
                    mean_across_seeds="mean",
                    std_across_seeds=lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                    n_seeds="count",
                )
                .reset_index()
            )
            agg["cv_across_seeds"] = np.where(
                agg["mean_across_seeds"].abs() > 0,
                agg["std_across_seeds"] / agg["mean_across_seeds"].abs(),
                np.nan,
            )
            agg["metric_name"] = panel["metric_name"]
            agg["panel_label"] = panel["label"]
            parts.append(agg)
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)

    def find_convergence_n(curve: pd.DataFrame, tolerance: float) -> float:
        """Smallest cell_count n such that all subsequent steps stay within tolerance.

        Uses absolute step size |m(n_{j+1}) - m(n_j)|. Returns ``inf`` if never converged.
        """
        df = curve.sort_values("cell_count").reset_index(drop=True)
        if len(df) < 2:
            return float("inf")
        counts = df["cell_count"].astype(int).tolist()
        means = df["mean_across_seeds"].astype(float).tolist()
        steps_ok = [
            abs(means[i + 1] - means[i]) <= tolerance for i in range(len(means) - 1)
        ]
        for k in range(len(counts)):
            if all(steps_ok[j] for j in range(k, len(steps_ok))):
                return float(counts[k])
        return float("inf")

    def compute_convergence_n_table(
        curves: pd.DataFrame,
        *,
        tolerances: dict[str, float],
    ) -> pd.DataFrame:
        id_cols = [
            c
            for c in (
                "atlas",
                "intervention_name",
                "param_value",
                "metric_name",
                "panel_label",
            )
            if c in curves.columns
        ]
        rows = []
        for keys, grp in curves.groupby(id_cols, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(id_cols, keys))
            metric_name = row.get("metric_name", "")
            tol = tolerances.get(metric_name, 0.05)
            row["step_tolerance"] = tol
            row["n_converged"] = find_convergence_n(grp, tol)
            rows.append(row)
        return pd.DataFrame(rows)

    def cv_percentile_by_cell_count(df: pd.DataFrame, quantile: float) -> pd.DataFrame:
        rows = []
        for cell_count, grp in df.groupby("cell_count", observed=True):
            vals = grp["cv_across_seeds"].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "cell_count": int(cell_count),
                    "cv_quantile": float(vals.quantile(quantile)) if len(vals) else np.nan,
                }
            )
        return pd.DataFrame(rows).sort_values("cell_count")

    def ordered_atlases(atlases: pd.Series, atlas_order: tuple[str, ...]) -> list[str]:
        known = [a for a in atlas_order if a in set(atlases)]
        extra = sorted(set(atlases) - set(atlas_order))
        return [*known, *extra]

    def sort_param_values(values) -> list:
        vals = list(pd.Series(values).dropna().unique())
        try:
            return sorted(vals, key=lambda v: float(v))
        except (TypeError, ValueError):
            return sorted(vals, key=str)

    return (
        aggregate_metric_curves,
        compute_convergence_n_table,
        cv_percentile_by_cell_count,
        ordered_atlases,
        per_seed_scalar,
        sort_param_values,
    )


@app.cell
def _(
    PARAM_KEYS,
    cv_percentile_by_cell_count,
    np,
    ordered_atlases,
    pd,
    per_seed_scalar,
    plt,
    sns,
    sort_param_values,
):
    def plot_stability_panels(
        curves: pd.DataFrame,
        *,
        atlas_order: tuple[str, ...],
        panels: tuple[dict, ...],
        title: str,
    ):
        if curves.empty:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(0.5, 0.5, "No data loaded", ha="center", va="center")
            ax.set_axis_off()
            return fig

        n_panels = len(panels)
        ncols = 2
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.5 * ncols, 4.2 * nrows),
            sharex=True,
            squeeze=False,
        )

        present_atlases = ordered_atlases(curves["atlas"].dropna(), atlas_order)
        palette = sns.color_palette("colorblind", n_colors=max(len(present_atlases), 3))
        atlas_colors = dict(zip(present_atlases, palette, strict=False))

        for ax, panel in zip(axes.flatten(), panels):
            panel_df = curves[curves["metric_name"] == panel["metric_name"]]
            for _atlas in present_atlases:
                sub = panel_df[panel_df["atlas"] == _atlas].sort_values("cell_count")
                if sub.empty:
                    continue
                ax.errorbar(
                    sub["cell_count"].astype(float),
                    sub["mean_across_seeds"].astype(float),
                    yerr=sub["std_across_seeds"].astype(float),
                    label=_atlas,
                    color=atlas_colors[_atlas],
                    marker="o",
                    capsize=3,
                    capthick=0.8,
                    elinewidth=0.8,
                    linewidth=2,
                    markersize=5,
                )
            ax.set_title(panel["label"])
            ax.set_ylim(panel["ylim"])

        for ax in axes.flatten()[n_panels:]:
            ax.set_visible(False)
        for ax in axes[-1, :]:
            if ax.get_visible():
                ax.set_xlabel("Subsample size")
        for ax in axes[:, 0]:
            if ax.get_visible():
                ax.set_ylabel("Metric value (mean across seeds)")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                title="Atlas",
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=min(len(present_atlases), 6),
                frameon=False,
            )
        fig.suptitle(title, y=1.06, fontsize=13)
        fig.tight_layout()
        return fig

    def plot_cv_envelope_by_atlas(
        curves: pd.DataFrame,
        *,
        atlas_order: tuple[str, ...],
        cell_counts: tuple[int, ...],
        cv_threshold: float,
        title: str,
    ):
        sub = curves.replace([np.inf, -np.inf], np.nan).dropna(subset=["cv_across_seeds"])
        if sub.empty:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(0.5, 0.5, "No CV data", ha="center", va="center")
            ax.set_axis_off()
            return fig

        atlas_list = ordered_atlases(sub["atlas"].dropna(), atlas_order)
        curve_cols = [
            c for c in ("intervention_name", "param_value", "metric_name") if c in sub.columns
        ]

        ncols = 3
        nrows = int(np.ceil(len(atlas_list) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.2 * ncols, 3.4 * nrows),
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        for idx, _atlas in enumerate(atlas_list):
            ax = axes[idx // ncols][idx % ncols]
            panel = sub[sub["atlas"] == _atlas]
            if panel.empty:
                ax.set_visible(False)
                continue

            for _, curve in panel.groupby(curve_cols, observed=True):
                curve = curve.sort_values("cell_count")
                if len(curve) < 2:
                    continue
                ax.plot(
                    curve["cell_count"],
                    curve["cv_across_seeds"],
                    color="#888888",
                    alpha=0.12,
                    linewidth=0.7,
                    zorder=1,
                )

            p10 = cv_percentile_by_cell_count(panel, 0.10)
            p90 = cv_percentile_by_cell_count(panel, 0.90)
            if not p90.empty:
                ax.fill_between(
                    p10["cell_count"],
                    p10["cv_quantile"],
                    p90["cv_quantile"],
                    color="#D55E00",
                    alpha=0.18,
                    linewidth=0,
                    zorder=2,
                )
                ax.plot(
                    p90["cell_count"],
                    p90["cv_quantile"],
                    color="#D55E00",
                    linewidth=2.4,
                    zorder=3,
                )

            ax.axhline(cv_threshold, color="gray", linestyle="--", linewidth=1.2, zorder=4)
            ymax = max(cv_threshold * 4, float(p90["cv_quantile"].max()) * 1.6, 0.12)
            ax.set_ylim(0, min(ymax, 1.0))
            ax.set_title(_atlas.replace("_", " "))

        for ax in axes[-1, :]:
            ax.set_xlabel("Subsample size")
        for row in axes:
            row[0].set_ylabel("CV across seeds")
        for j in range(len(atlas_list), nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)

        fig.suptitle(title, y=1.04, fontsize=12)
        fig.tight_layout()
        return fig

    def plot_convergence_n_by_atlas(table: pd.DataFrame, *, atlas_order: tuple[str, ...]):
        finite = table[np.isfinite(table["n_converged"])].copy()
        if finite.empty:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(0.5, 0.5, "No converged curves (all inf)", ha="center", va="center")
            ax.set_axis_off()
            return fig

        atlas_list = ordered_atlases(finite["atlas"].dropna(), atlas_order)
        finite["atlas"] = pd.Categorical(finite["atlas"], categories=atlas_list, ordered=True)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.boxplot(data=finite, x="atlas", y="n_converged", order=atlas_list, ax=ax)
        ax.set_xlabel("Atlas")
        ax.set_ylabel("Convergence n (cells)")
        ax.set_title("Step-diff convergence")
        plt.xticks(rotation=25, ha="right")
        fig.tight_layout()
        return fig

    def plot_manipulation_snapshot_grid(
        metrics_df: pd.DataFrame,
        *,
        cell_count: int,
        sweep_seed: int,
        manipulation_order: tuple[str, ...],
        panels: tuple[dict, ...],
        atlas_order: tuple[str, ...],
        model: str,
    ):
        snap = metrics_df[
            (metrics_df["cell_count"] == int(cell_count))
            & (metrics_df["sweep_seed"] == int(sweep_seed))
        ].copy()
        if snap.empty:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(0.5, 0.5, "No snapshot data", ha="center", va="center")
            ax.set_axis_off()
            return fig

        present_interventions = [
            m for m in manipulation_order if m in set(snap["intervention_name"])
        ]
        present_atlases = ordered_atlases(snap["atlas"].dropna(), atlas_order)
        palette = sns.color_palette("colorblind", n_colors=max(len(present_atlases), 3))
        atlas_colors = dict(zip(present_atlases, palette, strict=False))

        nrows = len(panels)
        ncols = len(present_interventions)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.8 * ncols, 3.4 * nrows),
            sharex="col",
            squeeze=False,
        )

        for row_idx, panel in enumerate(panels):
            for col_idx, intervention in enumerate(present_interventions):
                ax = axes[row_idx, col_idx]
                panel_snap = snap[snap["intervention_name"] == intervention]
                seed_rows = per_seed_scalar(
                    panel_snap,
                    panel,
                    extra_group_cols=("param_value",),
                )
                if seed_rows.empty:
                    ax.set_visible(False)
                    continue

                param_key = PARAM_KEYS.get(intervention, "param")
                x_vals = sort_param_values(seed_rows["param_value"])
                x_positions = list(range(len(x_vals)))

                for _atlas in present_atlases:
                    sub = seed_rows[seed_rows["atlas"] == _atlas]
                    if sub.empty:
                        continue
                    sub = sub.copy()
                    sub["param_value"] = sub["param_value"].astype(str)
                    y_by_param = {
                        str(k): float(v)
                        for k, v in zip(sub["param_value"], sub["value_mean"], strict=False)
                    }
                    y = [y_by_param.get(str(v), np.nan) for v in x_vals]
                    ax.plot(
                        x_positions,
                        y,
                        marker="o",
                        linewidth=2,
                        color=atlas_colors[_atlas],
                        label=_atlas if row_idx == 0 and col_idx == 0 else None,
                    )

                ax.set_ylim(panel["ylim"])
                if row_idx == 0:
                    ax.set_title(intervention.replace("_", " "))
                if col_idx == 0:
                    ax.set_ylabel(panel["label"])
                if row_idx == nrows - 1:
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels([str(v) for v in x_vals], rotation=30, ha="right")
                    ax.set_xlabel(param_key)
                else:
                    ax.tick_params(labelbottom=False)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                title="Atlas",
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=min(len(present_atlases), 6),
                frameon=False,
            )
        fig.suptitle(
            f"n = {cell_count:,} · seed = {sweep_seed} · {model}",
            y=1.05,
            fontsize=13,
        )
        fig.tight_layout()
        return fig

    return (
        plot_convergence_n_by_atlas,
        plot_manipulation_snapshot_grid,
        plot_stability_panels,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Metric vs subsample size (one manipulation)
    """)
    return


@app.cell
def _(
    INTERVENTION_NAME,
    INTERVENTION_PARAM_VALUE,
    STABILITY_PANELS,
    SWEEP_ATLASES,
    SWEEP_MODEL,
    aggregate_metric_curves,
    metrics_df,
    plot_stability_panels,
    plt,
):
    _param_note = (
        f", {INTERVENTION_PARAM_VALUE}" if INTERVENTION_PARAM_VALUE is not None else ""
    )
    _curves = aggregate_metric_curves(metrics_df)
    plot_stability_panels(
        _curves,
        atlas_order=SWEEP_ATLASES,
        panels=STABILITY_PANELS,
        title=f"{INTERVENTION_NAME}{_param_note} · {SWEEP_MODEL}",
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Stability aggregation (all manipulations)

    Faint lines = CV across seeds for every atlas × manipulation × param × metric curve.
    Bold band = p10–p90 of CV across those curves at each subsample size.

    **Convergence n** = smallest subsample size where every later step changes the metric by
    at most `METRIC_CONVERGENCE_TOLERANCE[metric]` (absolute); `inf` if that never happens.
    """)
    return


@app.cell
def _(
    METRIC_CONVERGENCE_TOLERANCE,
    SWEEP_ATLASES,
    aggregate_metric_curves,
    compute_convergence_n_table,
    metrics_all_df,
    plot_convergence_n_by_atlas,
    plt,
):
    stability_curves_all = aggregate_metric_curves(
        metrics_all_df,
        extra_group_cols=("intervention_name", "param_value"),
    )
    convergence_n_table = compute_convergence_n_table(
        stability_curves_all,
        tolerances=METRIC_CONVERGENCE_TOLERANCE,
    )


    plot_convergence_n_by_atlas(convergence_n_table, atlas_order=SWEEP_ATLASES)
    plt.gcf()
    return convergence_n_table, stability_curves_all


@app.cell
def _(convergence_n_table):
    convergence_n_table
    return


@app.cell
def _(stability_curves_all):
    stability_curves_all
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### CV vs subsample size (one line per atlas)

    Median cross-seed CV across all manipulation × param × metric curves at each subsample size.
    """)
    return


@app.cell
def _(
    CV_THRESHOLD,
    SWEEP_ATLASES,
    np,
    ordered_atlases,
    plt,
    stability_curves_all,
):
    sub = stability_curves_all.copy()
    sub["cv_across_seeds"] = sub["cv_across_seeds"].replace([np.inf, -np.inf], np.nan)
    sub = sub.dropna(subset=["cv_across_seeds"])

    fig, ax = plt.subplots(figsize=(12, 8))
    for _atlas in ordered_atlases(sub["atlas"].dropna(), SWEEP_ATLASES):
        cv_by_n_stats = (
            sub[sub["atlas"] == _atlas]
            .groupby("cell_count", observed=True)["cv_across_seeds"]
            .agg(median="median", std="std")
            .reset_index()
            .sort_values("cell_count")
        )
        if cv_by_n_stats.empty:
            continue
        ax.errorbar(
            cv_by_n_stats["cell_count"],
            cv_by_n_stats["median"],
            yerr=cv_by_n_stats["std"],
            marker="o",
            linewidth=2,
            label=_atlas,
            capsize=4,
        )

    ax.axhline(CV_THRESHOLD, color="gray", linestyle="--", label=f"CV = {CV_THRESHOLD}")
    ax.set_xlabel("Subsample size")
    ax.set_ylabel("Median CV across seeds")
    ax.set_title("CV across subsample sizes (error bars: ±1 SD)")
    ax.legend(title="Atlas")
    ax.grid(True, axis="both", linestyle="--", alpha=0.5)
    fig.tight_layout()
    plt.gcf()
    return


@app.cell
def _(convergence_n_table, mo, np):
    if convergence_n_table.empty:
        mo.md("No convergence n values computed.")
    else:
        _finite = convergence_n_table[np.isfinite(convergence_n_table["n_converged"])]
        _n_inf = len(convergence_n_table) - len(_finite)
        summary = (
            _finite.groupby("atlas", observed=True)["n_converged"]
            .agg(n_converged_median="median", n_converged_p90=lambda s: s.quantile(0.9), n_curves="count")
            .reset_index()
            .sort_values("n_converged_p90")
        )
        mo.vstack(
            [
                mo.md(
                    f"**Per-atlas convergence summary** (median / p90 of `n_converged`; "
                    f"{_n_inf} / {len(convergence_n_table)} curves never converged → `inf`)."
                ),
                summary,
                convergence_n_table.sort_values(["atlas", "metric_name", "n_converged"]),
            ]
        )
    return


@app.cell(hide_code=True)
def _(SNAPSHOT_CELL_COUNT, SNAPSHOT_SEED, mo):
    mo.md(rf"""
    ### Snapshot at n = {SNAPSHOT_CELL_COUNT}, seed = {SNAPSHOT_SEED}

    Rows = metrics, columns = manipulations. Each panel: x = intervention parameter,
    y = metric value, one line per atlas.
    """)
    return


@app.cell
def _(
    MANIPULATION_ORDER,
    SNAPSHOT_CELL_COUNT,
    SNAPSHOT_SEED,
    STABILITY_PANELS,
    SWEEP_ATLASES,
    SWEEP_MODEL,
    metrics_all_df,
    plot_manipulation_snapshot_grid,
    plt,
):
    plot_manipulation_snapshot_grid(
        metrics_all_df,
        cell_count=SNAPSHOT_CELL_COUNT,
        sweep_seed=SNAPSHOT_SEED,
        manipulation_order=MANIPULATION_ORDER,
        panels=STABILITY_PANELS,
        atlas_order=SWEEP_ATLASES,
        model=SWEEP_MODEL,
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Convergence fits — how each curve plateaus

    Relative-change thresholds fail here: metrics sit at different absolute levels
    across manipulations, so a fixed % change means different things and near-ceiling
    curves look trivially flat. Instead each metric-vs-*n* curve is fit to a power-law
    approach to an asymptote,

    $$m(n) = m_\infty - c\,n^{-\alpha},$$

    and summarized by **scale-free** descriptors that are comparable across curves at
    different levels:

    - **$m_\infty$** — the value the metric converges to (asymptote).
    - **gap@$n_\max$** — distance of the largest available $n$ from $m_\infty$, as a
      fraction of the curve's own range. Small = already well-converged.
    - **$n_\text{settled}$** — smallest *observed* $n$ already within a tolerance band of
      $m_\infty$ (data-grounded, no extrapolation).
    - **$R^2$** — whether the curve follows a clean convergence shape at all; a low
      $R^2$ means "no clean plateau," which is itself the answer.
    - **$\alpha$** — convergence rate (reported, but noisier than the above; lead with
      gap@$n_\max$ and $n_\text{settled}$).
    """)
    return


@app.cell
def _(STABILITY_PANELS, np):
    from scipy.optimize import curve_fit

    def _conv_model(n, m_inf, c, alpha):
        return m_inf - c * np.power(n, -alpha)

    def fit_convergence(counts, values, *, settle_frac: float = 0.05) -> dict:
        """Fit m(n)=m_inf - c n^-alpha; return asymptote, rate, fit quality, and
        data-grounded settled readouts (no extrapolation beyond observed n)."""
        counts = np.asarray(counts, dtype=float)
        values = np.asarray(values, dtype=float)
        ok = np.isfinite(counts) & np.isfinite(values)
        counts, values = counts[ok], values[ok]
        order = np.argsort(counts)
        counts, values = counts[order], values[order]

        base = dict(m_inf=np.nan, alpha=np.nan, c=np.nan, r2=np.nan,
                    gap_at_nmax=np.nan, n_settled=np.nan, status="too_few")
        if len(counts) < 4:
            return base
        rng = float(values.max() - values.min())
        if rng < 1e-3:
            return dict(m_inf=float(values.mean()), alpha=np.nan, c=0.0, r2=1.0,
                        gap_at_nmax=0.0, n_settled=float(counts.min()), status="flat")

        p0 = [values[-1], values[-1] - values[0], 0.5]
        lo = [values.min() - 2 * rng, -10 * abs(rng) - 1, 0.01]
        hi = [values.max() + 2 * rng, 10 * abs(rng) + 1, 5.0]
        try:
            popt, _ = curve_fit(_conv_model, counts, values, p0=p0,
                                bounds=(lo, hi), maxfev=40000)
        except Exception:
            return {**base, "status": "fit_fail"}

        m_inf, c, alpha = popt
        pred = _conv_model(counts, *popt)
        ss_res = float(((values - pred) ** 2).sum())
        ss_tot = float(((values - values.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        gap_at_nmax = abs(m_inf - values[-1]) / rng
        within = np.abs(values - m_inf) <= settle_frac * rng
        n_settled = float(counts[within][0]) if within.any() else np.nan
        return dict(m_inf=float(m_inf), alpha=float(alpha), c=float(c),
                    r2=float(r2) if np.isfinite(r2) else np.nan,
                    gap_at_nmax=float(gap_at_nmax), n_settled=n_settled,
                    status="ok" if within.any() else "not_settled")

    def compute_convergence_table(curves, *, settle_frac: float = 0.05):
        """Per (atlas × intervention × param × metric) convergence fit over cell_count."""
        import pandas as pd
        label_by_metric = {p["metric_name"]: p["label"] for p in STABILITY_PANELS}
        id_cols = [c for c in ("atlas", "intervention_name", "param_value", "metric_name")
                   if c in curves.columns]
        rows = []
        for keys, grp in curves.groupby(id_cols, observed=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            g = grp.sort_values("cell_count")
            fit = fit_convergence(g["cell_count"].to_numpy(),
                                  g["mean_across_seeds"].to_numpy(),
                                  settle_frac=settle_frac)
            row = dict(zip(id_cols, keys))
            row["panel_label"] = label_by_metric.get(row.get("metric_name"), row.get("metric_name"))
            row.update(fit)
            rows.append(row)
        return pd.DataFrame(rows)

    return compute_convergence_table, fit_convergence


@app.cell
def _(compute_convergence_table, stability_curves_all):
    convergence_table = compute_convergence_table(stability_curves_all)
    convergence_table
    return (convergence_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Fitted convergence curves

    Points = observed seed-mean metric; dashed = fitted $m_\infty - c\,n^{-\alpha}$;
    horizontal line = $m_\infty$; shaded band = $\pm$ settle tolerance. Rows = metrics,
    one line/fit per atlas (single chosen manipulation to stay legible).
    """)
    return


@app.cell
def _(
    INTERVENTION_NAME,
    STABILITY_PANELS,
    SWEEP_ATLASES,
    fit_convergence,
    np,
    ordered_atlases,
    plt,
    sns,
    stability_curves_all,
):
    def plot_convergence_fits(curves, *, intervention_name, atlas_order,
                              panels, settle_frac=0.05):
        sub = curves.copy()
        if "intervention_name" in sub.columns:
            sub = sub[sub["intervention_name"] == intervention_name]
        if sub.empty:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(0.5, 0.5, "No data for chosen intervention", ha="center", va="center")
            ax.set_axis_off()
            return fig

        present = ordered_atlases(sub["atlas"].dropna(), atlas_order)
        palette = sns.color_palette("colorblind", n_colors=max(len(present), 3))
        colors = dict(zip(present, palette, strict=False))

        n_panels = len(panels)
        ncols = 2
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.2 * nrows),
                                 squeeze=False)
        grid = np.logspace  # for smooth fit line

        for ax, panel in zip(axes.flatten(), panels):
            pdf = sub[sub["metric_name"] == panel["metric_name"]]
            for _atlas in present:
                s = pdf[pdf["atlas"] == _atlas].sort_values("cell_count")
                # collapse any param duplicates to one curve per (atlas, cell_count)
                s = s.groupby("cell_count", observed=True)["mean_across_seeds"].mean().reset_index()
                if len(s) < 2:
                    continue
                xc = s["cell_count"].to_numpy(float)
                yc = s["mean_across_seeds"].to_numpy(float)
                ax.scatter(xc, yc, s=28, color=colors[_atlas], zorder=3, label=_atlas)
                fit = fit_convergence(xc, yc, settle_frac=settle_frac)
                if np.isfinite(fit["m_inf"]):
                    if np.isfinite(fit["alpha"]) and np.isfinite(fit["c"]):
                        xs = grid(np.log10(xc.min()), np.log10(xc.max()), 100)
                        ys = fit["m_inf"] - fit["c"] * xs ** (-fit["alpha"])
                        ax.plot(xs, ys, ls="--", lw=1.4, color=colors[_atlas], alpha=0.9, zorder=2)
                    ax.axhline(fit["m_inf"], color=colors[_atlas], lw=0.8, alpha=0.35, zorder=1)
            ax.set_xscale("log")
            ax.set_title(panel["label"])
            ax.set_ylim(panel["ylim"])
            ax.set_xlabel("Subsample size")
            ax.set_ylabel("Metric value")
            ax.spines[["top", "right"]].set_visible(False)

        for ax in axes.flatten()[n_panels:]:
            ax.set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, title="Atlas", loc="upper center",
                       bbox_to_anchor=(0.5, 1.03), ncol=min(len(present), 6), frameon=False)
        fig.suptitle(f"Convergence fits · {intervention_name}", y=1.06, fontsize=13)
        fig.tight_layout()
        return fig

    convergence_fits_fig = plot_convergence_fits(
        stability_curves_all,
        intervention_name=INTERVENTION_NAME,
        atlas_order=SWEEP_ATLASES,
        panels=STABILITY_PANELS,
    )
    convergence_fits_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Convergence summary — gap@$n_\max$ heatmap

    Per metric × atlas, the gap between the largest available $n$ and the fitted
    asymptote (fraction of range; low/green = well-converged). Annotated with $R^2$;
    a low $R^2$ means the power-law plateau model does not describe that curve and the
    gap should not be over-read.
    """)
    return


@app.cell
def _(SWEEP_ATLASES, convergence_table, np, ordered_atlases, plt):
    def plot_convergence_heatmap(table, *, atlas_order):
        import pandas as pd
        if table.empty:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.text(0.5, 0.5, "No convergence table", ha="center", va="center")
            ax.set_axis_off()
            return fig
        # median gap across interventions/params, per metric × atlas
        agg = (table.groupby(["panel_label", "atlas"], observed=True)
                    .agg(gap=("gap_at_nmax", "median"), r2=("r2", "median"))
                    .reset_index())
        atlases = ordered_atlases(agg["atlas"].dropna(), atlas_order)
        metrics = list(agg["panel_label"].dropna().unique())
        gap_mat = np.full((len(metrics), len(atlases)), np.nan)
        r2_mat = np.full_like(gap_mat, np.nan)
        for _, r in agg.iterrows():
            i = metrics.index(r["panel_label"]); j = atlases.index(r["atlas"])
            gap_mat[i, j] = r["gap"]; r2_mat[i, j] = r["r2"]

        fig, ax = plt.subplots(figsize=(1.6 * len(atlases) + 3, 0.9 * len(metrics) + 2))
        im = ax.imshow(gap_mat, cmap="RdYlGn_r", aspect="auto", vmin=0,
                       vmax=np.nanmax(gap_mat) if np.isfinite(np.nanmax(gap_mat)) else 1)
        ax.set_xticks(range(len(atlases)))
        ax.set_xticklabels([a.replace("_", " ") for a in atlases], rotation=25, ha="right")
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(metrics)
        for i in range(len(metrics)):
            for j in range(len(atlases)):
                if np.isfinite(gap_mat[i, j]):
                    txt = f"{gap_mat[i, j]:.2f}\n$R^2$={r2_mat[i, j]:.2f}" if np.isfinite(r2_mat[i, j]) else f"{gap_mat[i, j]:.2f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                            color="white" if gap_mat[i, j] > np.nanmax(gap_mat) * 0.6 else "black")
        ax.set_title("gap@$n_{max}$ (fraction of range) · median over manipulations")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="gap@$n_{max}$")
        fig.tight_layout()
        return fig

    convergence_heatmap_fig = plot_convergence_heatmap(convergence_table, atlas_order=SWEEP_ATLASES)
    convergence_heatmap_fig
    return


if __name__ == "__main__":
    app.run()
