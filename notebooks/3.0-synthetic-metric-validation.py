import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Confounder characterization of structure-preservation metrics

    **Goal.** Establish, defensibly, how stable the finalized metrics are to nuisance
    properties of a dataset — so a value can (or cannot) be compared across datasets
    that differ in size, cluster count, dimensionality, or composition.

    **Metrics characterized:** distance correlation (`distcorr`), ViScore Local SP
    (`Sl`) and Global SP (`Sg`), and clustering agreement (adjusted Rand index, ARI,
    over a Leiden resolution sweep). $R_{NX}$ curves are shown alongside.

    **What makes this defensible (vs. a naive fixed-knob sweep):**

    1. **True distortion is held constant, not the distortion knob.** At a fixed dropout
       rate the *amount of perturbation actually injected* varies wildly across
       conditions (e.g. ~5× higher under heavy class imbalance). We instead define a
       metric-independent **injected-distortion anchor**
       $D=\langle\lVert \mathrm{man}-\mathrm{ref}\rVert^2\rangle / \langle\lVert\mathrm{ref}-\bar{x}\rVert^2\rangle$
       and **calibrate the knob per condition** so $D$ is constant. Residual metric
       movement then reflects the nuisance property, not a change in how much was done
       to the data. A dedicated panel verifies $D$ is held constant.
    2. **Two structurally different distortions** — coordinate dropout (axis-aligned,
       sparsifying) and additive Gaussian noise (isotropic) — both calibrated to the
       same $D$. A conclusion that holds under both is distortion-agnostic; one that
       flips is not.
    3. **ARI uses the *adjusted* Rand index over a Leiden resolution sweep**, reported
       as a band, since a single resolution is arbitrary and unadjusted agreement is
       cluster-count biased.
    4. **Enough seeds for inference.** Movement is judged against its own seed-level
       sampling noise; the summary table reports a slope vs. $\log$(property) with a
       bootstrap CI, so "flat" and "sloped" are statistical statements.

    **The anchor's honest scope.** $D$ holds the *magnitude of injected perturbation*
    constant — a well-defined, metric-independent quantity. It does **not** claim to
    hold "true neighbourhood distortion" constant; no single such quantity exists (that
    multiplicity is the project's premise). The claim is therefore precise: at matched
    injected perturbation, does the metric read differently because of the nuisance
    property?
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import pdist, squareform
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import adjusted_rand_score
    import leidenalg
    import igraph
    import viscore

    # ---- design parameters ----
    TARGET_D = 0.30               # injected-distortion anchor target (constant across all conditions)
    SEEDS = (0, 1, 2, 3, 4, 5)    # seeds per condition (inference on movement)

    LEIDEN_RESOLUTIONS = (0.25, 0.5, 1.0, 2.0)
    LEIDEN_K = 15

    # ---- synthetic blob baseline (non-swept properties held here) ----
    BASE_N = 1000
    BASE_DIM = 128
    BASE_CLUSTERS = 10
    BASE_SEP = 8.0

    # ---- confounder grids ----
    SWEEP_N_VALUES = (300, 600, 1000, 1500, 2000)
    SWEEP_CLUSTER_VALUES = (3, 10, 30, 75, 200)
    SWEEP_DIM_VALUES = (64, 128, 256, 512, 1024)
    SWEEP_IMBALANCE_VALUES = (1, 3, 10, 30, 100)   # dominant-cluster weight vs rest

    # ---- distortion mechanisms (calibrated to TARGET_D) ----
    DISTORTIONS = ("dropout", "gaussian")

    METRIC_COLUMNS = ("distcorr", "Sl", "Sg", "ari")
    METRIC_LABELS = {
        "distcorr": "Distance corr.",
        "Sl": "ViScore Local SP",
        "Sg": "ViScore Global SP",
        "ari": "Leiden ARI (median over res.)",
    }
    METRIC_COLORS = {
        "distcorr": "#55A868",
        "Sl": "#C44E52",
        "Sg": "#4C72B0",
        "ari": "#8172B3",
    }
    DISTORTION_STYLE = {"dropout": "-", "gaussian": "--"}
    return (
        BASE_CLUSTERS,
        BASE_DIM,
        BASE_N,
        BASE_SEP,
        DISTORTIONS,
        DISTORTION_STYLE,
        LEIDEN_K,
        LEIDEN_RESOLUTIONS,
        METRIC_COLORS,
        METRIC_COLUMNS,
        METRIC_LABELS,
        NearestNeighbors,
        SEEDS,
        SWEEP_CLUSTER_VALUES,
        SWEEP_DIM_VALUES,
        SWEEP_IMBALANCE_VALUES,
        SWEEP_N_VALUES,
        TARGET_D,
        adjusted_rand_score,
        igraph,
        leidenalg,
        np,
        pd,
        pdist,
        plt,
        squareform,
        viscore,
    )


@app.cell
def _(BASE_CLUSTERS, BASE_DIM, BASE_N, BASE_SEP, np):
    def make_blobs(*, n=BASE_N, dim=BASE_DIM, n_clusters=BASE_CLUSTERS,
                   sep=BASE_SEP, sizes=None, seed=0):
        """Isotropic Gaussian clusters (unit within-cluster sd). Returns (coords, labels)."""
        rng = np.random.default_rng(seed)
        centres = rng.normal(0.0, sep, size=(n_clusters, dim))
        if sizes is None:
            labels = rng.integers(0, n_clusters, size=n)
        else:
            p = np.asarray(sizes, dtype=np.float64); p = p / p.sum()
            labels = rng.choice(n_clusters, size=n, p=p)
        coords = centres[labels] + rng.normal(0.0, 1.0, size=(n, dim))
        return coords, labels

    def distort_dropout(emb, knob, *, seed=0):
        """Coordinate dropout: zero each coordinate independently w.p. `knob`."""
        if knob <= 0.0:
            return emb.copy()
        rng = np.random.default_rng(seed)
        out = emb.copy()
        out[rng.random(out.shape) < knob] = 0.0
        return out

    def distort_gaussian(emb, knob, *, seed=0):
        """Additive isotropic Gaussian noise with sd `knob`."""
        if knob <= 0.0:
            return emb.copy()
        return emb + np.random.default_rng(seed).normal(0.0, knob, size=emb.shape)

    def injected_distortion(ref, man):
        """Anchor D: mean squared displacement normalised by mean squared spread.
        Metric-independent measure of how much perturbation was injected."""
        disp = ((man - ref) ** 2).sum(1).mean()
        spread = ((ref - ref.mean(0)) ** 2).sum(1).mean()
        return float(disp / spread) if spread > 0 else 0.0

    return distort_dropout, distort_gaussian, injected_distortion, make_blobs


@app.cell
def _(injected_distortion, np):
    def calibrate_knob(ref, distort_fn, target_d, *, lo, hi, probe_seeds=(0, 1, 2),
                       tol=0.015, iters=30):
        """Bisect the distortion knob so the injected-distortion anchor D ≈ target_d
        (mean over probe seeds). Returns the calibrated knob value."""
        def mean_d(knob):
            return np.mean([injected_distortion(ref, distort_fn(ref, knob, seed=7000 + s))
                            for s in probe_seeds])
        _lo, _hi = lo, hi
        knob = 0.5 * (_lo + _hi)
        for _ in range(iters):
            knob = 0.5 * (_lo + _hi)
            d = mean_d(knob)
            if abs(d - target_d) < tol:
                return knob
            if d < target_d:
                _lo = knob
            else:
                _hi = knob
        return knob

    return (calibrate_knob,)


@app.cell
def _(
    LEIDEN_K,
    LEIDEN_RESOLUTIONS,
    NearestNeighbors,
    adjusted_rand_score,
    igraph,
    leidenalg,
    np,
    pdist,
    squareform,
    viscore,
):
    # ---- metric core -------------------------------------------------------------
    def _distcorr(ref, man):
        """Székely distance correlation via O(n^2) double-centering."""
        d_ref = squareform(pdist(ref.astype(np.float64)))
        d_man = squareform(pdist(man.astype(np.float64)))
        def _c(d):
            return d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + d.mean()
        a, b = _c(d_ref), _c(d_man)
        n = a.shape[0]
        na = np.sqrt((a * a).sum()) / n
        nb = np.sqrt((b * b).sum()) / n
        den = na * nb
        return float((a * b).sum() / (n * n) / den) if den > 0 else 0.0

    def _leiden_labels(X, resolution, *, k=LEIDEN_K, seed=0):
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        _, idx = nn.kneighbors(X)
        n = X.shape[0]
        edges = {(min(i, int(j)), max(i, int(j)))
                 for i in range(n) for j in idx[i, 1:]}
        g = igraph.Graph(n=n, edges=list(edges))
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution, seed=seed, n_iterations=2)
        return np.asarray(part.membership)

    def _ari_curve(ref, man, resolutions=LEIDEN_RESOLUTIONS):
        """Adjusted Rand between ref-clustering and man-clustering at each resolution."""
        out = []
        for res in resolutions:
            lr = _leiden_labels(ref, res, seed=0)
            lm = _leiden_labels(man, res, seed=0)
            out.append(adjusted_rand_score(lr, lm))
        return np.asarray(out, dtype=np.float64)

    def score_pair(ref, man, *, want_ari=True):
        """All metrics for one ref/man pair. Returns scalars + the RNX and ARI curves."""
        s = viscore.score(ref, man)
        rnx = np.asarray(s["RNX"], dtype=np.float64)
        ari_curve = _ari_curve(ref, man) if want_ari else np.array([np.nan])
        return {
            "distcorr": _distcorr(ref, man),
            "Sl": float(s["Sl"]),
            "Sg": float(s["Sg"]),
            "ari": float(np.median(ari_curve)),
            "rnx": rnx,
            "ari_curve": ari_curve,
        }

    return (score_pair,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Validation

    identity → distcorr/Sl/Sg/ARI = 1 · random → distcorr/Sl/Sg ≈ 0, ARI ≈ 0. Also
    confirms the calibrator hits the target $D$ for both distortion types.
    """)
    return


@app.cell
def _(
    TARGET_D,
    calibrate_knob,
    distort_dropout,
    distort_gaussian,
    injected_distortion,
    make_blobs,
    np,
    pd,
    score_pair,
):
    def _validation():
        ref, _ = make_blobs(n=500, seed=0)
        rng = np.random.default_rng(1)
        rows = []
        for name, man in (("identity", ref.copy()),
                          ("random", rng.normal(0, 1, size=ref.shape))):
            m = score_pair(ref, man)
            rows.append({"case": name, "distcorr": round(m["distcorr"], 3),
                         "Sl": round(m["Sl"], 3), "Sg": round(m["Sg"], 3),
                         "ari": round(m["ari"], 3)})
        # calibration check
        p = calibrate_knob(ref, distort_dropout, TARGET_D, lo=0.0, hi=0.95)
        s = calibrate_knob(ref, distort_gaussian, TARGET_D, lo=0.0, hi=15.0)
        rows.append({"case": f"calib dropout p={p:.3f}",
                     "distcorr": round(injected_distortion(ref, distort_dropout(ref, p, seed=99)), 3),
                     "Sl": np.nan, "Sg": np.nan, "ari": np.nan})
        rows.append({"case": f"calib gaussian σ={s:.3f}",
                     "distcorr": round(injected_distortion(ref, distort_gaussian(ref, s, seed=99)), 3),
                     "Sl": np.nan, "Sg": np.nan, "ari": np.nan})
        return pd.DataFrame(rows)

    validation_table = _validation()
    validation_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Run the calibrated sweep

    For every (confounder, value, distortion, seed): build the reference, **calibrate
    the distortion knob to the target $D$ for that exact condition**, apply it, and
    score. Records the realized $D$ so the next panel can verify it was held constant.

    *Runtime: ~240 conditions; the n-sweep dominates (O(n²) distcorr + Leiden). Several
    minutes. Reduce `SEEDS` or grids to speed up.*
    """)
    return


@app.cell
def _(
    BASE_N,
    DISTORTIONS,
    SEEDS,
    SWEEP_CLUSTER_VALUES,
    SWEEP_DIM_VALUES,
    SWEEP_IMBALANCE_VALUES,
    SWEEP_N_VALUES,
    TARGET_D,
    calibrate_knob,
    distort_dropout,
    distort_gaussian,
    injected_distortion,
    make_blobs,
    pd,
    score_pair,
):
    _DISTORT_FN = {"dropout": distort_dropout, "gaussian": distort_gaussian}
    _KNOB_HI = {"dropout": 0.95, "gaussian": 20.0}

    def _build(param_name, value, seed):
        v = value
        if param_name == "n":
            return make_blobs(n=int(v), seed=seed)
        if param_name == "clusters":
            return make_blobs(n=BASE_N, n_clusters=int(v), seed=seed)
        if param_name == "dim":
            return make_blobs(n=BASE_N, dim=int(v), seed=seed)
        if param_name == "imbalance":
            return make_blobs(n=BASE_N, n_clusters=10, sizes=[int(v)] + [1] * 9, seed=seed)
        raise ValueError(param_name)

    def _run_condition(param_name, value, distortion, rows, curve_store):
        fn = _DISTORT_FN[distortion]
        for seed in SEEDS:
            ref, _lab = _build(param_name, value, seed)
            # calibrate knob to target D for THIS condition (per seed ref, mild cost)
            knob = calibrate_knob(ref, fn, TARGET_D, lo=0.0, hi=_KNOB_HI[distortion])
            man = fn(ref, knob, seed=5000 + seed)
            realized_d = injected_distortion(ref, man)
            m = score_pair(ref, man)
            rows.append({
                "param_name": param_name, "param_value": float(value),
                "distortion": distortion, "seed": seed, "knob": knob,
                "realized_D": realized_d,
                "distcorr": m["distcorr"], "Sl": m["Sl"], "Sg": m["Sg"], "ari": m["ari"],
            })
            # keep one RNX + ARI curve per (condition, distortion) at seed 0 for figures
            if seed == 0:
                curve_store.append({
                    "param_name": param_name, "param_value": float(value),
                    "distortion": distortion, "rnx": m["rnx"], "ari_curve": m["ari_curve"],
                })

    def run_sweep():
        rows, curves = [], []
        grids = (
            ("n", SWEEP_N_VALUES),
            ("clusters", SWEEP_CLUSTER_VALUES),
            ("dim", SWEEP_DIM_VALUES),
            ("imbalance", SWEEP_IMBALANCE_VALUES),
        )
        for param_name, values in grids:
            for value in values:
                for distortion in DISTORTIONS:
                    _run_condition(param_name, value, distortion, rows, curves)
        return pd.DataFrame(rows), curves

    sweep_raw, sweep_curves = run_sweep()
    sweep_summary = (
        sweep_raw
        .groupby(["param_name", "param_value", "distortion"], as_index=False)
        .agg(distcorr=("distcorr", "mean"), distcorr_sd=("distcorr", "std"),
             Sl=("Sl", "mean"), Sl_sd=("Sl", "std"),
             Sg=("Sg", "mean"), Sg_sd=("Sg", "std"),
             ari=("ari", "mean"), ari_sd=("ari", "std"),
             realized_D=("realized_D", "mean"), realized_D_sd=("realized_D", "std"))
        .sort_values(["param_name", "distortion", "param_value"])
    )
    print(f"{len(sweep_raw)} scored pairs · target D={TARGET_D}")
    sweep_summary
    return sweep_curves, sweep_raw, sweep_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### 0 · Calibration check — was true distortion held constant?

    The whole design rests on this. Realized $D$ should sit on the target line across
    **every** condition and both distortion types. Flat here = the confounder panels
    below compare conditions at matched injected perturbation. (Contrast: at a fixed
    dropout rate, $D$ under heavy imbalance is ~5× the base level.)
    """)
    return


@app.cell
def _(DISTORTION_STYLE, TARGET_D, plt, sweep_summary):
    def _calibration_figure(summary):
        panels = ("n", "clusters", "dim", "imbalance")
        titles = {"n": "dataset size $n$", "clusters": "cluster count",
                  "dim": "embedding dim", "imbalance": "size imbalance"}
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
        for ax, pname in zip(axes, panels):
            for dist, ls in DISTORTION_STYLE.items():
                part = summary[(summary["param_name"] == pname)
                               & (summary["distortion"] == dist)].sort_values("param_value")
                ax.errorbar(part["param_value"], part["realized_D"], yerr=part["realized_D_sd"],
                            ls=ls, marker="o", ms=4, lw=1.5, capsize=2, label=dist)
            ax.axhline(TARGET_D, color="grey", ls=":", lw=1.2)
            ax.set_xscale("log"); ax.set_xlabel(titles[pname])
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("realized $D$")
        axes[0].set_ylim(0, TARGET_D * 2)
        axes[0].legend(frameon=False, fontsize=8, title=f"target $D$={TARGET_D}")
        fig.suptitle("Calibration check — injected distortion held constant across all conditions", y=1.02)
        fig.tight_layout()
        return fig

    calibration_fig = _calibration_figure(sweep_summary)
    calibration_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### 1 · Confounder panels — metric movement at constant injected distortion

    Rows = metrics, columns = confounders. Solid = dropout, dashed = Gaussian noise;
    bands = ±1 SD over seeds. With $D$ held constant (panel 0), any slope is the
    metric responding to the nuisance property, not to a change in distortion. A
    conclusion that agrees across both distortion types is distortion-agnostic.
    """)
    return


@app.cell
def _(
    DISTORTION_STYLE,
    METRIC_COLORS,
    METRIC_COLUMNS,
    METRIC_LABELS,
    plt,
    sweep_summary,
):
    def _confounder_grid(summary):
        confounders = (("n", "dataset size $n$"), ("clusters", "cluster count"),
                       ("dim", "embedding dim"), ("imbalance", "size imbalance"))
        nrow, ncol = len(METRIC_COLUMNS), len(confounders)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
        for i, metric in enumerate(METRIC_COLUMNS):
            for j, (pname, ptitle) in enumerate(confounders):
                ax = axes[i, j]
                for dist, ls in DISTORTION_STYLE.items():
                    part = summary[(summary["param_name"] == pname)
                                   & (summary["distortion"] == dist)].sort_values("param_value")
                    m, sd = part[metric], part[f"{metric}_sd"]
                    ax.fill_between(part["param_value"], m - sd, m + sd,
                                    color=METRIC_COLORS[metric], alpha=0.12, linewidth=0)
                    ax.plot(part["param_value"], m, ls, color=METRIC_COLORS[metric],
                            marker="o", ms=3.5, lw=1.6, label=dist)
                ax.set_xscale("log")
                ax.spines[["top", "right"]].set_visible(False)
                if i == 0:
                    ax.set_title(ptitle, fontsize=11)
                if i == nrow - 1:
                    ax.set_xlabel(ptitle)
                if j == 0:
                    ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
                if i == 0 and j == ncol - 1:
                    ax.legend(frameon=False, fontsize=8)
        fig.suptitle("Confounder sweeps — metric value vs nuisance property at constant injected distortion",
                     y=1.0, fontsize=13)
        fig.tight_layout()
        return fig

    confounder_grid_fig = _confounder_grid(sweep_summary)
    confounder_grid_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### 2 · Robustness statistics — is movement real or seed noise?

    For each (metric, confounder, distortion): the **slope** of the metric vs
    $\log_{10}$(property), with a bootstrap 95% CI over seeds, plus the **confounder
    span** (max−min of seed means) and the typical **seed-noise band** (median per-point
    SD). A metric is *robust* along an axis when the slope CI brackets 0 and the span is
    within the seed-noise band. `|slope| CI excludes 0` flags a real confounding effect.
    """)
    return


@app.cell
def _(METRIC_COLUMNS, METRIC_LABELS, np, pd, sweep_raw):
    def _robustness_table(raw, n_boot=2000, rng_seed=0):
        rng = np.random.default_rng(rng_seed)
        confounders = ("n", "clusters", "dim", "imbalance")
        distortions = ("dropout", "gaussian")
        rows = []
        for metric in METRIC_COLUMNS:
            for pname in confounders:
                for dist in distortions:
                    part = raw[(raw["param_name"] == pname) & (raw["distortion"] == dist)]
                    if part.empty:
                        continue
                    logx = np.log10(part["param_value"].to_numpy())
                    y = part[metric].to_numpy()
                    # slope via least squares
                    A = np.vstack([logx, np.ones_like(logx)]).T
                    slope = float(np.linalg.lstsq(A, y, rcond=None)[0][0])
                    # bootstrap CI over rows (resample conditions+seeds)
                    boot = np.empty(n_boot)
                    n = len(y)
                    for b in range(n_boot):
                        idx = rng.integers(0, n, n)
                        boot[b] = np.linalg.lstsq(A[idx], y[idx], rcond=None)[0][0]
                    lo, hi = np.percentile(boot, [2.5, 97.5])
                    # span of seed means + median seed-noise band
                    g = part.groupby("param_value")[metric]
                    span = float(g.mean().max() - g.mean().min())
                    seed_noise = float(g.std().median())
                    rows.append({
                        "metric": METRIC_LABELS[metric], "confounder": pname,
                        "distortion": dist, "slope": round(slope, 4),
                        "slope_ci_lo": round(lo, 4), "slope_ci_hi": round(hi, 4),
                        "confounded": "yes" if (lo > 0 or hi < 0) else "no",
                        "span": round(span, 4), "seed_noise": round(seed_noise, 4),
                    })
        return pd.DataFrame(rows)

    robustness_table = _robustness_table(sweep_raw)
    robustness_table
    return (robustness_table,)


@app.cell
def _(METRIC_COLUMNS, METRIC_LABELS, np, plt, robustness_table):
    def _robustness_heatmap(tbl):
        confounders = ("n", "clusters", "dim", "imbalance")
        metrics = [METRIC_LABELS[m] for m in METRIC_COLUMNS]
        # use the larger |slope| across the two distortions as the conservative summary
        mat = np.zeros((len(metrics), len(confounders)))
        flag = np.empty((len(metrics), len(confounders)), dtype=object)
        for i, mlab in enumerate(metrics):
            for j, c in enumerate(confounders):
                sub = tbl[(tbl["metric"] == mlab) & (tbl["confounder"] == c)]
                k = sub["slope"].abs().idxmax()
                mat[i, j] = abs(sub.loc[k, "slope"])
                flag[i, j] = "*" if (sub["confounded"] == "yes").any() else ""
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto",
                       vmin=0, vmax=max(0.05, mat.max()))
        ax.set_xticks(range(len(confounders))); ax.set_xticklabels(confounders)
        ax.set_yticks(range(len(metrics))); ax.set_yticklabels(metrics)
        for i in range(len(metrics)):
            for j in range(len(confounders)):
                ax.text(j, i, f"{mat[i, j]:.3f}{flag[i, j]}", ha="center", va="center",
                        fontsize=9, color="white" if mat[i, j] > mat.max() * 0.6 else "black")
        ax.set_title("Max |slope| vs $\\log_{10}$(property)\n(* = slope CI excludes 0 for ≥1 distortion)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="|slope|")
        fig.tight_layout()
        return fig

    robustness_heatmap_fig = _robustness_heatmap(robustness_table)
    robustness_heatmap_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### 3 · $R_{NX}$ curves along the sweeps

    The qualitative companion to the scalar scores: how the whole preservation curve
    deforms as a confounder changes (at constant injected distortion, dropout). If the
    curve shifts vertically the scalar moves; if it changes *shape*, Local and Global
    SP move differently. $u=K/n$ on the x-axis makes curves at different $n$ comparable.
    """)
    return


@app.cell
def _(np, plt, sweep_curves):
    def _rnx_sweep_figure(curves):
        panels = (("n", "dataset size"), ("clusters", "cluster count"),
                  ("dim", "embedding dim"), ("imbalance", "size imbalance"))
        fig, axes = plt.subplots(1, 4, figsize=(17, 4))
        for ax, (pname, title) in zip(axes, panels):
            sub = [c for c in curves if c["param_name"] == pname and c["distortion"] == "dropout"]
            sub = sorted(sub, key=lambda c: c["param_value"])
            colors = plt.cm.viridis(np.linspace(0, 0.85, len(sub)))
            for c, col in zip(sub, colors):
                rnx = c["rnx"]
                n_emb = len(rnx) + 2
                u = np.arange(1, n_emb - 1) / n_emb
                ax.plot(u, rnx, color=col, lw=1.6, label=f"{c['param_value']:.0f}")
            ax.set_xscale("log")
            ax.set_xlabel(r"relative neighbourhood size $u=K/n$")
            ax.set_title(title, fontsize=11)
            ax.set_ylim(-0.05, 1.05)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=7, title=title.split()[0])
        axes[0].set_ylabel(r"$R_{NX}(K)$")
        fig.suptitle(r"$R_{NX}$ curves across each confounder (dropout, constant $D$)", y=1.02)
        fig.tight_layout()
        return fig

    rnx_sweep_fig = _rnx_sweep_figure(sweep_curves)
    rnx_sweep_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### 4 · ARI resolution dependence

    ARI is reported as the median over a Leiden resolution sweep, because a single
    resolution is arbitrary. This panel shows the full resolution dependence at the
    extremes of the cluster-count sweep — the axis most likely to confound clustering
    agreement. Spread across resolutions = how resolution-sensitive the ARI reading is.
    """)
    return


@app.cell
def _(LEIDEN_RESOLUTIONS, np, plt, sweep_curves):
    def _ari_resolution_figure(curves):
        sub = [c for c in curves if c["param_name"] == "clusters"
               and c["distortion"] == "dropout"]
        sub = sorted(sub, key=lambda c: c["param_value"])
        fig, ax = plt.subplots(figsize=(7.5, 5))
        colors = plt.cm.plasma(np.linspace(0, 0.85, len(sub)))
        res = np.asarray(LEIDEN_RESOLUTIONS)
        for c, col in zip(sub, colors):
            ax.plot(res, c["ari_curve"], color=col, marker="o", ms=5, lw=1.6,
                    label=f"{c['param_value']:.0f} clusters")
        ax.set_xscale("log")
        ax.set_xlabel("Leiden resolution")
        ax.set_ylabel("adjusted Rand index")
        ax.set_title("ARI vs Leiden resolution, across cluster count\n(constant injected distortion)")
        ax.set_ylim(-0.05, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        return fig

    ari_resolution_fig = _ari_resolution_figure(sweep_curves)
    ari_resolution_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Reading this figure set / what it licenses

    - **Panel 0 (calibration)** is the validity gate. If realized $D$ is flat on target,
      everything downstream compares like with like. This is the methodological
      difference from a naive fixed-knob sweep.
    - **Panel 1 + the stats table** are the result: for each metric and each confounder,
      whether the metric moves at constant injected distortion, judged against seed
      noise with a bootstrap CI. The `confounded` column is the defensible verdict.
    - **Two distortion types** guard against a conclusion that is an artefact of
      coordinate dropout; agreement between solid and dashed lines is the evidence.
    - **$R_{NX}$ curves and ARI-vs-resolution** show the qualitative mechanism behind
      the scalar movements, and demonstrate the resolution-robustness of the ARI summary.

    **Scope, stated plainly.** This characterises metric behaviour on Gaussian-blob
    geometry under two synthetic distortions, at one anchor level $D$. It is evidence
    about confounder *sensitivity of the metrics*, suitable as a supplementary
    methods-validation figure. It does not by itself establish behaviour on real scFM
    embeddings under the biological interventions — that requires the same calibrated
    design applied to the real reference/manipulation pairs, which this notebook's
    structure (anchor + calibrate + score) ports to directly.
    """)
    return


if __name__ == "__main__":
    app.run()
