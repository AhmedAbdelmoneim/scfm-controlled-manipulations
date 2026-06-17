import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ViScore — package functions, interpretation, and stability

    Uses the **actual `viscore` package** (saeyslab/ViScore) on synthetic Gaussian-blob
    data with a reference / manipulated embedding pair. Three tools:

    | Function | Type | Output | Direction |
    |---|---|---|---|
    | `viscore.score` | unsupervised, label-free | Local SP `Sl`, Global SP `Sg`, `RNX` curve | higher better, ∈[−1,1] |
    | `viscore.xnpe` | supervised, per-population | EMD-based distortion per label | lower better, ∈[0,~1] |
    | `viscore.neighbourhood_composition` | supervised, qualitative | neighbourhood label composition | diagnostic plot |

    Sections: (1) `score` + RNX curve, (2) Local vs Global across distortion,
    (3) `xnpe` per population, (4) neighbourhood composition plots, (5) **confounder
    sweeps** — how stable each score is when a nuisance property changes at fixed distortion.

    > Install: `pip install pyemd==2.0.0 POT` then
    > `pip install git+https://github.com/saeyslab/ViScore.git`.
    > (pyemd 2.0 is the POT-backed build; the older 1.0 Cython build breaks on NumPy 2.x.)
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    import viscore

    # synthetic defaults
    SYNTH_N = 600
    SYNTH_DIM = 64
    SYNTH_N_CLUSTERS = 6
    SYNTH_SEP = 8.0

    # distortion sweeps
    DROPOUT_RATES = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95)

    XNPE_K = 100          # neighbourhood size for xNPE
    NC_K = 150            # max neighbourhood size for composition plots
    NC_STEPSIZE = 15

    # confounder sweeps (fixed distortion). Heavier dropout so xNPE has dynamic range.
    FIXED_DROPOUT_RATE = 0.7
    SWEEP_N_VALUES = (300, 600, 1000, 1500, 2000)
    SWEEP_CLUSTER_VALUES = (3, 6, 12, 30, 60)
    SWEEP_SEP_VALUES = (2.0, 4.0, 8.0, 16.0, 32.0)
    SWEEP_DIM_VALUES = (32, 64, 128, 256, 512)
    SWEEP_IMBALANCE_VALUES = (1, 3, 10, 30, 100)
    SWEEP_SEEDS = (0, 1, 2)
    return (
        DROPOUT_RATES,
        FIXED_DROPOUT_RATE,
        NC_K,
        NC_STEPSIZE,
        SWEEP_CLUSTER_VALUES,
        SWEEP_DIM_VALUES,
        SWEEP_IMBALANCE_VALUES,
        SWEEP_N_VALUES,
        SWEEP_SEEDS,
        SWEEP_SEP_VALUES,
        SYNTH_DIM,
        SYNTH_N,
        SYNTH_N_CLUSTERS,
        SYNTH_SEP,
        np,
        pd,
        plt,
        viscore,
    )


@app.cell
def _(SYNTH_DIM, SYNTH_N, SYNTH_N_CLUSTERS, SYNTH_SEP, np):
    def make_blobs(*, n=SYNTH_N, dim=SYNTH_DIM, n_clusters=SYNTH_N_CLUSTERS,
                   sep=SYNTH_SEP, sizes=None, seed=0):
        """Isotropic Gaussian clusters; returns (coords, string labels)."""
        rng = np.random.default_rng(seed)
        centres = rng.normal(0.0, sep, size=(n_clusters, dim))
        if sizes is None:
            labels = rng.integers(0, n_clusters, size=n)
        else:
            p = np.asarray(sizes, dtype=np.float64); p = p / p.sum()
            labels = rng.choice(n_clusters, size=n, p=p)
        coords = centres[labels] + rng.normal(0.0, 1.0, size=(n, dim))
        return coords, labels.astype(str)

    def degrade(emb, rate, *, seed=0):
        """Coordinate dropout: zero each coordinate independently w.p. `rate`."""
        if rate <= 0.0:
            return emb.copy()
        rng = np.random.default_rng(seed)
        out = emb.copy()
        out[rng.random(out.shape) < rate] = 0.0
        return out

    return degrade, make_blobs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1 · `viscore.score` — Local & Global structure preservation

    **What it computes.** From the ranks of pairwise distances in the reference vs the
    manipulated embedding, it builds the $R_{NX}(K)$ curve: for each neighbourhood size
    $K$, the chance-corrected fraction of $K$-nearest neighbours preserved. Two AUC
    summaries collapse the curve:

    - **Local SP `Sl`** — AUC with a *log* $K$ axis, upweighting small neighbourhoods.
      Sensitive to whether tight local structure survives.
    - **Global SP `Sg`** — AUC with a *linear* $K$ axis, equal weight to all scales.
      Sensitive to whether coarse/global arrangement survives.

    **How to read it.** Both are bounded in $[-1, 1]$. **1** = perfect preservation,
    **0** = no better than a random embedding, **negative** = worse than random. There
    is no labelling involved — this is the unsupervised, label-free score. A large gap
    `Sl > Sg` means local structure survived better than global (or vice versa), which
    is the main diagnostic value of having both.
    """)
    return


@app.cell
def _(degrade, make_blobs, np, pd, plt, viscore):
    def _score_demo():
        ref, _ = make_blobs(n=500, seed=0)
        rng = np.random.default_rng(1)
        cases = {
            "identity": ref.copy(),
            "dropout 0.3": degrade(ref, 0.3, seed=9),
            "dropout 0.7": degrade(ref, 0.7, seed=9),
            "random": rng.normal(0, 1, size=ref.shape),
        }
        results = {name: viscore.score(ref, man) for name, man in cases.items()}
        table = pd.DataFrame([
            {"case": name, "Sl (Local)": round(r["Sl"], 4), "Sg (Global)": round(r["Sg"], 4)}
            for name, r in results.items()
        ])

        fig, ax = plt.subplots(figsize=(7.5, 5))
        colors = plt.cm.viridis(np.linspace(0, 0.85, len(results)))
        for (name, r), col in zip(results.items(), colors):
            K = np.arange(1, len(r["RNX"]) + 1)
            ax.plot(K, r["RNX"], color=col, lw=1.8,
                    label=f"{name}  (Sl={r['Sl']:.2f}, Sg={r['Sg']:.2f})")
        ax.set_xscale("log")
        ax.axhline(0, color="grey", ls="--", lw=1)
        ax.set_xlabel("neighbourhood size $K$ (log scale)")
        ax.set_ylabel(r"$R_{NX}(K)$")
        ax.set_title("ViScore RNX curves — reference vs manipulated")
        ax.set_ylim(-0.05, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        return table, fig

    score_table, score_fig = _score_demo()
    score_fig
    return (score_table,)


@app.cell
def _(score_table):
    score_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2 · Local vs Global SP across a distortion sweep

    **What to look for.** As dropout increases, both scores fall from 1 toward 0. If the
    two curves separate, the manipulation is breaking local and global structure at
    different rates — exactly the multi-scale distinction ViScore is built to expose.
    Bands are ±1 SD over seeds; if they are tight, the score is reproducible at that
    distortion level.
    """)
    return


@app.cell
def _(DROPOUT_RATES, degrade, make_blobs, pd, plt, viscore):
    def _sp_sweep():
        rows = []
        for rate in DROPOUT_RATES:
            for seed in (0, 1, 2):
                ref, _ = make_blobs(n=500, seed=seed)
                man = degrade(ref, float(rate), seed=100 + seed)
                s = viscore.score(ref, man)
                rows.append({"dropout": rate, "seed": seed, "Sl": s["Sl"], "Sg": s["Sg"]})
        return pd.DataFrame(rows)

    def _sp_figure(df):
        g = df.groupby("dropout")
        mean, sd = g.mean(), g.std()
        x = mean.index.to_numpy()
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for col, color, lab in [("Sl", "#C44E52", "Local SP $S_l$"),
                                 ("Sg", "#4C72B0", "Global SP $S_g$")]:
            ax.fill_between(x, mean[col] - sd[col], mean[col] + sd[col],
                            color=color, alpha=0.15, linewidth=0)
            ax.plot(x, mean[col], color=color, marker="o", ms=5, lw=1.9, label=lab)
        ax.axhline(0, color="grey", ls="--", lw=1)
        ax.set_xlabel("dropout rate $p$")
        ax.set_ylabel("structure preservation")
        ax.set_title("Local vs Global SP across distortion")
        ax.set_ylim(-0.05, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=10)
        fig.tight_layout()
        return fig

    sp_sweep_fig = _sp_figure(_sp_sweep())
    sp_sweep_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4 · `viscore.neighbourhood_composition` — positional distortion

    **What it computes.** For a query population, it finds that population's nearest
    neighbours among all *other* populations (excluding itself) and records the label
    composition of those neighbours across neighbour-rank bins. The result is a
    stacked-area plot: the bands show which populations sit closest to the query, and
    how that mix changes as you widen the neighbourhood.

    **How to read it.** Compare the **reference** panel against the **manipulated**
    panel for the same query population. If the manipulation preserves relative
    position, the two stacks look the same. If a different population's band swells into
    the near neighbourhood (left side) in the manipulated panel, the manipulation has
    pulled that population spuriously close — a concrete, localised picture of
    positional distortion that the scalar scores cannot show.
    """)
    return


@app.cell
def _(NC_K, NC_STEPSIZE, degrade, make_blobs, np, plt, viscore):
    _PALETTE = ['#1CE6FF', '#FF34FF', '#FF4A46', '#008941', '#006FA6', '#A30059',
                '#7A4900', '#dedb8c', '#63FFAC', '#B79762', '#004D43', '#8FB0FF']

    def _composition_panel(ax, nc, title):
        _pop_query, pops_ref, _counts, proportions, x = nc
        y = np.vstack(proportions)
        ax.stackplot(x, y.T, labels=[str(p) for p in pops_ref],
                     colors=_PALETTE[:len(pops_ref)])
        ax.set_xlabel("neighbour rank")
        ax.set_ylabel("composition")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.margins(x=0)

    def _composition_demo(query_pop="0", dropout=0.85):
        ref, lab = make_blobs(n=600, seed=0)
        man = degrade(ref, dropout, seed=99)

        nc_ref = viscore.neighbourhood_composition(
            X=ref, pop=query_pop, annot=lab, k=NC_K, stepsize=NC_STEPSIZE)
        nc_man = viscore.neighbourhood_composition(
            X=man, pop=query_pop, annot=lab, k=NC_K, stepsize=NC_STEPSIZE)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
        _composition_panel(axes[0], nc_ref, f"Reference — neighbours of pop {query_pop}")
        _composition_panel(axes[1], nc_man, f"Manipulated (p={dropout}) — neighbours of pop {query_pop}")
        axes[1].legend(title="reference pop", bbox_to_anchor=(1.02, 1.0),
                       loc="upper left", frameon=False, fontsize=8)
        fig.suptitle("Neighbourhood composition — reference vs manipulated", y=1.02)
        fig.tight_layout()
        return fig

    composition_fig = _composition_demo()
    composition_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Confounder sweeps — how stable are the scores?

    The key question for cross-dataset use: at a **fixed** level of true distortion
    (dropout held constant), does a score move when a *nuisance* property of the data
    changes? If yes, that score is confounded by that property and is not directly
    comparable across datasets that differ in it.

    Each panel varies one property — dataset size $n$, cluster count, cluster
    separation, embedding dimension, size imbalance — with dropout fixed at
    $p={FIXED}$. **Flat = stable / robust; sloped = confounded.** Tracked scores:
    Local SP, Global SP, and mean xNPE.

    This is the same confounder design used to characterise the rank-based metrics
    elsewhere; here it stress-tests the ViScore scores specifically.

    *Runtime: the full sweep is 5 properties × 5 values × 3 seeds = 75 pairs. `score`
    is fast (vantage-point-tree approximation) but `xnpe` builds kNN graphs via
    pynndescent (~3-5 s each), so this cell takes several minutes. Reduce
    `SWEEP_SEEDS` or the value tuples to speed it up.
    """)
    return


@app.cell
def _(
    FIXED_DROPOUT_RATE,
    SWEEP_CLUSTER_VALUES,
    SWEEP_DIM_VALUES,
    SWEEP_IMBALANCE_VALUES,
    SWEEP_N_VALUES,
    SWEEP_SEEDS,
    SWEEP_SEP_VALUES,
    SYNTH_N,
    degrade,
    make_blobs,
    pd,
    viscore,
):
    def _score_one(ref, lab, dropout, seed, xnpe_k):
        man = degrade(ref, dropout, seed=5000 + seed)
        s = viscore.score(ref, man)
        xn = viscore.xnpe(hd=ref, ld=man, annot=lab, k=xnpe_k, reduce="average")
        return {"Sl": s["Sl"], "Sg": s["Sg"], "xnpe": float(xn)}

    def _sweep(param_name, values, build, *, xnpe_k_of):
        rows = []
        for v in values:
            for seed in SWEEP_SEEDS:
                ref, lab = build(v, seed)
                m = _score_one(ref, lab, FIXED_DROPOUT_RATE, seed, xnpe_k_of(v, ref))
                rows.append({"param_name": param_name, "param_value": float(v), "seed": seed, **m})
        return pd.DataFrame(rows)

    # xNPE k must be < smallest population; scale it conservatively with n / clusters
    def _k_default(v, ref):
        return min(100, max(10, ref.shape[0] // 12))

    confounder_raw = pd.concat([
        _sweep("n", SWEEP_N_VALUES,
               lambda v, s: make_blobs(n=int(v), seed=s), xnpe_k_of=_k_default),
        _sweep("clusters", SWEEP_CLUSTER_VALUES,
               lambda v, s: make_blobs(n=SYNTH_N, n_clusters=int(v), seed=s), xnpe_k_of=_k_default),
        _sweep("sep", SWEEP_SEP_VALUES,
               lambda v, s: make_blobs(n=SYNTH_N, sep=float(v), seed=s), xnpe_k_of=_k_default),
        _sweep("dim", SWEEP_DIM_VALUES,
               lambda v, s: make_blobs(n=SYNTH_N, dim=int(v), seed=s), xnpe_k_of=_k_default),
        _sweep("imbalance", SWEEP_IMBALANCE_VALUES,
               lambda v, s: make_blobs(n=SYNTH_N, n_clusters=6, sizes=[int(v)] + [1] * 5, seed=s),
               xnpe_k_of=_k_default),
    ], ignore_index=True)

    confounder_summary = (
        confounder_raw
        .groupby(["param_name", "param_value"], as_index=False)[["Sl", "Sg", "xnpe"]]
        .mean()
        .sort_values(["param_name", "param_value"])
    )
    print(f"{len(confounder_raw)} scored pairs · fixed dropout p={FIXED_DROPOUT_RATE}")
    confounder_summary
    return (confounder_summary,)


@app.cell
def _(FIXED_DROPOUT_RATE, confounder_summary, plt):
    _METRICS = [("Sl", "#C44E52", "Local SP"),
                ("Sg", "#4C72B0", "Global SP"),
                ("xnpe", "#55A868", "mean xNPE")]

    def _confounder_figure(summary):
        panels = (
            ("n", "dataset size $n$", True),
            ("clusters", "cluster count", True),
            ("sep", "cluster separation", True),
            ("dim", "embedding dimension", True),
            ("imbalance", "size imbalance (dominant:rest)", True),
        )
        fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4))
        flat = axes.ravel()
        for ax, (pname, title, logx) in zip(flat[:5], panels):
            part = summary[summary["param_name"] == pname].sort_values("param_value")
            for col, color, lab in _METRICS:
                ax.plot(part["param_value"], part[col], color=color, marker="o",
                        ms=4, lw=1.6, label=lab)
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(title)
            ax.set_ylabel("score")
            ax.axhline(0, color="grey", ls=":", lw=0.8)
            ax.axhline(1, color="grey", ls=":", lw=0.8)
            ax.set_ylim(-0.05, 1.15)
            ax.spines[["top", "right"]].set_visible(False)
        flat[5].axis("off")
        flat[5].legend(*flat[0].get_legend_handles_labels(), loc="center",
                       frameon=False, fontsize=11,
                       title=f"fixed dropout $p={FIXED_DROPOUT_RATE}$\n(flat = stable)")
        fig.suptitle("ViScore confounder sweeps — score movement at constant distortion", y=0.99)
        fig.tight_layout()
        return fig

    confounder_fig = _confounder_figure(confounder_summary)
    confounder_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the confounder panels

    - **`Sl` / `Sg` (higher better)**: a flat line means the score reports the same
      preservation regardless of that nuisance property — the property is *not* a
      confound for cross-dataset comparison. A rising or falling line means it is.
      Watch cluster count and separation in particular, the usual offenders for
      rank-based scores.
    - **`xnpe` (lower better)**: flat means the per-population error is stable; movement
      means apparent population distortion depends on the property even at fixed true
      distortion.
    - **Reproducibility** is the seed spread folded into the means here; if a score is
      both flat *and* low-variance across the sweep, it is a candidate for
      cross-dataset use along that axis. None of these scores is guaranteed
      confound-free — the plot is the evidence for which axes each can and cannot be
      compared across.
    """)
    return


if __name__ == "__main__":
    app.run()
