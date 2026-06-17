import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Synthetic metric characterization

    Gaussian-blob reference embeddings; **manipulated** copies degraded by coordinate
    **dropout** (each coordinate zeroed independently with probability $p$).

    - **Distortion sweep** varies $p$ across the full range → each metric's dynamic range.
    - **Confounder sweeps** vary one nuisance property at fixed $p$ → bias at constant distortion.
    - **Signal-vs-confounder** ratios = confounder span / distortion span per metric.

    Metrics (all derived from two distance matrices + one co-ranking matrix per pair):
    Trustworthiness, Continuity, $R_\mathrm{NX}$ (raw, normalized, & $\log u$ AUC),
    Distance correlation, and **per-cell neighbour-survival AUROC** (raw & EGAD null-corrected).
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import pdist, squareform

    # --- metric parameters ---
    TC_K = 15                                  # neighbourhood size for T & C
    AUROC_K = 15                               # reference-neighbourhood size for survival AUROC
    RNX_AUC_FRACS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95)

    # --- synthetic blob defaults (confounder baseline) ---
    SYNTH_N = 1000
    SYNTH_DIM = 128
    SYNTH_N_CLUSTERS = 10
    SYNTH_SEP = 8.0

    # --- distortion sweep: full dynamic range (dropout needs ~0.95 to saturate) ---
    DROPOUT_RATES = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95)
    # signal range measured over this interior band (exclude p=0 identity and the extreme tail)
    SIGNAL_BAND = (0.1, 0.85)

    # --- confounder sweeps (dropout held fixed) ---
    FIXED_DROPOUT_RATE = 0.3
    SWEEP_N_VALUES = (200, 500, 1000, 2000, 5000)
    SWEEP_CLUSTER_VALUES = (3, 10, 30, 75, 200)
    SWEEP_SEP_VALUES = (2.0, 4.0, 8.0, 16.0, 32.0)
    SWEEP_DIM_VALUES = (128, 256, 512, 1024, 2048)
    SWEEP_IMBALANCE_VALUES = (1, 3, 10, 30, 100)

    SWEEP_SEEDS = (0, 1, 2)

    # # display order + pretty labels
    # METRIC_COLUMNS = ("trustworthiness", "continuity", "distcorr",
    #                   "rnx_auc_raw", "rnx_auc_norm", "auroc", "auroc_corrected")

    METRIC_COLUMNS = ("distcorr", "rnx_auc_raw", "rnx_auc_norm", "rnx_auc_logu")
    METRIC_LABELS = {
        "trustworthiness": "Trustworthiness",
        "continuity": "Continuity",
        "distcorr": "Distance corr.",
        "rnx_auc_raw": r"$R_{NX}$ AUC (raw)",
        "rnx_auc_norm": r"$R_{NX}$ AUC (norm)",
        "rnx_auc_logu": r"$R_{NX}$ AUC ($\log u$)",
        "auroc": "Survival AUROC",
        "auroc_corrected": "Survival AUROC (null-corr.)",
    }
    METRIC_COLORS = {
        "trustworthiness": "#4C72B0",
        "continuity": "#DD8452",
        "distcorr": "#55A868",
        "rnx_auc_raw": "#C44E52",
        "rnx_auc_norm": "#8172B3",
        "rnx_auc_logu": "#CCB974",
        "auroc": "#937860",
        "auroc_corrected": "#DA8BC3",
    }
    return (
        AUROC_K,
        DROPOUT_RATES,
        FIXED_DROPOUT_RATE,
        METRIC_COLORS,
        METRIC_COLUMNS,
        METRIC_LABELS,
        RNX_AUC_FRACS,
        SIGNAL_BAND,
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
        TC_K,
        np,
        pd,
        pdist,
        plt,
        squareform,
    )


@app.cell
def _(SYNTH_DIM, SYNTH_N, SYNTH_N_CLUSTERS, SYNTH_SEP, np):
    def make_blobs(
        *,
        n: int = SYNTH_N,
        dim: int = SYNTH_DIM,
        n_clusters: int = SYNTH_N_CLUSTERS,
        sep: float = SYNTH_SEP,
        sizes: list[int] | None = None,
        seed: int = 0,
    ) -> np.ndarray:
        """Isotropic Gaussian clusters (unit within-cluster sd) with centres spread at
        scale `sep`. `sizes` gives per-cluster relative weights for imbalance sweeps."""
        rng = np.random.default_rng(seed)
        centres = rng.normal(0.0, sep, size=(n_clusters, dim))
        if sizes is None:
            labels = rng.integers(0, n_clusters, size=n)
        else:
            probs = np.asarray(sizes, dtype=np.float64)
            probs = probs / probs.sum()
            labels = rng.choice(n_clusters, size=n, p=probs)
        return centres[labels] + rng.normal(0.0, 1.0, size=(n, dim))

    def degrade(emb: np.ndarray, rate: float, *, seed: int = 0) -> np.ndarray:
        """Zero each coordinate independently with probability `rate` (dropout)."""
        if rate <= 0.0:
            return emb.copy()
        rng = np.random.default_rng(seed)
        out = emb.copy()
        out[rng.random(out.shape) < rate] = 0.0
        return out

    return degrade, make_blobs


@app.cell
def _(AUROC_K, RNX_AUC_FRACS, TC_K, np, pdist, squareform):
    # ---- Fused metric core --------------------------------------------------------
    # Everything flows from two distance matrices (ref, man) + one co-ranking matrix.
    # Fully vectorized, no sklearn:
    #   * ranks via double-argsort on the shared distance matrix
    #   * co-ranking via np.bincount (≈4x faster than np.add.at)
    #   * T / C read directly off Q (match sklearn.trustworthiness to 1e-6)
    #   * distance correlation via O(n^2) mean-centering (no n^3 H S H matmul)
    #   * per-cell neighbour-survival AUROC via the Mann-Whitney rank-sum identity
    #     (no per-cell loop; matches sklearn.roc_auc_score to 1e-16)

    def _ranks_from_distance(D: np.ndarray) -> np.ndarray:
        Dd = D.copy()
        np.fill_diagonal(Dd, np.inf)
        return Dd.argsort(1, kind="stable").argsort(1, kind="stable")

    def _coranking_from_ranks(rk_ref, rk_man):
        n = rk_ref.shape[0]
        mask = ~np.eye(n, dtype=bool)
        kk = rk_ref[mask].ravel()
        ll = rk_man[mask].ravel()
        valid = (kk < n - 1) & (ll < n - 1)
        kk, ll = kk[valid], ll[valid]
        flat = kk * (n - 1) + ll
        Q = np.bincount(flat, minlength=(n - 1) * (n - 1)).astype(np.float64)
        return Q.reshape(n - 1, n - 1)

    def _rnx_curve(Q):
        n = Q.shape[0] + 1
        csum = Q.cumsum(0).cumsum(1)
        K = np.arange(1, Q.shape[0] + 1)
        qnx = (np.diagonal(csum) / (K * n))[:-1]
        K = np.arange(1, n - 1)
        return ((n - 1) * qnx - K) / (n - 1 - K)

    def _auc_raw(curve):
        K = np.arange(1, len(curve) + 1, dtype=np.float64)
        w = 1.0 / K
        return float((curve * w).sum() / w.sum())

    def _auc_norm(curve, fracs=RNX_AUC_FRACS):
        n = len(curve) + 1
        idx = np.array([max(0, min(len(curve) - 1, int(round(f * n)) - 1)) for f in fracs])
        w = 1.0 / np.asarray(fracs, dtype=np.float64)
        return float((curve[idx] * w).sum() / w.sum())

    def _auc_logu(curve):
        n = len(curve) + 2  # embedding size; len(curve) = n - 2
        u = np.arange(1, n - 1, dtype=np.float64) / n
        return float(np.trapezoid(curve, x=np.log(u)))

    def _tc_from_coranking(Q, k):
        n = Q.shape[0] + 1
        ranks = np.arange(1, n)
        norm = n * k * (2 * n - 3 * k - 1) if k < n / 2 else n * (n - k) * (n - k - 1)
        pen_t = ((ranks[k:][:, None] - k) * Q[k:, :k]).sum()
        pen_c = ((ranks[k:][None, :] - k) * Q[:k, k:]).sum()
        return 1.0 - (2.0 / norm) * pen_t, 1.0 - (2.0 / norm) * pen_c

    def _distcorr_from_distances(D_ref, D_man):
        def _center(D):
            return D - D.mean(1, keepdims=True) - D.mean(0, keepdims=True) + D.mean()
        A, B = _center(D_ref), _center(D_man)
        n = A.shape[0]
        norm_a = np.sqrt((A * A).sum()) / n
        norm_b = np.sqrt((B * B).sum()) / n
        den = norm_a * norm_b
        return float((A * B).sum() / (n * n) / den) if den > 0.0 else 0.0

    def _neighbour_survival_auroc(D_ref, D_man, k):
        """Per-cell neighbour-survival AUROC and its EGAD-style structure null.

        Positives for anchor i = i's k nearest cells in the REFERENCE embedding.
        Raw score    : rank candidates by MANIPULATED similarity to i  (-D_man[i]).
        Null score   : rank candidates by HUBNESS (total similarity to everyone),
                       identical for every anchor → the AUROC attributable to graph
                       structure alone, independent of i. Corrected = raw - null.

        AUROC computed via the Mann-Whitney U identity (no per-cell loop):
            AUROC = U / (npos * nneg),  U = sum(score_rank over positives) - npos(npos-1)/2 - npos
        the trailing -npos removes the self-candidate (forced to lowest score)."""
        n = D_ref.shape[0]
        rows = np.arange(n)[:, None]
        Dr = D_ref.copy(); np.fill_diagonal(Dr, np.inf)
        pos_mask = np.zeros((n, n), dtype=bool)
        pos_mask[rows, Dr.argsort(1, kind="stable")[:, :k]] = True
        npos = k
        nneg = (n - 1) - npos

        def _auroc(score):
            s = score.astype(np.float64).copy()
            np.fill_diagonal(s, -np.inf)                      # self always lowest
            order = s.argsort(1, kind="stable")
            rnk = np.empty((n, n))
            rnk[rows, order] = np.arange(n)[None, :]          # 0 = lowest score
            sum_pos = (rnk * pos_mask).sum(1)
            U = sum_pos - npos * (npos - 1) / 2 - npos
            return U / (npos * nneg)

        sim_man = -D_man
        raw = _auroc(sim_man)
        sim_pos = sim_man.copy(); np.fill_diagonal(sim_pos, 0.0)
        hub = sim_pos.sum(0)                                  # j's total similarity to all anchors
        null = _auroc(np.broadcast_to(hub, (n, n)))
        return float(raw.mean()), float((raw - null).mean())

    def score_embedding_pair(emb_ref, emb_man, *, k: int = TC_K, auroc_k: int = AUROC_K) -> dict:
        """All seven metrics from one shared set of distance / rank / co-ranking intermediates."""
        D_ref = squareform(pdist(emb_ref.astype(np.float64)))
        D_man = squareform(pdist(emb_man.astype(np.float64)))
        Q = _coranking_from_ranks(_ranks_from_distance(D_ref), _ranks_from_distance(D_man))
        rnx = _rnx_curve(Q)
        t, c = _tc_from_coranking(Q, k)
        auroc, auroc_corr = _neighbour_survival_auroc(D_ref, D_man, auroc_k)
        return {
            "trustworthiness": t,
            "continuity": c,
            "distcorr": _distcorr_from_distances(D_ref, D_man),
            "rnx_auc_raw": _auc_raw(rnx),
            "rnx_auc_norm": _auc_norm(rnx),
            "rnx_auc_logu": _auc_logu(rnx),
            "auroc": auroc,
            "auroc_corrected": auroc_corr,
        }

    def embedding_pair_rnx_curve(emb_ref, emb_man) -> dict:
        """R_NX curve plus normalized and log-u summary scores for one ref/man pair."""
        D_ref = squareform(pdist(emb_ref.astype(np.float64)))
        D_man = squareform(pdist(emb_man.astype(np.float64)))
        Q = _coranking_from_ranks(_ranks_from_distance(D_ref), _ranks_from_distance(D_man))
        rnx = _rnx_curve(Q)
        n_emb = len(rnx) + 2
        u = np.arange(1, n_emb - 1, dtype=np.float64) / n_emb
        n_norm = len(rnx) + 1
        norm_idx = np.array([
            max(0, min(len(rnx) - 1, int(round(f * n_norm)) - 1))
            for f in RNX_AUC_FRACS
        ])
        return {
            "rnx": rnx,
            "u": u,
            "norm_idx": norm_idx,
            "rnx_auc_norm": _auc_norm(rnx),
            "rnx_auc_logu": _auc_logu(rnx),
        }

    return embedding_pair_rnx_curve, score_embedding_pair


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Validation

    identity → T/C/R_NX/AUROC = 1, distcorr = 1, corrected ≈ 0.5 (preservation above
    the hubness null) · rescale → all 1 · random → T/C ≈ chance, R_NX ≈ 0,
    AUROC ≈ 0.5, corrected ≈ 0 (no preservation beyond structure).
    """)
    return


@app.cell
def _(make_blobs, np, pd, score_embedding_pair):
    def _validation():
        base = make_blobs(n=300, seed=0)
        rng = np.random.default_rng(1)
        cases = {"identity": base.copy(), "rescale_x5": base * 5.0,
                 "random": rng.normal(0, 1, size=base.shape)}
        rows = []
        for name, man in cases.items():
            rows.append({"case": name, **{k: round(v, 4)
                         for k, v in score_embedding_pair(base, man).items()}})
        return pd.DataFrame(rows)

    validation_table = _validation()
    validation_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Run all experiments

    One pass building the distortion sweep and all five confounder sweeps. Long-form
    `synthetic_raw` (per seed) and seed-averaged `synthetic_summary` feed every figure.
    """)
    return


@app.cell
def _(
    DROPOUT_RATES,
    FIXED_DROPOUT_RATE,
    METRIC_COLUMNS,
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
    score_embedding_pair,
):
    def run_all_synthetic_experiments() -> pd.DataFrame:
        rows = []

        # distortion sweep: vary dropout on the default blob
        for rate in DROPOUT_RATES:
            for seed in SWEEP_SEEDS:
                ref = make_blobs(seed=seed)
                man = degrade(ref, float(rate), seed=100 + seed)
                rows.append({"sweep_kind": "distortion", "param_name": "dropout_rate",
                             "param_value": float(rate), "seed": seed,
                             **score_embedding_pair(ref, man)})

        # confounder sweeps: vary one property, dropout fixed
        confounder_builders = (
            ("n", SWEEP_N_VALUES, lambda v, s: make_blobs(n=int(v), seed=s)),
            ("clusters", SWEEP_CLUSTER_VALUES,
             lambda v, s: make_blobs(n=SYNTH_N, n_clusters=int(v), seed=s)),
            ("sep", SWEEP_SEP_VALUES,
             lambda v, s: make_blobs(n=SYNTH_N, sep=float(v), seed=s)),
            ("dim", SWEEP_DIM_VALUES,
             lambda v, s: make_blobs(n=SYNTH_N, dim=int(v), seed=s)),
            ("imbalance", SWEEP_IMBALANCE_VALUES,
             lambda v, s: make_blobs(n=SYNTH_N, n_clusters=10,
                                     sizes=[int(v)] + [1] * 9, seed=s)),
        )
        for param_name, values, build in confounder_builders:
            for value in values:
                for seed in SWEEP_SEEDS:
                    ref = build(value, seed)
                    man = degrade(ref, FIXED_DROPOUT_RATE, seed=1000 + seed)
                    rows.append({"sweep_kind": "confounder", "param_name": param_name,
                                 "param_value": float(value), "seed": seed,
                                 **score_embedding_pair(ref, man)})

        return pd.DataFrame(rows)

    synthetic_raw = run_all_synthetic_experiments()
    synthetic_summary = (
        synthetic_raw
        .groupby(["sweep_kind", "param_name", "param_value"], as_index=False)[list(METRIC_COLUMNS)]
        .mean()
        .sort_values(["sweep_kind", "param_name", "param_value"])
    )
    print(f"{len(synthetic_raw)} scored pairs · confounder dropout p={FIXED_DROPOUT_RATE}")
    synthetic_summary
    return synthetic_raw, synthetic_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### 1 · Distortion sweep — dynamic range

    Each metric vs dropout rate. A usable metric spans a wide vertical range here;
    one that stays flat near 1 cannot grade distortion. Bands show ±1 SD over seeds.
    Dashed = reference-only / non-primary variants ($R_{NX}$ raw, raw AUROC).
    """)
    return


@app.cell
def _(METRIC_COLORS, METRIC_COLUMNS, METRIC_LABELS, plt, synthetic_raw):
    _DASHED = {"rnx_auc_raw", "auroc"}

    def _distortion_figure(raw):
        df = raw[raw["sweep_kind"] == "distortion"]
        g = df.groupby("param_value")[list(METRIC_COLUMNS)]
        mean, sd = g.mean(), g.std()
        x = mean.index.to_numpy()

        fig, ax = plt.subplots(figsize=(8, 5.2))
        for m in METRIC_COLUMNS:
            ls = "--" if m in _DASHED else "-"
            ax.fill_between(x, mean[m] - sd[m], mean[m] + sd[m],
                            color=METRIC_COLORS[m], alpha=0.10, linewidth=0)
            ax.plot(x, mean[m], ls, color=METRIC_COLORS[m], marker="",
                    ms=4, lw=1.8, label=METRIC_LABELS[m])
        ax.set_xlabel("dropout rate $p$")
        ax.set_ylabel("metric value")
        ax.set_title("Distortion sweep — metric dynamic range")
        ax.set_ylim(-0.05, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, ncol=2)
        fig.tight_layout()
        return fig

    distortion_fig = _distortion_figure(synthetic_raw)
    distortion_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### 2 · Confounder sweeps — bias at fixed distortion

    Each panel varies one nuisance property with dropout held fixed. Flat = robust;
    movement = the metric responds to the property rather than to distortion. The
    null-corrected AUROC is the candidate for confounder robustness — watch whether it
    flattens where raw AUROC and $R_{NX}$ slope (esp. cluster count, imbalance).
    """)
    return


@app.cell
def _(
    FIXED_DROPOUT_RATE,
    METRIC_COLORS,
    METRIC_COLUMNS,
    METRIC_LABELS,
    plt,
    synthetic_summary,
):
    _DASHED2 = {"rnx_auc_raw", "auroc"}

    def _confounder_figure(summary):
        panels = (
            ("n", "dataset size $n$", True),
            ("clusters", "cluster count", True),
            ("sep", "cluster separation", True),
            ("dim", "embedding dimension", True),
            ("imbalance", "size imbalance (dominant:rest)", True),
        )
        fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4), sharey=True)
        flat = axes.ravel()
        for ax, (pname, title, logx) in zip(flat[:5], panels):
            part = summary[(summary["sweep_kind"] == "confounder")
                           & (summary["param_name"] == pname)].sort_values("param_value")
            for m in METRIC_COLUMNS:
                ls = "--" if m in _DASHED2 else "-"
                ax.plot(part["param_value"], part[m], ls, color=METRIC_COLORS[m],
                        marker="o", ms=3.5, lw=1.5, label=METRIC_LABELS[m])
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(title)
            ax.set_ylim(-0.05, 1.03)
            ax.spines[["top", "right"]].set_visible(False)
        flat[0].set_ylabel("metric value")
        flat[3].set_ylabel("metric value")
        flat[5].axis("off")
        flat[5].legend(*flat[0].get_legend_handles_labels(),
                       loc="center", frameon=False, fontsize=9,
                       title=f"fixed dropout $p={FIXED_DROPOUT_RATE}$")
        fig.suptitle("Confounder sweeps — metric movement at constant distortion", y=0.99)
        fig.tight_layout()
        return fig

    confounder_fig = _confounder_figure(synthetic_summary)
    confounder_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### 3 · Signal vs confounder — usability

    `ratio = confounder_span / distortion_span`, distortion span over the interior band
    (excludes identity and saturated tail). Green = confounder small next to real
    distortion (controllable); red = it moves the metric as much as distortion does
    (unusable for cross-condition comparison along that property). A metric with a tiny
    signal span is unusable regardless. The null-corrected AUROC should show the
    greenest cluster/imbalance columns if the EGAD correction works.
    """)
    return


@app.cell
def _(
    METRIC_COLUMNS,
    METRIC_LABELS,
    SIGNAL_BAND,
    np,
    pd,
    plt,
    synthetic_summary,
):
    def _signal_vs_confounder(summary):
        lo, hi = SIGNAL_BAND
        band = summary[(summary["sweep_kind"] == "distortion")
                       & (summary["param_value"] >= lo)
                       & (summary["param_value"] <= hi)]
        signal = {m: float(band[m].max() - band[m].min()) for m in METRIC_COLUMNS}
        confounders = ("n", "clusters", "sep", "dim", "imbalance")
        rows = []
        for m in METRIC_COLUMNS:
            row = {"metric": m, "signal_span": round(signal[m], 4)}
            for c in confounders:
                part = summary[(summary["sweep_kind"] == "confounder")
                               & (summary["param_name"] == c)]
                span = float(part[m].max() - part[m].min())
                row[f"{c}_ratio"] = (span / signal[m]) if signal[m] > 1e-6 else np.inf
            rows.append(row)
        return pd.DataFrame(rows)

    def _ratio_heatmap(ratio_df):
        ratio_cols = [c for c in ratio_df.columns if c.endswith("_ratio")]
        mat = ratio_df[ratio_cols].to_numpy()
        labels = [METRIC_LABELS[m] for m in ratio_df["metric"]]
        fig, ax = plt.subplots(figsize=(7.8, 5.2))
        im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0.0, vmax=1.5, aspect="auto")
        ax.set_xticks(range(len(ratio_cols)))
        ax.set_xticklabels([c.replace("_ratio", "") for c in ratio_cols])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                txt = "∞" if np.isinf(v) else f"{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="white" if (np.isinf(v) or v > 0.9) else "black")
        ax.set_title("confounder span / distortion span\n(green robust · red confounded)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="ratio")
        fig.tight_layout()
        return fig

    signal_vs_confounder = _signal_vs_confounder(synthetic_summary)
    ratio_heatmap_fig = _ratio_heatmap(signal_vs_confounder)
    signal_vs_confounder
    return (ratio_heatmap_fig,)


@app.cell
def _(ratio_heatmap_fig):
    ratio_heatmap_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### R_NX curves

    Raw $R_{NX}(K)$ curves vs relative neighbourhood size $u=K/n$, with normalized
    and $\int R_{NX}\, d\log u$ summary scores. Left: dropout sweep at default $n$;
    right: dataset-size sweep at fixed dropout.
    """)
    return


@app.cell
def _(
    FIXED_DROPOUT_RATE,
    SWEEP_N_VALUES,
    degrade,
    embedding_pair_rnx_curve,
    make_blobs,
    plt,
):
    def _plot_rnx_panel(ax, pairs, *, title, xlabel):
        for label, ref, man in pairs:
            out = embedding_pair_rnx_curve(ref, man)
            rnx, u, idx = out["rnx"], out["u"], out["norm_idx"]
            ax.plot(
                u,
                rnx,
                lw=2,
                label=(
                    f"{label}, norm={out['rnx_auc_norm']:.3f}, "
                    f"logu={out['rnx_auc_logu']:.3f}"
                ),
            )
            ax.scatter(u[idx], rnx[idx], s=30)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$R_{NX}(K)$")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    ref = make_blobs(seed=0)
    dropout_pairs = [
        (f"p={p}", ref, degrade(ref, p, seed=100))
        for p in (0.0, 0.1, 0.3, 0.7, 0.95)
    ]
    size_pairs = []
    for n_val in SWEEP_N_VALUES:
        ref_n = make_blobs(n=int(n_val), seed=0)
        size_pairs.append(
            (f"n={n_val}", ref_n, degrade(ref_n, FIXED_DROPOUT_RATE, seed=100))
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_rnx_panel(
        axes[0],
        dropout_pairs,
        title=r"$R_{NX}$ vs dropout (default $n$)",
        xlabel=r"relative neighbourhood size $u=K/n$",
    )
    _plot_rnx_panel(
        axes[1],
        size_pairs,
        title=rf"$R_{{NX}}$ vs dataset size ($p={FIXED_DROPOUT_RATE}$)",
        xlabel=r"relative neighbourhood size $u=K/n$",
    )
    fig.suptitle(
        r"$R_{NX}$ curves · markers = normalized-AUC sample points",
        y=1.02,
    )
    fig.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
