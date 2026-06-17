import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Neighbourhood metrics — paired ref vs manipulation

    Loads **one intervention** from the size-sweep vault and builds **paired** ref/man
    embedding arrays at nested subsample sizes (same cell indices at each *n*).

      Edit constants in the setup cell to change atlas, run, intervention, or model.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd

    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO_ROOT))

    VAULT = Path("/vault/amoneim/scfm-controlled-manipulations")
    SIZE_SWEEP_ROOT = VAULT / "processed_size_sweep"

    # --- edit these to point at a different run / intervention ---
    SWEEP_ATLAS = "retina"
    RUN_ID = "n20000_s0"  # needs n_cells >= max(SUBSAMPLE_SIZES)
    MODEL = "scimilarity"
    INTERVENTION_NAME = "gene_shuffle"
    INTERVENTION_ID = None  # None → first manipulation matching INTERVENTION_NAME
    REFERENCE_INTERVENTION_ID = "reference"

    SUBSAMPLE_SIZES = (200, 500, 1000, 2000, 5000, 10000)
    SUBSAMPLE_SEED = 0  # fixed seed → reproducible nested cell indices
    return (
        INTERVENTION_ID,
        INTERVENTION_NAME,
        MODEL,
        REFERENCE_INTERVENTION_ID,
        RUN_ID,
        SIZE_SWEEP_ROOT,
        SUBSAMPLE_SEED,
        SUBSAMPLE_SIZES,
        SWEEP_ATLAS,
        mo,
        np,
        pd,
    )


@app.cell
def _(
    INTERVENTION_ID,
    INTERVENTION_NAME,
    MODEL,
    REFERENCE_INTERVENTION_ID,
    RUN_ID,
    SIZE_SWEEP_ROOT,
    SUBSAMPLE_SEED,
    SUBSAMPLE_SIZES,
    SWEEP_ATLAS,
    np,
):
    from scfm_controlled_manipulations.evaluation.context import (
        load_dataset_context,
        load_intervention_bundle,
        load_model_context,
    )

    def _resolve_intervention_id(results_dir, intervention_name: str, intervention_id: str | None):
        if intervention_id is not None:
            path = results_dir / "manipulations" / f"{intervention_id}.h5ad"
            if not path.is_file():
                raise FileNotFoundError(path)
            return intervention_id
        matches = sorted(results_dir.glob(f"manipulations/{intervention_name}_*.h5ad"))
        if not matches:
            raise FileNotFoundError(
                f"No manipulation h5ad for {intervention_name!r} under {results_dir / 'manipulations'}"
            )
        return matches[0].stem

    def load_paired_bundle(
        *,
        sweep_root,
        atlas: str,
        run_id: str,
        model: str,
        intervention_name: str,
        intervention_id: str | None = None,
        reference_id: str = "reference",
    ):
        processed_root = sweep_root / atlas / run_id
        results_dir = processed_root / "results"
        embeddings_root = processed_root / "embeddings"
        if not results_dir.is_dir():
            raise FileNotFoundError(results_dir)

        iid = _resolve_intervention_id(results_dir, intervention_name, intervention_id)
        dataset_ctx = load_dataset_context(results_dir)
        model_ctx = load_model_context(
            embeddings_root, model, reference_id, dataset_ctx.obs.index
        )
        bundle = load_intervention_bundle(
            dataset_ctx=dataset_ctx,
            model_ctx=model_ctx,
            results_dir=results_dir,
            embeddings_root=embeddings_root,
            model=model,
            intervention_id=iid,
        )
        meta = {
            "atlas": atlas,
            "run_id": run_id,
            "model": model,
            "intervention_name": intervention_name,
            "intervention_id": iid,
            "n_cells": int(bundle.emb_ref.shape[0]),
            "embedding_dim": int(bundle.emb_ref.shape[1]),
            "results_dir": str(results_dir),
            "embeddings_root": str(embeddings_root),
        }
        return bundle, meta

    def nested_paired_subsamples(
        emb_ref: np.ndarray,
        emb_man: np.ndarray,
        sizes: tuple[int, ...],
        *,
        seed: int = 0,
    ) -> tuple[dict[int, np.ndarray], dict[int, tuple[np.ndarray, np.ndarray]]]:
        """Same cell indices for ref and man at each size; sizes are nested (prefix of one draw)."""
        n_cells = int(emb_ref.shape[0])
        max_n = max(int(s) for s in sizes)
        if max_n > n_cells:
            raise ValueError(f"max subsample {max_n} exceeds n_cells={n_cells}")

        rng = np.random.default_rng(seed)
        base_idx = np.sort(rng.choice(n_cells, size=max_n, replace=False))

        indices_by_n: dict[int, np.ndarray] = {}
        paired_emb: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for n in sorted(int(s) for s in sizes):
            idx = base_idx[:n]
            indices_by_n[n] = idx
            paired_emb[n] = (emb_ref[idx].copy(), emb_man[idx].copy())
        return indices_by_n, paired_emb

    bundle, meta = load_paired_bundle(
        sweep_root=SIZE_SWEEP_ROOT,
        atlas=SWEEP_ATLAS,
        run_id=RUN_ID,
        model=MODEL,
        intervention_name=INTERVENTION_NAME,
        intervention_id=INTERVENTION_ID,
        reference_id=REFERENCE_INTERVENTION_ID,
    )

    emb_ref = np.asarray(bundle.emb_ref, dtype=np.float32)
    emb_man = np.asarray(bundle.emb_man, dtype=np.float32)
    obs = bundle.obs.copy()

    subsample_indices, paired_emb = nested_paired_subsamples(
        emb_ref,
        emb_man,
        SUBSAMPLE_SIZES,
        seed=SUBSAMPLE_SEED,
    )

    emb_ref_by_n = {n: paired_emb[n][0] for n in paired_emb}
    emb_man_by_n = {n: paired_emb[n][1] for n in paired_emb}
    obs_by_n = {n: obs.iloc[subsample_indices[n]].copy() for n in subsample_indices}
    return (
        emb_man,
        emb_man_by_n,
        emb_ref,
        emb_ref_by_n,
        meta,
        subsample_indices,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Loaded arrays

    | Name | Description |
    |------|-------------|
    | `emb_ref`, `emb_man` | Full aligned embedding matrices `(n_cells, dim)` |
    | `paired_emb[n]` | `(ref, man)` tuple at nested subsample size *n* |
    | `emb_ref_by_n[n]`, `emb_man_by_n[n]` | Same arrays, keyed by *n* |
    | `subsample_indices[n]` | Cell indices into the full run (shared ref/man) |
    | `obs_by_n[n]` | `obs` rows for subsample *n* |
    | `meta` | Run / intervention metadata dict |
    """)
    return


@app.cell
def _(
    SUBSAMPLE_SIZES,
    emb_man_by_n,
    emb_ref_by_n,
    meta,
    pd,
    subsample_indices,
):
    summary = pd.DataFrame(
        [
            {
                "n": n,
                "emb_ref_shape": emb_ref_by_n[n].shape,
                "emb_man_shape": emb_man_by_n[n].shape,
                "index_min": int(subsample_indices[n].min()),
                "index_max": int(subsample_indices[n].max()),
            }
            for n in SUBSAMPLE_SIZES
        ]
    )
    print(
        f"{meta['atlas']} · {meta['run_id']} · {meta['model']} · "
        f"{meta['intervention_name']} ({meta['intervention_id']}) · "
        f"full n={meta['n_cells']} dim={meta['embedding_dim']}"
    )
    summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Metric sandbox

    Add metric cells below. Example access at *n*=500:

    ```python
    ref, man = paired_emb[500]
    # or: emb_ref_by_n[500], emb_man_by_n[500]
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 1 · Rank-based core (co-ranking → T, C, R_NX)

    The co-ranking matrix `Q` cross-tabulates neighbour ranks in the reference vs
    manipulated embedding. Trustworthiness, continuity, and R_NX are read off it
    (T&C via sklearn; R_NX from `Q` directly).
    """)
    return


@app.cell
def _(np):
    from scipy.spatial.distance import squareform, pdist

    def _rank_matrix(emb: np.ndarray) -> np.ndarray:
        d = squareform(pdist(emb.astype(np.float64)))
        n = d.shape[0]
        np.fill_diagonal(d, np.inf)
        order = np.argsort(d, axis=1, kind="stable")
        ranks = np.empty_like(order)
        rows = np.arange(n)[:, None]
        ranks[rows, order] = np.arange(n)[None, :]
        return ranks

    def coranking_matrix(emb_ref: np.ndarray, emb_man: np.ndarray) -> np.ndarray:
        """Q[k, l] = #pairs with reference rank k+1 and manipulated rank l+1.
        Size (n-1, n-1). Framing-agnostic: ref-vs-man is one use."""
        n = emb_ref.shape[0]
        rk_ref = _rank_matrix(emb_ref)
        rk_man = _rank_matrix(emb_man)
        mask = ~np.eye(n, dtype=bool)
        kk = rk_ref[mask].ravel()
        ll = rk_man[mask].ravel()
        valid = (kk < n - 1) & (ll < n - 1)
        Q = np.zeros((n - 1, n - 1), dtype=np.float64)
        np.add.at(Q, (kk[valid], ll[valid]), 1.0)
        return Q

    return coranking_matrix, pdist, squareform


@app.cell
def _(np):
    def q_nx(Q: np.ndarray) -> np.ndarray:
        """Fraction of K-neighbours preserved (top-left KxK block). K = 1..n-1."""
        n_minus_1 = Q.shape[0]
        n = n_minus_1 + 1
        csum = Q.cumsum(0).cumsum(1)
        K = np.arange(1, n_minus_1 + 1)
        return np.diagonal(csum) / (K * n)

    def r_nx(Q: np.ndarray) -> np.ndarray:
        """Baseline-corrected, rescaled neighbourhood preservation. K = 1..n-2.
        R_NX(K) = 1 perfect, 0 random. Removes the K/(n-1) chance floor."""
        n = Q.shape[0] + 1
        qnx = q_nx(Q)[:-1]
        K = np.arange(1, n - 1)
        return ((n - 1) * qnx - K) / (n - 1 - K)

    def auc_raw(curve: np.ndarray) -> float:
        """Non-normalized: log-AUC over absolute K. Drifts with n. Reference only."""
        K = np.arange(1, len(curve) + 1)
        w = 1.0 / K
        return float((curve * w).sum() / w.sum())

    def auc_norm(curve: np.ndarray, fracs=(0.01, 0.02, 0.05, 0.1, 0.2)) -> float:
        """Normalized: log-AUC over K as fractions of n. n-stable summary."""
        n = len(curve) + 1
        idx = np.array([max(0, min(len(curve) - 1, int(round(f * n)) - 1)) for f in fracs])
        return float((curve[idx] * (1.0 / np.array(fracs))).sum() / (1.0 / np.array(fracs)).sum())

    return auc_norm, r_nx


@app.cell
def _(np, pdist):
    from sklearn.manifold import trustworthiness as _sk_trust
    from scipy.stats import spearmanr as _spearmanr

    def trustworthiness(emb_ref, emb_man, k=15):
        """Penalizes intrusions. Saturation-prone: near 1 until near-total breakdown.
        Best as a breakdown detector and the directional T-vs-C diagnostic."""
        return float(_sk_trust(emb_ref, emb_man, n_neighbors=k))

    def continuity(emb_ref, emb_man, k=15):
        """Penalizes extrusions. = trustworthiness with arguments swapped."""
        return float(_sk_trust(emb_man, emb_ref, n_neighbors=k))

    def distance_correlation_spearman(emb_ref, emb_man):
        """Spearman of pairwise distances. The usable-range rank-of-distances variant."""
        dr = pdist(emb_ref.astype(np.float64))
        dm = pdist(emb_man.astype(np.float64))
        return float(_spearmanr(dr, dm).correlation)

    return continuity, distance_correlation_spearman, trustworthiness


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 2 · Representational-similarity core (CKA, DistCorr) + multi-scale variants

    From Khandait & Gerken (2026), *From Layers to Networks: Comparing Neural
    Representations via Diffusion Geometry*. CKA and (corrected) DistCorr are
    RSM-based similarity measures; the paper shows any centered, scale-invariant
    RSM measure can be rewritten as a row-stochastic **Markov matrix** via an affine
    shift-and-rescale, then powered to **t** to probe geometry at coarser scales.

    - **CKA** — linear-kernel Gram RSM, double-centered (H S H), aligned via HSIC.
    - **DistCorr** — pairwise **distance** RSM, **double-centered in rows AND columns**
      (Székely–Rizzo: subtract row mean, subtract column mean, add grand mean), then
      the normalized distance-covariance. This is the corrected form.
    - **MS-CKA(t), MS-DistCorr(t)** — same measures computed on the **t-th power** of
      the Markov-reformulated RSM. `t=1` exactly recovers standard CKA / DistCorr
      (paper Cor. 4.3/4.4); larger `t` shifts from direct pairwise affinity to
      multi-scale neighbourhood structure. This is the bounded, fixed-null,
      multi-scale replacement for a raw diffusion-KL readout.
    """)
    return


@app.cell
def _(np, pdist, squareform):
    # All underscore helpers are kept inside this single cell so marimo does not need
    # to pass private names between cells. Only the public functions are returned.
    def _double_center(S):
        """H S H with H = I - (1/N) 11^T : remove row + column means, add grand mean."""
        N = S.shape[0]
        H = np.eye(N) - np.ones((N, N)) / N
        return H @ S @ H

    def _linear_rsm(R):
        R = R.astype(np.float64)
        return R @ R.T

    def _distance_rsm(R):
        return squareform(pdist(R.astype(np.float64)))

    def _cka_on_rsm(S1, S2):
        """CKA = HSIC(S1,S2) / sqrt(HSIC(S1,S1) HSIC(S2,S2)), HSIC via H S H centering."""
        N = S1.shape[0]
        H = np.eye(N) - np.ones((N, N)) / N
        c = lambda A: H @ A @ H
        hsic = lambda A, B: np.trace(c(A) @ c(B)) / (N - 1) ** 2
        den = np.sqrt(hsic(S1, S1) * hsic(S2, S2))
        return float(hsic(S1, S2) / den) if den > 0 else 0.0

    def _distcorr_on_rsm(S1, S2):
        """Normalized distance covariance on double-centered RSMs."""
        A, B = _double_center(S1), _double_center(S2)
        N = A.shape[0]
        dc = lambda U, V: (U * V).sum() / N ** 2
        den = np.sqrt(dc(A, A) * dc(B, B))
        return float(dc(A, B) / den) if den > 0 else 0.0

    def _to_markov(S):
        """Affine shift-and-rescale of a centered RSM to a row-stochastic Markov matrix:
        shift to nonnegative, then row-normalize (Theorem 4.1 construction)."""
        A = S - S.min()
        rs = A.sum(1, keepdims=True)
        rs[rs == 0] = 1.0
        return A / rs

    def _markov_from(emb, kind):
        S = _linear_rsm(emb) if kind == "cka" else _distance_rsm(emb)
        return _to_markov(_double_center(S))

    def cka(emb_ref, emb_man):
        """Linear CKA (standard)."""
        return _cka_on_rsm(_linear_rsm(emb_ref), _linear_rsm(emb_man))

    def distcorr(emb_ref, emb_man):
        """Distance correlation (corrected, double-centered distance RSMs)."""
        return _distcorr_on_rsm(_distance_rsm(emb_ref), _distance_rsm(emb_man))

    def ms_cka(emb_ref, emb_man, t):
        """MS-CKA(t): CKA on the t-th power of the Markov-reformulated linear RSM.
        t=1 recovers standard CKA."""
        P1 = np.linalg.matrix_power(_markov_from(emb_ref, "cka"), t)
        P2 = np.linalg.matrix_power(_markov_from(emb_man, "cka"), t)
        return _cka_on_rsm(P1, P2)

    def ms_distcorr(emb_ref, emb_man, t):
        """MS-DistCorr(t): DistCorr on the t-th power of the Markov-reformulated
        distance RSM. t=1 recovers standard (corrected) DistCorr."""
        P1 = np.linalg.matrix_power(_markov_from(emb_ref, "dc"), t)
        P2 = np.linalg.matrix_power(_markov_from(emb_man, "dc"), t)
        return _distcorr_on_rsm(P1, P2)

    def ms_curve(emb_ref, emb_man, ts, kind="cka"):
        """Vector of MS-measure over a list of t values."""
        fn = ms_cka if kind == "cka" else ms_distcorr
        return np.array([fn(emb_ref, emb_man, int(_t)) for _t in ts])

    return cka, distcorr, ms_cka, ms_curve, ms_distcorr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Validation (rank + RSM metrics together)

    identity → all = 1. rescale → all = 1 (CKA/DistCorr scale-invariant; rank metrics
    ordinal-blind). random → all ≈ baseline. MS at t=1 must equal standard CKA/DistCorr.
    """)
    return


@app.cell
def _(
    auc_norm,
    cka,
    continuity,
    coranking_matrix,
    distance_correlation_spearman,
    distcorr,
    ms_cka,
    ms_distcorr,
    np,
    pd,
    r_nx,
    trustworthiness,
):
    def make_blobs(n=1500, dim=128, n_clusters=10, sep=8.0, sizes=None, seed=0):
        rng = np.random.default_rng(seed)
        centres = rng.normal(0, sep, size=(n_clusters, dim))
        if sizes is None:
            lab = rng.integers(0, n_clusters, n)
        else:
            p = np.asarray(sizes, dtype=float)
            p = p / p.sum()
            lab = rng.choice(n_clusters, size=n, p=p)
        return centres[lab] + rng.normal(0, 1, size=(n, dim))

    def degrade(emb, noise, seed=0):
        return emb + np.random.default_rng(seed).normal(0, noise, size=emb.shape)

    def _validation():
        base = make_blobs(n=400)  # small for speed (RSM metrics are O(n^2)-O(n^3))
        rng = np.random.default_rng(1)
        cases = {"identity": base.copy(), "rescale_x5": base * 5.0,
                 "random": rng.normal(0, 1, size=base.shape)}
        rows = []
        for _name, _man in cases.items():
            _rnx = r_nx(coranking_matrix(base, _man))
            rows.append({
                "case": _name,
                "T": round(trustworthiness(base, _man), 3),
                "C": round(continuity(base, _man), 3),
                "RNX_norm": round(auc_norm(_rnx), 3),
                "CKA": round(cka(base, _man), 3),
                "DistCorr": round(distcorr(base, _man), 3),
                "DC_spear": round(distance_correlation_spearman(base, _man), 3),
                "MS-CKA t1": round(ms_cka(base, _man, 1), 3),
                "MS-DC t1": round(ms_distcorr(base, _man, 1), 3),
            })
        return pd.DataFrame(rows)

    validation_table = _validation()
    validation_table
    return degrade, make_blobs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 3 · Multi-scale t-sweep (the diffusion-scale axis)

    MS-CKA(t) and MS-DistCorr(t) as t increases: small t = direct pairwise affinity
    (local), large t = coarser neighbourhood structure (global). The curve over t is
    the multi-scale signature — the bounded, fixed-null analogue of sweeping diffusion
    time. Shown for a few distortion magnitudes so you can see how the scale profile
    changes with severity.
    """)
    return


@app.cell
def _(degrade, make_blobs, ms_curve, np):
    import matplotlib.pyplot as plt

    def _t_sweep_figure(ts=(1, 2, 4, 8, 16, 32, 64), noises=(0.5, 1.0, 2.0, 4.0), seed=0):
        ref = make_blobs(n=400, seed=seed)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
        colors = plt.cm.viridis(np.linspace(0, 0.85, len(noises)))
        for _no, _col in zip(noises, colors):
            _man = degrade(ref, _no, seed=99)
            _cka_c = ms_curve(ref, _man, ts, kind="cka")
            _dc_c = ms_curve(ref, _man, ts, kind="dc")
            ax[0].plot(ts, _cka_c, "o-", color=_col, lw=1.6, label=f"σ={_no}")
            ax[1].plot(ts, _dc_c, "o-", color=_col, lw=1.6, label=f"σ={_no}")
        for _a, _title in [(ax[0], "MS-CKA(t)"), (ax[1], "MS-DistCorr(t)")]:
            _a.set_xscale("log", base=2)
            _a.set_xlabel("diffusion scale t"); _a.set_ylabel("similarity")
            _a.set_title(_title); _a.set_ylim(-0.05, 1.05); _a.legend(fontsize=8)
            _a.axhline(0, color="grey", ls="--", lw=0.8)
        fig.suptitle("Multi-scale t-sweep — scale profile per distortion magnitude", y=1.0)
        fig.tight_layout()
        return fig

    t_sweep_fig = _t_sweep_figure()
    t_sweep_fig
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 4 · Distortion sweep — each metric's dynamic range

    Vary the actual distortion (additive noise σ) and watch each metric. MS measures
    shown at a representative local t and global t. A metric is usable only if it has
    range here.
    """)
    return


@app.cell
def _(
    auc_norm,
    cka,
    continuity,
    coranking_matrix,
    degrade,
    distance_correlation_spearman,
    distcorr,
    make_blobs,
    ms_cka,
    ms_distcorr,
    pd,
    plt,
    r_nx,
    trustworthiness,
):
    def _distortion_sweep(noises=(0.1, 0.3, 0.6, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
                          seeds=(0, 1, 2), t_local=2, t_global=16):
        rows = []
        for _no in noises:
            for _s in seeds:
                _ref = make_blobs(n=400, seed=_s)
                _man = degrade(_ref, _no, seed=100 + _s)
                _rnx = r_nx(coranking_matrix(_ref, _man))
                rows.append({
                    "noise": _no, "seed": _s,
                    "T": trustworthiness(_ref, _man),
                    "C": continuity(_ref, _man),
                    "RNX_norm": auc_norm(_rnx),
                    "DC_spear": distance_correlation_spearman(_ref, _man),
                    "CKA": cka(_ref, _man),
                    "DistCorr": distcorr(_ref, _man),
                    f"MS-CKA t{t_local}": ms_cka(_ref, _man, t_local),
                    f"MS-CKA t{t_global}": ms_cka(_ref, _man, t_global),
                    f"MS-DC t{t_local}": ms_distcorr(_ref, _man, t_local),
                    f"MS-DC t{t_global}": ms_distcorr(_ref, _man, t_global),
                })
        df = pd.DataFrame(rows)
        return df.groupby("noise").mean().reset_index().drop(columns="seed")

    def _distortion_figure(df):
        metric_cols = [c for c in df.columns if c != "noise"]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for _c in metric_cols:
            _ls = "--" if ("t16" in _c or "DistCorr" == _c or _c == "CKA") else "-"
            ax.plot(df["noise"], df[_c], _ls, ms=3, lw=1.4, label=_c)
        ax.set_xscale("log")
        ax.set_xlabel("distortion (additive noise σ)"); ax.set_ylabel("metric")
        ax.set_title("Distortion sweep — usable metric = wide vertical range")
        ax.legend(fontsize=7, ncol=2); ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        return fig

    distortion_table = _distortion_sweep()
    distortion_fig = _distortion_figure(distortion_table)
    distortion_fig
    return (distortion_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 5 · Confounder sweeps (fixed distortion)

    Each sweep varies **one nuisance property** while the true distortion is **held
    constant** (additive noise σ=1.0). Any movement is the metric responding to the
    property, not to distortion — a confounder. Read against §4: a confounder matters
    only if it moves a metric comparably to actual distortion (quantified in §6).

    All metrics including CKA, DistCorr, and MS variants (local + global t). Realistic
    ranges: n to 5000, dim 128–2048, clusters 3–200.
    """)
    return


@app.cell
def _(
    auc_norm,
    cka,
    continuity,
    coranking_matrix,
    degrade,
    distance_correlation_spearman,
    distcorr,
    make_blobs,
    ms_cka,
    ms_distcorr,
    pd,
    r_nx,
    trustworthiness,
):
    _T_LOCAL = 2
    _T_GLOBAL = 16
    _METRIC_COLS = ["T", "C", "RNX_norm", "DC_spear", "CKA", "DistCorr",
                    f"MS-CKA t{_T_LOCAL}", f"MS-CKA t{_T_GLOBAL}",
                    f"MS-DC t{_T_LOCAL}", f"MS-DC t{_T_GLOBAL}"]

    def _one_point(ref, noise=1.0, dseed=99):
        man = degrade(ref, noise, seed=dseed)
        rnx = r_nx(coranking_matrix(ref, man))
        return {
            "T": trustworthiness(ref, man),
            "C": continuity(ref, man),
            "RNX_norm": auc_norm(rnx),
            "DC_spear": distance_correlation_spearman(ref, man),
            "CKA": cka(ref, man),
            "DistCorr": distcorr(ref, man),
            f"MS-CKA t{_T_LOCAL}": ms_cka(ref, man, _T_LOCAL),
            f"MS-CKA t{_T_GLOBAL}": ms_cka(ref, man, _T_GLOBAL),
            f"MS-DC t{_T_LOCAL}": ms_distcorr(ref, man, _T_LOCAL),
            f"MS-DC t{_T_GLOBAL}": ms_distcorr(ref, man, _T_GLOBAL),
        }

    def _sweep(values, build, seeds=(0, 1, 2)):
        """build(value, seed) -> ref embedding. Distortion held fixed (noise=1.0).
        n=400 substrate for RSM-metric tractability."""
        rows = []
        for _v in values:
            for _s in seeds:
                _ref = build(_v, _s)
                _m = _one_point(_ref, dseed=1000 + _s)
                rows.append({"value": _v, "seed": _s, **_m})
        df = pd.DataFrame(rows)
        return df.groupby("value")[_METRIC_COLS].mean().reset_index()

    # n=400 fixed for RSM tractability except the explicit n sweep
    sweep_n = _sweep((200, 300, 400, 600, 800),
                     lambda v, s: make_blobs(n=v, seed=s))
    sweep_clusters = _sweep((3, 10, 30, 75, 200),
                            lambda v, s: make_blobs(n=400, n_clusters=v, seed=s))
    sweep_sep = _sweep((2.0, 4.0, 8.0, 16.0, 32.0),
                       lambda v, s: make_blobs(n=400, sep=v, seed=s))
    sweep_dim = _sweep((128, 256, 512, 1024, 2048),
                       lambda v, s: make_blobs(n=400, dim=v, seed=s))
    sweep_imbalance = _sweep((1, 3, 10, 30, 100),
                             lambda v, s: make_blobs(n=400, n_clusters=10, sizes=[v] + [1] * 9, seed=s))
    return sweep_clusters, sweep_dim, sweep_imbalance, sweep_n, sweep_sep


@app.cell
def _(plt, sweep_clusters, sweep_dim, sweep_imbalance, sweep_n, sweep_sep):
    def _confounder_figure():
        panels = [
            (sweep_n, "dataset size n", True),
            (sweep_clusters, "cluster count", False),
            (sweep_sep, "cluster separation", False),
            (sweep_dim, "embedding dimensionality", True),
            (sweep_imbalance, "cluster-size imbalance (dominant:rest)", True),
        ]
        cols = [c for c in sweep_n.columns if c != "value"]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        _flat = axes.ravel()
        for _ax, (_df, _title, _logx) in zip(_flat, panels):
            for _c in cols:
                _ls = "--" if ("t16" in _c or _c in ("CKA", "DistCorr")) else "-"
                _ax.plot(_df["value"], _df[_c], _ls, ms=3, lw=1.3, label=_c)
            if _logx:
                _ax.set_xscale("log")
            _ax.set_xlabel(_title); _ax.set_ylabel("metric (fixed distortion)")
            _ax.set_title(_title, fontsize=10); _ax.set_ylim(0, 1.02)
            _ax.legend(fontsize=6, ncol=2)
        _flat[-1].axis("off")
        _flat[-1].text(0.0, 0.5,
            "Reading the curves\n"
            "(flat = robust to this property;\n"
            "interpret movement against §4 range):\n\n"
            "• cluster count: rank metrics RISE\n"
            "  (more clusters = tighter rel. gaps)\n"
            "• separation: distance metrics move,\n"
            "  rank metrics flat\n"
            "• imbalance: rank metrics FALL\n"
            "  (dominant cluster)\n"
            "• n: check drift per metric\n"
            "• dim: check creep over 128–2048\n\n"
            "dashed = global-t MS / unbounded refs\n"
            "(t16, CKA, DistCorr)",
            fontsize=8, va="center")
        fig.suptitle("Confounder sweeps — movement at fixed distortion = bias to be aware of", y=1.0)
        fig.tight_layout()
        return fig

    confounder_fig = _confounder_figure()
    confounder_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 6 · Signal vs confounder — the usability criterion

    A confounder only matters if it moves a metric **comparably to actual
    distortion**. Per metric: **signal range** = span across the realistic distortion
    sweep (§4, noise 0.3→8); **confounder range** = span across each nuisance sweep
    (§5, fixed distortion).

    `ratio = confounder_range / signal_range`. Small → controllable nuisance. Near or
    above 1 → the confounder moves the metric as much as real distortion does, so the
    metric is **unusable for cross-condition comparison** along that property. A metric
    with near-zero signal range (saturated) is unusable regardless.
    """)
    return


@app.cell
def _(
    distortion_table,
    pd,
    sweep_clusters,
    sweep_dim,
    sweep_imbalance,
    sweep_n,
    sweep_sep,
):
    def _signal_vs_confounder():
        # metrics present in BOTH the distortion table and the confounder sweeps
        metrics = [c for c in sweep_n.columns
                   if c != "value" and c in distortion_table.columns]

        dt = distortion_table[(distortion_table["noise"] >= 0.3)
                              & (distortion_table["noise"] <= 8.0)]
        signal = {m: float(dt[m].max() - dt[m].min()) for m in metrics}

        confounders = {
            "n": sweep_n, "clusters": sweep_clusters, "separation": sweep_sep,
            "dim": sweep_dim, "imbalance": sweep_imbalance,
        }
        rows = []
        for _m in metrics:
            _row = {"metric": _m, "signal_range": round(signal[_m], 3)}
            for _cname, _cdf in confounders.items():
                _crange = float(_cdf[_m].max() - _cdf[_m].min())
                _ratio = _crange / signal[_m] if signal[_m] > 1e-6 else float("inf")
                _row[f"{_cname}_ratio"] = round(_ratio, 2)
            rows.append(_row)
        return pd.DataFrame(rows)

    signal_vs_confounder = _signal_vs_confounder()
    signal_vs_confounder
    return (signal_vs_confounder,)


@app.cell
def _(plt, signal_vs_confounder):
    def _ratio_heatmap(df):
        ratio_cols = [c for c in df.columns if c.endswith("_ratio")]
        mat = df[ratio_cols].to_numpy()
        fig, ax = plt.subplots(figsize=(8, 5.5))
        im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=1.5, aspect="auto")
        ax.set_xticks(range(len(ratio_cols)))
        ax.set_xticklabels([c.replace("_ratio", "") for c in ratio_cols])
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["metric"])
        for _i in range(mat.shape[0]):
            for _j in range(mat.shape[1]):
                _v = mat[_i, _j]
                ax.text(_j, _i, "∞" if _v == float("inf") else f"{_v:.2f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if (_v > 0.9 or _v == float("inf")) else "black")
        ax.set_title("confounder_range / signal_range\n(green = controllable, red = confounder dominates)")
        fig.colorbar(im, ax=ax, fraction=0.046, label="ratio")
        fig.tight_layout()
        return fig

    ratio_heatmap_fig = _ratio_heatmap(signal_vs_confounder)
    ratio_heatmap_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 7 · Real-data n-sweep

    All metrics across subsample sizes × seeds on the loaded `emb_ref`/`emb_man`.
    Usable iff mean is flat and seed CV shrinks with n. MS measures at a local and a
    global t. Capped at n=2000 (RSM metrics are O(n²) memory and the Markov power is
    O(n³); raise if compute allows).
    """)
    return


@app.cell
def _(
    auc_norm,
    cka,
    continuity,
    coranking_matrix,
    distance_correlation_spearman,
    distcorr,
    emb_man,
    emb_ref,
    ms_cka,
    ms_distcorr,
    np,
    pd,
    r_nx,
    trustworthiness,
):
    def _real_n_sweep(emb_r, emb_m, sizes=(200, 500, 1000, 2000),
                      seeds=(0, 1, 2, 3, 4), t_local=2, t_global=16):
        n_cells = emb_r.shape[0]
        rows = []
        for _n in sizes:
            if _n > n_cells:
                continue
            for _s in seeds:
                _rng = np.random.default_rng(_s)
                _idx = _rng.choice(n_cells, size=_n, replace=False)
                _r, _m = emb_r[_idx], emb_m[_idx]
                _rnx = r_nx(coranking_matrix(_r, _m))
                rows.append({
                    "n": _n, "seed": _s,
                    "T": trustworthiness(_r, _m),
                    "C": continuity(_r, _m),
                    "RNX_norm": auc_norm(_rnx),
                    "DC_spear": distance_correlation_spearman(_r, _m),
                    "CKA": cka(_r, _m),
                    "DistCorr": distcorr(_r, _m),
                    f"MS-CKA t{t_local}": ms_cka(_r, _m, t_local),
                    f"MS-CKA t{t_global}": ms_cka(_r, _m, t_global),
                    f"MS-DC t{t_local}": ms_distcorr(_r, _m, t_local),
                    f"MS-DC t{t_global}": ms_distcorr(_r, _m, t_global),
                })
        return pd.DataFrame(rows)

    def _summary(df):
        cols = [c for c in df.columns if c not in ("n", "seed")]
        agg = {}
        for _c in cols:
            agg[f"{_c}_mean"] = (_c, "mean")
            agg[f"{_c}_cv"] = (_c, lambda x: x.std() / abs(x.mean()) if abs(x.mean()) > 1e-9 else 0.0)
        return df.groupby("n").agg(**agg).reset_index().round(4)

    real_sweep_raw = _real_n_sweep(emb_ref, emb_man)
    real_sweep_summary = _summary(real_sweep_raw)
    real_sweep_summary
    return (real_sweep_raw,)


@app.cell
def _(plt, real_sweep_raw):
    def _real_figure(df):
        cols = [c for c in df.columns if c not in ("n", "seed")]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for _c in cols:
            _ls = "--" if ("t16" in _c or _c in ("CKA", "DistCorr")) else "-"
            _mean = df.groupby("n")[_c].mean()
            ax.plot(_mean.index, _mean.values, _ls, marker="o", ms=4, lw=1.5, label=_c)
        ax.set_xscale("log")
        ax.set_xlabel("n"); ax.set_ylabel("metric (mean over seeds)")
        ax.set_title("Real-data n-sweep — flat = n-stable")
        ax.legend(fontsize=7, ncol=2); ax.set_ylim(0, 1.02)
        fig.tight_layout()
        return fig

    real_sweep_fig = _real_figure(real_sweep_raw)
    real_sweep_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7b · Real-data multi-scale t-sweep

    MS-CKA(t) and MS-DistCorr(t) on the loaded ref/man at a fixed n, to read the
    actual scale profile of this manipulation on real embeddings.
    """)
    return


@app.cell
def _(emb_man, emb_ref, ms_curve, np, plt):
    def _real_t_sweep(emb_r, emb_m, ts=(1, 2, 4, 8, 16, 32, 64), n=2000, seed=0):
        n_cells = emb_r.shape[0]
        _n = min(n, n_cells)
        _idx = np.random.default_rng(seed).choice(n_cells, size=_n, replace=False)
        _r, _m = emb_r[_idx], emb_m[_idx]
        cka_c = ms_curve(_r, _m, ts, kind="cka")
        dc_c = ms_curve(_r, _m, ts, kind="dc")
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.plot(ts, cka_c, "o-", color="C0", lw=1.8, label="MS-CKA")
        ax.plot(ts, dc_c, "s-", color="C3", lw=1.8, label="MS-DistCorr")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("diffusion scale t"); ax.set_ylabel("similarity")
        ax.set_title(f"Real-data multi-scale profile (n={_n})")
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=9)
        ax.axhline(0, color="grey", ls="--", lw=0.8)
        fig.tight_layout()
        return fig

    real_t_sweep_fig = _real_t_sweep(emb_ref, emb_man)
    real_t_sweep_fig
    return


if __name__ == "__main__":
    app.run()
