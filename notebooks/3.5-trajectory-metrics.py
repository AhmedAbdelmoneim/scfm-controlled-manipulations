import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Trajectory metrics

    Standard (dyneval-style) trajectory inference metrics, scored against a known
    reference ordering held in `adata.obs['trajectory']` (integer milestones, e.g.
    0, 1, 2, 3, 4). The embedding under evaluation is in `adata.X`.

    Pseudotime is inferred with **scanpy diffusion pseudotime (DPT)**, the de facto
    standard estimator for reading a pseudotemporal ordering off an embedding
    (Haghverdi et al. 2016, Wolf et al. 2019). Pipeline: `pp.neighbors` on `.X`
    then `tl.diffmap` then `tl.dpt`, rooted at the earliest reference milestone.
    DPT is run in pure pseudotime mode (`n_branchings=0`); branch detection, if
    needed, is handled separately via PAGA per scanpy's own recommendation.

    Metric set, is **Ordering correlation** (Spearman). DPT pseudotime vs reference order.
       This is dyneval's `correlation` metric and the primary readout for mostly-linear
       trajectories.

    Every metric carries a permutation null: the reference order is shuffled and the
    metric recomputed, so each score is interpretable on the same footing as the other
    controlled manipulations.
    """)
    return


@app.cell
def _():
    import warnings
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from scipy.stats import spearmanr, kendalltau
    from scipy.optimize import linear_sum_assignment
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import KMeans
    from sklearn.metrics import f1_score

    # DPT internals are chatty on small / disconnected graphs; silence expected warnings.
    warnings.filterwarnings("ignore")
    sc.settings.verbosity = 0

    DATASET = Path(
        "/vault/amoneim/scfm-controlled-manipulations/5.1-reference-embeddings/ebdata/scimilarity/scimilarity_reference.h5ad"
    )

    TRAJ_KEY = "trajectory"   # obs column holding the integer reference order
    BRANCH_KEY = None         # set to an obs column name if a branch label exists, else None
    N_NEIGHBORS = 15          # kNN graph size for DPT and local readouts
    N_DCS = 10                # diffusion components used by DPT
    N_PERM = 10              # permutation null replicates
    SEED = 0
    return (
        DATASET,
        N_DCS,
        N_NEIGHBORS,
        N_PERM,
        SEED,
        TRAJ_KEY,
        mo,
        np,
        pd,
        sc,
        spearmanr,
    )


@app.cell
def _(DATASET, sc):
    adata = sc.read_h5ad(DATASET)
    return (adata,)


@app.cell
def _(adata):
    adata.obs
    return


@app.cell
def _(TRAJ_KEY, adata, np):
    # Pull embedding and reference order. .X is the embedding by construction.
    _X = adata.X
    emb = np.asarray(_X.todense()) if hasattr(_X, "todense") else np.asarray(_X)
    emb = emb.astype(np.float64)

    ref_raw = adata.obs[TRAJ_KEY].values
    ref = np.asarray(ref_raw).astype(float)

    # Drop cells with missing reference order; keep an index map back to adata.
    _valid = np.isfinite(ref)
    emb = emb[_valid]
    ref = ref[_valid].astype(int)
    valid_idx = np.where(_valid)[0]

    milestones = np.unique(ref)
    n_milestones = len(milestones)
    print(f"cells={emb.shape[0]}  dims={emb.shape[1]}  milestones={list(milestones)}")
    return emb, milestones, ref


@app.cell
def _(np, sc):
    import anndata as _ad

    def dpt_pseudotime(emb, ref, milestones, k, n_dcs):
        """scanpy diffusion pseudotime on the embedding.

        Builds a fresh AnnData whose .X is the embedding, runs neighbors -> diffmap ->
        dpt, rooted at the centroid-nearest cell of the earliest reference milestone.
        Returns pseudotime and a finite mask. DPT returns inf for cells in a graph
        component disconnected from the root; those are excluded from scoring and show
        up as frac_connected < 1, which is the diagnostic to raise k on sparse datasets.
        """
        a = _ad.AnnData(X=emb.astype(np.float32))
        sc.pp.neighbors(a, n_neighbors=k, use_rep="X")
        sc.tl.diffmap(a, n_comps=max(n_dcs, 10))

        first = milestones.min()
        m0 = np.where(ref == first)[0]
        centroid0 = emb[m0].mean(axis=0)
        root = int(m0[np.argmin(np.linalg.norm(emb[m0] - centroid0, axis=1))])
        a.uns["iroot"] = root

        sc.tl.dpt(a, n_dcs=n_dcs, n_branchings=0)
        pt = a.obs["dpt_pseudotime"].values.astype(float)
        finite = np.isfinite(pt)
        return pt, finite

    return (dpt_pseudotime,)


@app.cell
def _(np, spearmanr):
    def ordering_correlation(pseudotime, ref, finite):
        """dyneval `correlation`: monotone agreement between inferred pseudotime and
        reference order. Spearman is primary; Kendall reported alongside as it is more
        robust to ties in a discrete reference. Absolute value is taken because the
        direction of DPT pseudotime is arbitrary relative to the milestone labelling."""
        pt = pseudotime[finite]
        rr = ref[finite]
        if len(pt) < 3 or np.unique(pt).size < 2 or np.unique(rr).size < 2:
            return {"spearman": np.nan, "frac_connected": float(finite.mean())}
        rho = spearmanr(pt, rr).correlation
        return {
            "spearman": float(abs(rho)) if np.isfinite(rho) else np.nan,
            "frac_connected": float(finite.mean()),
        }

    return (ordering_correlation,)


@app.cell
def _(dpt_pseudotime, ordering_correlation):
    # Explicit metric registry. No auto-discovery: each metric is a named callable that
    # takes (emb, ref, milestones, k, n_dcs, seed) and returns a dict of scalar scores
    # plus optional per-cell arrays.
    def _correlation_metric(emb, ref, milestones, k, n_dcs, seed):
        pt, finite = dpt_pseudotime(emb, ref, milestones, k, n_dcs)
        return ordering_correlation(pt, ref, finite)

    METRIC_REGISTRY = {
        "ordering_correlation": _correlation_metric,
    }

    # Scalar keys used for the permutation null (per-cell arrays are excluded from nulls).
    NULL_SCALAR_KEYS = {
        "ordering_correlation": ["spearman"],
    }
    return METRIC_REGISTRY, NULL_SCALAR_KEYS


@app.cell
def _(METRIC_REGISTRY, N_DCS, N_NEIGHBORS, SEED, emb, milestones, ref):
    # Observed scores.
    observed = {}
    for _name, _fn in METRIC_REGISTRY.items():
        observed[_name] = _fn(emb, ref, milestones, N_NEIGHBORS, N_DCS, SEED)
    observed
    return (observed,)


@app.cell
def _(
    METRIC_REGISTRY,
    NULL_SCALAR_KEYS,
    N_DCS,
    N_NEIGHBORS,
    N_PERM,
    SEED,
    emb,
    milestones,
    np,
    observed,
    ref,
):
    # Permutation null: shuffle the reference order, recompute scalar scores. For the
    # correlation metric this also moves the DPT root (it is chosen from the shuffled
    # earliest milestone), so the null reflects scoring this embedding against a random
    # ordering end to end. This is the reference baseline used across the manipulation
    # metrics.
    def permutation_null(metric_name, n_perm, seed):
        fn = METRIC_REGISTRY[metric_name]
        keys = NULL_SCALAR_KEYS[metric_name]
        rng = np.random.default_rng(seed)
        out = {key: np.empty(n_perm) for key in keys}
        for p in range(n_perm):
            ref_shuf = rng.permutation(ref)
            res = fn(emb, ref_shuf, milestones, N_NEIGHBORS, N_DCS, seed)
            for key in keys:
                out[key][p] = res[key]
        return out

    null_dist = {}
    null_summary = {}
    for _name in METRIC_REGISTRY:
        _nd = permutation_null(_name, N_PERM, SEED)
        null_dist[_name] = _nd
        for _key, _vals in _nd.items():
            _obs = observed[_name][_key]
            _vals_finite = _vals[np.isfinite(_vals)]
            # one-sided empirical p: P(null >= observed), with +1 smoothing
            _p = (np.sum(_vals_finite >= _obs) + 1) / (len(_vals_finite) + 1)
            null_summary[f"{_name}.{_key}"] = {
                "observed": float(_obs),
                "null_mean": float(np.nanmean(_vals)),
                "null_std": float(np.nanstd(_vals)),
                "z": float((_obs - np.nanmean(_vals)) / (np.nanstd(_vals) + 1e-12)),
                "p_value": float(_p),
            }
    return (null_summary,)


@app.cell
def _(null_summary, pd):
    summary_df = pd.DataFrame(null_summary).T
    summary_df = summary_df[["observed", "null_mean", "null_std", "z", "p_value"]]
    summary_df
    return


if __name__ == "__main__":
    app.run()
