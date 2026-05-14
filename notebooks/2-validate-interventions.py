import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Validate interventions
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
    import scipy.sparse as sp
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.figsize"] = (8, 4.5)

    MANIP_DIR = Path(
        "/vault/amoneim/scfm-controlled-manipulations/results/tabula_sapiens_kidney/manipulations"
    )
    REF_PATH = Path(
        "/vault/amoneim/scfm-controlled-manipulations/raw_datasets/tabula_sapiens_kidney.h5ad"
    )
    return MANIP_DIR, REF_PATH, ad, mo, np, pd, plt, sns, sp


@app.cell
def _(MANIP_DIR, REF_PATH, ad):
    ref = ad.read_h5ad(REF_PATH)

    manip_files = sorted(MANIP_DIR.glob("*.h5ad"))
    by_family = {}
    for _path in manip_files:
        _family = "_".join(_path.stem.split("_")[:-1])
        by_family.setdefault(_family, []).append(_path)

    print(f"Reference: {ref.shape}, layers={list(ref.layers.keys())}")
    print(f"\nFound {len(manip_files)} manipulations across {len(by_family)} families:")
    for _family, _paths in by_family.items():
        print(f"  {_family}: {len(_paths)}")
    return by_family, ref


@app.cell
def _(ad, by_family, np, pd):
    rows = []
    for _family, _paths in by_family.items():
        for _path in _paths:
            _adata = ad.read_h5ad(_path, backed="r")
            _uns_keys = list(_adata.uns.get("scfm_intervention", {}).keys())
            _params = _adata.uns.get("scfm_intervention", {})
            _intervention_name = _uns_keys[0] if _uns_keys else "?"
            _params_dict = _params.get(_intervention_name, {})
            _params_brief = {
                _key: _value
                for _key, _value in _params_dict.items()
                if not isinstance(_value, (list, np.ndarray))
                or (hasattr(_value, "__len__") and len(_value) < 10)
            }
            rows.append(
                {
                    "family": _family,
                    "file": _path.name,
                    "n_obs": _adata.n_obs,
                    "n_vars": _adata.n_vars,
                    **{f"param_{_key}": _value for _key, _value in _params_brief.items()},
                }
            )
            _adata.file.close()

    params_df = pd.DataFrame(rows)
    params_df
    return


@app.cell
def _(REF_PATH, ad, by_family, np, pd, ref, sp):
    def qc_one(path, ref):
        _adata = ad.read_h5ad(path)
        _x = _adata.X

        if sp.issparse(_x):
            _counts = np.asarray(_x.sum(axis=1)).flatten()
            _ngenes = np.asarray((_x > 0).sum(axis=1)).flatten()
            _min_value = _x.data.min() if _x.nnz else 0
            _integer_counts = _x.data.dtype.kind in "iu" or np.all(_x.data == np.round(_x.data))
        else:
            _counts = _x.sum(axis=1)
            _ngenes = (_x > 0).sum(axis=1)
            _min_value = _x.min()
            _integer_counts = _x.dtype.kind in "iu" or np.all(_x == np.round(_x))

        return {
            "file": path.name,
            "shape_ok": (_adata.n_obs == ref.n_obs) and (_adata.n_vars == ref.n_vars),
            "obs_aligned": _adata.obs_names.equals(ref.obs_names),
            "median_counts": float(np.median(_counts)),
            "median_ngenes": float(np.median(_ngenes)),
            "min_value": float(_min_value),
            "integer_counts": bool(_integer_counts),
        }

    ref_qc = qc_one(REF_PATH, ref)
    ref_qc["family"] = "REFERENCE"

    qc_rows = [ref_qc]
    for _family, _paths in by_family.items():
        for _path in _paths:
            _row = qc_one(_path, ref)
            _row["family"] = _family
            qc_rows.append(_row)

    qc_df = pd.DataFrame(qc_rows)
    qc_df
    return (qc_df,)


@app.cell
def _(plt, qc_df, sns):
    fig_counts, _axis_counts = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=qc_df, x="family", y="median_counts", ax=_axis_counts)
    _axis_counts.set_title("Median library size per manipulation, grouped by family")
    _axis_counts.set_yscale("log")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(plt, qc_df, sns):
    fig_genes, _axis_genes = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=qc_df, x="family", y="median_ngenes", ax=_axis_genes)
    _axis_genes.set_title("Median number of detected genes per cell, by family")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(ad, by_family, np, plt, ref, sp):
    # Detection rate vs intervention parameter, for poisson and smoothing
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, family in zip(axes, ["poisson_resampling", "local_smoothing"]):
        param_key = "iterations" if family == "poisson_resampling" else "k"
        points = []
        for p in by_family.get(family, []):
            a = ad.read_h5ad(p)
            params = a.uns.get("scfm_intervention", {}).get(family, {})
            param_val = params.get(param_key)
            ngenes = np.asarray((a.X > 0).sum(axis=1)).flatten() if sp.issparse(a.X) else (a.X > 0).sum(axis=1)
            points.append((param_val, float(np.median(ngenes))))
    
        # Reference baseline
        ref_ngenes = np.asarray((ref.X > 0).sum(axis=1)).flatten() if sp.issparse(ref.X) else (ref.X > 0).sum(axis=1)
        ax.axhline(np.median(ref_ngenes), color="k", ls="--", label="reference")
    
        if points:
            points.sort()
            xs, ys = zip(*points)
            ax.plot(xs, ys, "o-", lw=2, ms=8)
    
        ax.set_xlabel(param_key)
        ax.set_ylabel("median detected genes per cell")
        ax.set_title(family)
        ax.legend()

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(ad, by_family, np, plt, ref, sp):
    _n_families = max(1, len(by_family))
    _ncols = min(3, _n_families)
    _nrows = int(np.ceil(_n_families / _ncols))
    fig_scatter, axes_scatter = plt.subplots(
        _nrows,
        _ncols,
        figsize=(5 * _ncols, 4 * _nrows),
        sharex=True,
        sharey=True,
    )
    axes_scatter = np.atleast_1d(axes_scatter).flatten()

    ref_counts = (
        np.asarray(ref.X.sum(axis=1)).flatten() if sp.issparse(ref.X) else ref.X.sum(axis=1)
    )

    for _axis, (_family, _paths) in zip(axes_scatter, by_family.items()):
        for _path in _paths:
            _adata = ad.read_h5ad(_path)
            _variant_counts = (
                np.asarray(_adata.X.sum(axis=1)).flatten()
                if sp.issparse(_adata.X)
                else _adata.X.sum(axis=1)
            )
            _axis.scatter(ref_counts, _variant_counts, s=2, alpha=0.15)
        _max_count = ref_counts.max()
        _axis.plot([0, _max_count], [0, _max_count], "k--", lw=1, alpha=0.7)
        _axis.set_xscale("log")
        _axis.set_yscale("log")
        _axis.set_title(_family)
        _axis.set_xlabel("ref counts")
        _axis.set_ylabel("variant counts")

    for _axis in axes_scatter[len(by_family) :]:
        _axis.axis("off")

    plt.suptitle("Per-cell total counts: variant vs reference", y=1.02)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(ad, by_family, np, ref, sp):
    gene_shuffle_files = by_family.get("gene_shuffle", [])

    print("Gene shuffle X-identity check (X must be byte-identical to reference):")
    for _path in gene_shuffle_files:
        _adata = ad.read_h5ad(_path)
        if sp.issparse(_adata.X) and sp.issparse(ref.X):
            _same = (_adata.X != ref.X).nnz == 0
        else:
            _same = np.array_equal(
                np.asarray(_adata.X.todense() if sp.issparse(_adata.X) else _adata.X),
                np.asarray(ref.X.todense() if sp.issparse(ref.X) else ref.X),
            )
        _var_diff = (_adata.var.index != ref.var.index).sum()
        print(f"  {_path.name}: X identical={_same}, var labels differ at {_var_diff} positions")
    return


@app.cell
def _(ad, by_family, np, ref, sp):
    smooth_files = by_family.get("local_smoothing", [])

    print("Local smoothing checks:")
    for _path in smooth_files:
        _adata = ad.read_h5ad(_path)
        _uns = _adata.uns.get("scfm_intervention", {}).get("local_smoothing", {})
        _has_op = "operator_indices" in _uns or "operator_data" in _uns

        _ref_var = (
            np.var(np.asarray(ref.X.sum(axis=1)).flatten())
            if sp.issparse(ref.X)
            else np.var(ref.X.sum(axis=1))
        )
        _variant_var = (
            np.var(np.asarray(_adata.X.sum(axis=1)).flatten())
            if sp.issparse(_adata.X)
            else np.var(_adata.X.sum(axis=1))
        )
        _narrowed = _variant_var < _ref_var

        print(
            f"  {_path.name}: operator stored={_has_op}, "
            f"libsize variance narrowed={_narrowed} "
            f"(ref var={_ref_var:.0f}, variant var={_variant_var:.0f})"
        )
    return


@app.cell
def _(qc_df):
    failures = qc_df[
        (~qc_df["shape_ok"])
        | (~qc_df["obs_aligned"])
        | (~qc_df["integer_counts"])
        | (qc_df["min_value"] < 0)
    ]
    if len(failures) > 0:
        print(f"FAILURES: {len(failures)} files failed validation")
    else:
        print(f"All {len(qc_df) - 1} manipulation files passed basic validation.")
    failures
    return


if __name__ == "__main__":
    app.run()
