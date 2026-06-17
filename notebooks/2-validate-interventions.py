import marimo

__generated_with = "0.23.6"
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
        "/vault/amoneim/scfm-controlled-manipulations/processed/tabula_sapiens/results/manipulations"
    )
    REF_PATH = Path(
        "/vault/amoneim/scfm-controlled-manipulations/raw_datasets/tabula_sapiens.h5ad"
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


@app.function
def plot_reference_and_manipulations_grid(
    ref_path,
    manipulation_paths_by_family,
    ad,
    sp,
    plt,
    np,
    output_dir="manip_cell_gene_count_plots_grid",
    random_seed=0,
    rank_start=0,
    rank_end=10,
):
    """
    Combines reference and manipulations into a single plot.
    Layout: 2 columns x 3 rows (6 panels - 1 reference and 5 manipulations).
    Top left is reference cell, others are from manipulations (one per family).

    Shows genes ranked rank_start:rank_end by expression in the reference cell,
    looked up by gene identity (var_names) not position, so manipulations that
    reorder genes are handled correctly.

    For manipulation panels, overlays the reference at low opacity.
    Y-axis is shared across all panels.
    Saves as SVG and PNG.

    ------
    Why am I seeing no bars for the manipulations, only the reference?

    This could happen if, for the genes being plotted (selected_gene_names),
    there are *no corresponding genes* found in the manipulation AnnData's .var_names.
    In this code, for each manipulation, the relevant cell's counts are indexed
    by gene name, not by position. If a gene from the reference is not in the
    manipulated .var_names, a zero will be inserted in its place:

      manip_window = np.array([
          manip_counts_full[name_to_idx[g]] if g in name_to_idx else 0.0
          for g in selected_gene_names
      ], dtype=float)

    If all genes in selected_gene_names are missing in the manipulated file,
    then manip_window will be all zeros -- so the reference bar (overlay)
    will show, but not the manipulation.

    Possible reasons for this:
    - The manipulation process changed .var_names such that most or all of the
      top N genes in the reference (used for plotting) are *not present*
      in the manipulated files.
    - There is a mismatch between gene name formats (e.g., Ensembl IDs vs gene symbols).
    - The manipulated AnnData files have truncated, filtered, or otherwise reordered
      .var_names so that the mapping by name fails for the reference's top genes.
    - The assignment of selected_gene_names depends on the reference,
      and if these gene names do not exist in the manip AnnData, the
      result is zero bars.

    How to debug:
    - Print or check  selected_gene_names and manip_var_names in the function to ensure overlap.
        print("selected_gene_names:", selected_gene_names)
        print("manip_var_names:", manip_var_names)
    - Print or check [g for g in selected_gene_names if g not in manip_var_names]
      to see which genes are missing.
    - Check if the manipulation process (upstream) changes the semantics or
      format of .var_names or their order.
    - Check that .var_names_make_unique() does not further alter/rename
      gene names in a non-matching way.
    - Ensure that the .h5ad files contain overlapping gene names, and that
      their order and format is consistent.

    ------
    Parameters
    ----------
    ref_path : str or Path
        Path to the reference .h5ad file.
    manipulation_paths_by_family : dict
        Mapping from family name to list of .h5ad file paths. Only the first
        file per family is used.
    ad, sp, plt, np : modules
        anndata, scipy.sparse, matplotlib.pyplot, numpy.
    output_dir : str
        Output directory for saved figures.
    random_seed : int
        Seed for reproducible cell selection.
    rank_start : int
        Start of the gene rank window (0-indexed, descending expression order).
        Default 0. The top genes (0:10) give the strongest visual contrast,
        especially for gene_shuffle where the gradient is destroyed.
    rank_end : int
        End of the gene rank window (exclusive). Default 10.
    """
    import os
    import warnings

    COLOR_REF = "#155289"
    COLOR_MANIP = "#A7455D"

    rng = np.random.default_rng(random_seed)
    os.makedirs(output_dir, exist_ok=True)

    def read_and_unique(path, label):
        adata = ad.read_h5ad(path, backed=None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            adata.var_names_make_unique()
        return adata

    ref = read_and_unique(ref_path, "REF")
    ref_var_names = np.array(ref.var_names)

    n_cells_ref = ref.shape[0]
    if n_cells_ref == 0:
        raise ValueError("Reference AnnData has zero cells.")

    i_cell_ref = int(rng.integers(0, n_cells_ref))

    ref_counts_full = (
        ref.X[i_cell_ref].toarray().ravel()
        if sp.issparse(ref.X)
        else np.asarray(ref.X[i_cell_ref]).ravel()
    )

    ranked = np.argsort(-ref_counts_full)
    selected_positions = ranked[rank_start:rank_end]
    selected_gene_names = list(ref_var_names[selected_positions])
    ref_window = ref_counts_full[selected_positions]
    n_genes = len(selected_gene_names)

    families = [f for f in manipulation_paths_by_family if manipulation_paths_by_family[f]][:5]

    panels = [{"counts": ref_window, "label": "Reference", "ref_overlay": None}]

    for fam in families:
        adata = read_and_unique(manipulation_paths_by_family[fam][0], fam)
        manip_var_names = np.array(adata.var_names)
        n_cells = adata.shape[0]

        if n_cells == 0:
            panels.append({
                "counts": np.zeros(n_genes),
                "label": f"{fam}\n(no cells)",
                "ref_overlay": ref_window,
            })
            continue

        i_cell = i_cell_ref if i_cell_ref < n_cells else int(rng.integers(0, n_cells))

        manip_counts_full = (
            adata.X[i_cell].toarray().ravel()
            if sp.issparse(adata.X)
            else np.asarray(adata.X[i_cell]).ravel()
        )

        # Index by gene name so gene_shuffle (which permutes values in-place,
        # keeping var_names order) and any reordered var is handled correctly
        if np.array_equal(manip_var_names, ref_var_names):
            manip_window = manip_counts_full[selected_positions]
        else:
            name_to_idx = {g: i for i, g in enumerate(manip_var_names)}
            manip_window = np.array([
                manip_counts_full[name_to_idx[g]] if g in name_to_idx else 0.0
                for g in selected_gene_names
            ], dtype=float)

        panels.append({"counts": manip_window, "label": fam, "ref_overlay": ref_window})

    all_values = np.concatenate(
        [p["counts"] for p in panels]
        + [p["ref_overlay"] for p in panels if p["ref_overlay"] is not None]
    )
    ymin, ymax = float(all_values.min()), float(all_values.max())
    ypad = max(1.0, 0.05 * (ymax - ymin))
    ylim = (ymin - ypad, ymax + ypad)

    fig, axes = plt.subplots(3, 2, figsize=(8, 6), sharey=True)
    axes_flat = axes.flatten()
    bar_x = np.arange(n_genes)
    width = 0.85

    for idx, panel in enumerate(panels):
        ax = axes_flat[idx]
        if panel["ref_overlay"] is not None:
            ax.bar(bar_x, panel["ref_overlay"], width=width,
                   color=COLOR_REF, alpha=0.18, linewidth=0, zorder=0)
        ax.bar(
            bar_x, panel["counts"], width=width,
            color=COLOR_REF if panel["ref_overlay"] is None else COLOR_MANIP,
            alpha=0.7 if panel["ref_overlay"] is None else 0.92,
            linewidth=0, zorder=1,
        )
        ax.set_xlim(-0.5, n_genes - 0.5)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        # Remove all spines for a clean look
        for side, spine in ax.spines.items():
            spine.set_visible(False)
        ax.set_title(panel["label"], fontsize=11, pad=2)

    for j in range(len(panels), 6):
        axes_flat[j].axis("off")

    # Reduce whitespace between grid elements and figure edges
    fig.subplots_adjust(wspace=0.04, hspace=0.10, left=0.02, right=0.985, top=0.98, bottom=0.05)

    basename = os.path.join(output_dir, "reference_vs_manipulation_grid")
    fig.savefig(f"{basename}.png", dpi=160, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(f"{basename}.svg", bbox_inches="tight", pad_inches=0.01)
    print(f"Saved: {basename}.png and .svg")

    # If running in a notebook and want output inline, uncomment the next line:
    # plt.show()

    return fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualization of manipulations
    """)
    return


@app.cell
def _(MANIP_DIR, ad, np, plt, sp):
    # Example usage: Provide explicit files/paths here
    manipulation_files_by_family = {
        "downsample": [f"{MANIP_DIR}/downsample_7725a8fbfa4f.h5ad"],
        "gene_dropout": [f"{MANIP_DIR}/gene_dropout_f65ff3bc0ced.h5ad"],
        "gene_shuffle": [f"{MANIP_DIR}/gene_shuffle_66757744a1cd.h5ad"],
        "local_smoothing": [f"{MANIP_DIR}/local_smoothing_82382916c19d.h5ad"],
        "poisson_resampling": [f"{MANIP_DIR}/poisson_resampling_14c1c037e87f.h5ad"],
    }
    # For full workflow, you would point to the actual file paths in the above dict.

    # Call using these files and structure, seed can be set in the function call for reproducibility
    plot_reference_and_manipulations_grid(
        MANIP_DIR / "reference.h5ad",
        manipulation_files_by_family,
        ad,
        sp,
        plt,
        np,
        random_seed=1,
        rank_start=1,
        rank_end=10,
    )
    return


if __name__ == "__main__":
    app.run()
