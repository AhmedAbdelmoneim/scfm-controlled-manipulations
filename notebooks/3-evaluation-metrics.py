import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def title_cell(mo):
    mo.md(r"""
    **Note:** Canonical evaluation runs through `make evaluate CONFIG=...` (see README). Metrics
    live in `scfm_controlled_manipulations/evaluation/`; this notebook is for exploration.

    # SCFM Latent Space Evaluation: Reference vs Manipulated Structure

    This notebook compares a paired reference/manipulated dataset in raw space and embedding space.
    It assumes a 1:1 cell correspondence across all four matrices.
    """)
    return


@app.cell(hide_code=True)
def phase0_header_cell(mo):
    mo.md(r"""
    ---
    ### Phase 0: Setup, config, discovery, data load, and alignment checks
    """)
    return


@app.cell
def imports_cell():
    from pathlib import Path
    import anndata as ad
    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from scipy.spatial.distance import cdist
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
    )
    import scanpy as sc

    sns.set_theme(style="whitegrid")
    return (
        NearestNeighbors,
        Path,
        ad,
        adjusted_rand_score,
        mo,
        normalized_mutual_info_score,
        np,
        pd,
        plt,
        sc,
        silhouette_score,
        sns,
        sp,
    )


@app.cell
def config_cell(Path):
    cfg_base_dir = Path("/vault/amoneim/scfm-controlled-manipulations/processed")
    cfg_atlas = "immune"
    cfg_model = "pca"
    cfg_seed = 42

    cfg_target_family = "local_smoothing"
    cfg_target_params = {"k": 5}

    cfg_distance_metrics = ["euclidean", "cosine"]
    cfg_k_values = [15, 30, 50]
    cfg_diffusion_t_values = [1, 4, 8]
    cfg_leiden_resolutions = [0.5, 1.0]

    cfg_cell_type_col = "cell_type"
    cfg_batch_col = "batch"

    cfg_diffusion_sample_n = 2000
    cfg_silhouette_sample_n = 5000
    return (
        cfg_atlas,
        cfg_base_dir,
        cfg_batch_col,
        cfg_cell_type_col,
        cfg_diffusion_sample_n,
        cfg_diffusion_t_values,
        cfg_distance_metrics,
        cfg_k_values,
        cfg_leiden_resolutions,
        cfg_model,
        cfg_seed,
        cfg_silhouette_sample_n,
        cfg_target_family,
        cfg_target_params,
    )


@app.cell
def io_helper_cell(ad, np, sp):
    def discover_manipulation_id(manip_dir, family, target_params):
        matched_ids = []
        for candidate_path in sorted(manip_dir.glob(f"{family}_*.h5ad")):
            adata_backed = ad.read_h5ad(candidate_path, backed="r")
            candidate_params = adata_backed.uns.get("scfm_intervention", {}).get(family, {})
            adata_backed.file.close()

            if all(candidate_params.get(param_key) == param_value for param_key, param_value in target_params.items()):
                matched_ids.append(candidate_path.stem)

        if len(matched_ids) == 0:
            raise FileNotFoundError(
                f"No manipulation found for family={family}, params={target_params}"
            )
        if len(matched_ids) > 1:
            print(f"Multiple matching manipulations found; using first: {matched_ids[0]}")
            print("All matches:")
            for matched_id_value in matched_ids:
                print(f"  {matched_id_value}")
        return matched_ids[0], matched_ids

    def load_matrix_from_h5ad(h5ad_path):
        adata_obj = ad.read_h5ad(h5ad_path)
        matrix_obj = adata_obj.X
        if sp.issparse(matrix_obj):
            matrix_obj = matrix_obj.toarray()
        matrix_arr = np.asarray(matrix_obj, dtype=np.float32)
        obs_names = adata_obj.obs_names.copy()
        obs_df = adata_obj.obs.copy()
        return matrix_arr, obs_names, obs_df

    def assert_obs_names_equal(obs_names_a, obs_names_b, label_a, label_b):
        if not obs_names_a.equals(obs_names_b):
            first_mismatch = None
            min_len = min(len(obs_names_a), len(obs_names_b))
            for mismatch_idx in range(min_len):
                if obs_names_a[mismatch_idx] != obs_names_b[mismatch_idx]:
                    first_mismatch = (mismatch_idx, obs_names_a[mismatch_idx], obs_names_b[mismatch_idx])
                    break
            raise ValueError(
                f"obs_names are not aligned between {label_a} and {label_b}. "
                f"First mismatch: {first_mismatch}"
            )

    return (
        assert_obs_names_equal,
        discover_manipulation_id,
        load_matrix_from_h5ad,
    )


@app.cell
def data_load_cell(
    assert_obs_names_equal,
    cfg_atlas,
    cfg_base_dir,
    cfg_model,
    cfg_target_family,
    cfg_target_params,
    discover_manipulation_id,
    load_matrix_from_h5ad,
):
    data_manip_dir = cfg_base_dir / cfg_atlas / "results" / "manipulations"
    data_emb_dir = cfg_base_dir / cfg_atlas / "embeddings" / cfg_model

    data_resolved_manip_id, data_resolved_matches = discover_manipulation_id(
        data_manip_dir,
        cfg_target_family,
        cfg_target_params,
    )

    data_raw_ref_path = data_manip_dir / "reference.h5ad"
    data_raw_man_path = data_manip_dir / f"{data_resolved_manip_id}.h5ad"
    data_emb_ref_path = data_emb_dir / f"{cfg_model}_reference.h5ad"
    data_emb_man_path = data_emb_dir / f"{cfg_model}_{data_resolved_manip_id}.h5ad"

    raw_ref_mat, raw_ref_obs_names, raw_ref_obs = load_matrix_from_h5ad(data_raw_ref_path)
    raw_man_mat, raw_man_obs_names, raw_man_obs = load_matrix_from_h5ad(data_raw_man_path)
    emb_ref_mat, emb_ref_obs_names, emb_ref_obs = load_matrix_from_h5ad(data_emb_ref_path)
    emb_man_mat, emb_man_obs_names, emb_man_obs = load_matrix_from_h5ad(data_emb_man_path)

    assert_obs_names_equal(raw_ref_obs_names, raw_man_obs_names, "raw_ref", "raw_manipulated")
    assert_obs_names_equal(emb_ref_obs_names, emb_man_obs_names, "emb_ref", "emb_manipulated")
    assert_obs_names_equal(raw_ref_obs_names, emb_ref_obs_names, "raw_ref", "emb_ref")
    assert_obs_names_equal(raw_ref_obs_names, emb_man_obs_names, "raw_ref", "emb_manipulated")

    print(f"Resolved manipulation: {data_resolved_manip_id}")
    print(f"Raw reference:       {data_raw_ref_path}")
    print(f"Raw manipulated:     {data_raw_man_path}")
    print(f"Embedding reference: {data_emb_ref_path}")
    print(f"Embedding manip:     {data_emb_man_path}")
    print(f"Cells: {raw_ref_mat.shape[0]}")
    print(f"Raw dims: {raw_ref_mat.shape[1]} | Embedding dims: {emb_ref_mat.shape[1]}")
    return emb_man_mat, emb_ref_mat, raw_man_mat, raw_ref_mat, raw_ref_obs


@app.cell(hide_code=True)
def phase1_header_cell(mo):
    mo.md(r"""
    ---
    ### Phase 1: Basic space-level sanity checks
    """)
    return


@app.cell
def phase1_stats_funcs_cell(np):
    def calculate_space_stats(ref_mat, man_mat, space_name):
        ref_centroid = np.mean(ref_mat, axis=0)
        man_centroid = np.mean(man_mat, axis=0)
        paired_shift = np.linalg.norm(man_mat - ref_mat, axis=1)
        return {
            "space": space_name,
            "ref_mean_norm": float(np.mean(np.linalg.norm(ref_mat, axis=1))),
            "manip_mean_norm": float(np.mean(np.linalg.norm(man_mat, axis=1))),
            "ref_var_per_dim": float(np.mean(np.var(ref_mat, axis=0))),
            "manip_var_per_dim": float(np.mean(np.var(man_mat, axis=0))),
            "centroid_shift": float(np.linalg.norm(man_centroid - ref_centroid)),
            "median_paired_shift": float(np.median(paired_shift)),
        }

    return (calculate_space_stats,)


@app.cell
def phase1_run_cell(
    calculate_space_stats,
    emb_man_mat,
    emb_ref_mat,
    pd,
    raw_man_mat,
    raw_ref_mat,
):
    df_space_stats = pd.DataFrame([
        calculate_space_stats(raw_ref_mat, raw_man_mat, "raw"),
        calculate_space_stats(emb_ref_mat, emb_man_mat, "embedding"),
    ])
    df_space_stats
    return (df_space_stats,)


@app.cell
def phase1_plot_funcs_cell(plt, sns):
    def plot_space_stats(df_space_stats):
        fig_space_stats, axes_space_stats = plt.subplots(1, 2, figsize=(10, 4))

        df_space_norm_plot = df_space_stats.melt(
            id_vars=["space"],
            value_vars=["ref_mean_norm", "manip_mean_norm"],
            var_name="state",
            value_name="mean_norm",
        )
        sns.barplot(
            data=df_space_norm_plot,
            x="space",
            y="mean_norm",
            hue="state",
            ax=axes_space_stats[0],
        )
        axes_space_stats[0].set_title("Mean vector norm")

        df_space_shift_plot = df_space_stats.melt(
            id_vars=["space"],
            value_vars=["centroid_shift", "median_paired_shift"],
            var_name="metric",
            value_name="value",
        )
        sns.barplot(
            data=df_space_shift_plot,
            x="space",
            y="value",
            hue="metric",
            ax=axes_space_stats[1],
        )
        axes_space_stats[1].set_title("Global and paired shift")

        plt.tight_layout()
        plt.show()
        return fig_space_stats

    return (plot_space_stats,)


@app.cell
def phase1_plot_cell(df_space_stats, plot_space_stats):
    fig_space_stats = plot_space_stats(df_space_stats)
    return


@app.cell(hide_code=True)
def phase2_header_cell(mo):
    mo.md(r"""
    ---
    ### Phase 2: KNN preservation

    This measures whether each cell keeps the same neighbors after manipulation.
    Both per-cell recall and per-cell Jaccard are retained.
    """)
    return


@app.cell
def knn_helper_funcs_cell(NearestNeighbors, np, sp):
    def knn_indices(mat, k, metric="euclidean"):
        nn_model = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(mat)
        nn_distances, nn_idx = nn_model.kneighbors(mat)
        return nn_distances[:, 1:], nn_idx[:, 1:]

    def knn_overlap_per_cell(ref_idx, man_idx, k):
        n_cells = ref_idx.shape[0]
        row_idx = np.repeat(np.arange(n_cells), k)
        ref_graph = sp.csr_matrix(
            (np.ones(n_cells * k), (row_idx, ref_idx.ravel())),
            shape=(n_cells, n_cells),
        )
        man_graph = sp.csr_matrix(
            (np.ones(n_cells * k), (row_idx, man_idx.ravel())),
            shape=(n_cells, n_cells),
        )
        intersection = ref_graph.multiply(man_graph).sum(axis=1).A1
        union = ref_graph.maximum(man_graph).sum(axis=1).A1
        recall = intersection / k
        jaccard = intersection / np.maximum(union, 1)
        return recall, jaccard

    def compute_knn_preservation_table(raw_ref_mat, raw_man_mat, emb_ref_mat, emb_man_mat, distance_metrics, k_values, pd_module, np_module):
        knn_summary_rows = []
        knn_per_cell_frames = []
        knn_n_cells = raw_ref_mat.shape[0]

        space_triplets = [
            ("raw", raw_ref_mat, raw_man_mat),
            ("embedding", emb_ref_mat, emb_man_mat),
        ]
        for current_space, current_ref, current_man in space_triplets:
            for current_metric in distance_metrics:
                for current_k in k_values:
                    _, current_ref_idx = knn_indices(current_ref, k=current_k, metric=current_metric)
                    _, current_man_idx = knn_indices(current_man, k=current_k, metric=current_metric)
                    current_recall_vals, current_jaccard_vals = knn_overlap_per_cell(current_ref_idx, current_man_idx, current_k)
                    current_null_expectation = current_k / max(knn_n_cells - 1, 1)

                    knn_summary_rows.append({
                        "space": current_space,
                        "distance_metric": current_metric,
                        "k": current_k,
                        "mean_recall": float(np_module.mean(current_recall_vals)),
                        "median_recall": float(np_module.median(current_recall_vals)),
                        "mean_jaccard": float(np_module.mean(current_jaccard_vals)),
                        "median_jaccard": float(np_module.median(current_jaccard_vals)),
                        "null_recall_expectation": float(current_null_expectation),
                    })
                    knn_per_cell_frames.append(pd_module.DataFrame({
                        "space": current_space,
                        "distance_metric": current_metric,
                        "k": current_k,
                        "cell_index": np_module.arange(knn_n_cells),
                        "knn_recall": current_recall_vals,
                        "knn_jaccard": current_jaccard_vals,
                    }))

        summary_df = pd_module.DataFrame(knn_summary_rows)
        per_cell_df = pd_module.concat(knn_per_cell_frames, ignore_index=True)
        return summary_df, per_cell_df

    return compute_knn_preservation_table, knn_indices


@app.cell
def phase2_run_cell(
    cfg_distance_metrics,
    cfg_k_values,
    compute_knn_preservation_table,
    emb_man_mat,
    emb_ref_mat,
    np,
    pd,
    raw_man_mat,
    raw_ref_mat,
):
    df_knn_summary, df_knn_per_cell = compute_knn_preservation_table(
        raw_ref_mat,
        raw_man_mat,
        emb_ref_mat,
        emb_man_mat,
        cfg_distance_metrics,
        cfg_k_values,
        pd,
        np,
    )
    df_knn_summary
    return (df_knn_summary,)


@app.cell
def phase2_plot_funcs_cell(plt, sns):
    def plot_knn_summary(df_knn_summary, distance_metrics, k_values):
        fig_knn, axes_knn = plt.subplots(1, len(distance_metrics), figsize=(12, 4), sharey=True)
        if len(distance_metrics) == 1:
            axes_knn = [axes_knn]

        for axis_position, metric_name in enumerate(distance_metrics):
            plot_df = df_knn_summary[df_knn_summary["distance_metric"] == metric_name]
            sns.lineplot(
                data=plot_df,
                x="k",
                y="mean_recall",
                hue="space",
                marker="o",
                ax=axes_knn[axis_position],
            )
            null_df = plot_df.drop_duplicates("k")
            axes_knn[axis_position].plot(
                null_df["k"],
                null_df["null_recall_expectation"],
                "k--",
                label="random expectation",
            )
            axes_knn[axis_position].set_title(f"kNN recall: {metric_name}")
            axes_knn[axis_position].set_xticks(k_values)

        plt.tight_layout()
        plt.show()
        return fig_knn

    return (plot_knn_summary,)


@app.cell
def phase2_plot_cell(
    cfg_distance_metrics,
    cfg_k_values,
    df_knn_summary,
    plot_knn_summary,
):
    fig_knn = plot_knn_summary(df_knn_summary, cfg_distance_metrics, cfg_k_values)
    return


@app.cell(hide_code=True)
def phase3_header_cell(mo):
    mo.md(r"""
    ---
    ### Phase 3: Diffusion KL/JS divergence

    This compares random-walk transition distributions on the reference and manipulated graphs.
    JS is usually more stable; symmetric KL is reported too.
    """)
    return


@app.cell
def diffusion_helper_funcs_cell(NearestNeighbors, np, sp):
    def build_weighted_knn_adjacency(mat, k, metric="euclidean"):
        nn_model = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(mat)
        nn_distances, nn_idx = nn_model.kneighbors(mat)
        nn_distances = nn_distances[:, 1:]
        nn_idx = nn_idx[:, 1:]

        n_cells = mat.shape[0]
        row_idx = np.repeat(np.arange(n_cells), k)
        col_idx = nn_idx.ravel()
        sigma = np.median(nn_distances[:, -1])
        if sigma <= 0:
            positive_distances = nn_distances[nn_distances > 0]
            sigma = np.median(positive_distances) if positive_distances.size > 0 else 1.0
        edge_weights = np.exp(-(nn_distances.ravel() ** 2) / (2 * sigma ** 2))
        adj = sp.csr_matrix((edge_weights, (row_idx, col_idx)), shape=(n_cells, n_cells))
        adj = adj.maximum(adj.T)
        adj.eliminate_zeros()
        return adj

    def row_normalize_sparse(adj):
        row_sums = np.asarray(adj.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        return sp.diags(1.0 / row_sums) @ adj

    def sparse_transition_power(adj, t):
        transition = row_normalize_sparse(adj)
        transition_t = transition.copy()
        for power_step in range(1, t):
            transition_t = transition_t @ transition
        return transition_t

    def rowwise_symmetric_kl_js(p_sparse, q_sparse, sample_idx, eps=1e-12):
        p_arr = p_sparse[sample_idx].toarray().astype(np.float64)
        q_arr = q_sparse[sample_idx].toarray().astype(np.float64)
        p_arr = p_arr + eps
        q_arr = q_arr + eps
        p_arr = p_arr / p_arr.sum(axis=1, keepdims=True)
        q_arr = q_arr / q_arr.sum(axis=1, keepdims=True)
        m_arr = 0.5 * (p_arr + q_arr)

        kl_pq = np.sum(p_arr * np.log(p_arr / q_arr), axis=1)
        kl_qp = np.sum(q_arr * np.log(q_arr / p_arr), axis=1)
        sym_kl = 0.5 * (kl_pq + kl_qp)
        js = 0.5 * (
            np.sum(p_arr * np.log(p_arr / m_arr), axis=1)
            + np.sum(q_arr * np.log(q_arr / m_arr), axis=1)
        )
        return sym_kl, js

    def compute_diffusion_table(
        raw_ref_mat,
        raw_man_mat,
        emb_ref_mat,
        emb_man_mat,
        distance_metrics,
        k_values,
        t_values,
        sample_n,
        seed,
        pd_module,
        np_module,
    ):
        diffusion_rng = np_module.random.default_rng(seed)
        diffusion_n_cells = raw_ref_mat.shape[0]
        diffusion_sample_idx = diffusion_rng.choice(
            diffusion_n_cells,
            size=min(sample_n, diffusion_n_cells),
            replace=False,
        )
        diffusion_rows = []
        space_triplets = [
            ("raw", raw_ref_mat, raw_man_mat),
            ("embedding", emb_ref_mat, emb_man_mat),
        ]

        for current_space, current_ref, current_man in space_triplets:
            for current_metric in distance_metrics:
                for current_k in k_values:
                    current_ref_adj = build_weighted_knn_adjacency(current_ref, k=current_k, metric=current_metric)
                    current_man_adj = build_weighted_knn_adjacency(current_man, k=current_k, metric=current_metric)
                    for current_t in t_values:
                        current_ref_t = sparse_transition_power(current_ref_adj, current_t)
                        current_man_t = sparse_transition_power(current_man_adj, current_t)
                        current_sym_kl, current_js = rowwise_symmetric_kl_js(
                            current_ref_t,
                            current_man_t,
                            diffusion_sample_idx,
                        )
                        diffusion_rows.append({
                            "space": current_space,
                            "distance_metric": current_metric,
                            "k": current_k,
                            "t": current_t,
                            "sym_kl_mean": float(np_module.mean(current_sym_kl)),
                            "sym_kl_median": float(np_module.median(current_sym_kl)),
                            "js_mean": float(np_module.mean(current_js)),
                            "js_median": float(np_module.median(current_js)),
                        })

        return pd_module.DataFrame(diffusion_rows), diffusion_sample_idx

    return (compute_diffusion_table,)


@app.cell
def phase3_run_cell(
    cfg_diffusion_sample_n,
    cfg_diffusion_t_values,
    cfg_distance_metrics,
    cfg_k_values,
    cfg_seed,
    compute_diffusion_table,
    emb_man_mat,
    emb_ref_mat,
    np,
    pd,
    raw_man_mat,
    raw_ref_mat,
):
    df_diffusion, diffusion_sample_idx = compute_diffusion_table(
        raw_ref_mat,
        raw_man_mat,
        emb_ref_mat,
        emb_man_mat,
        cfg_distance_metrics,
        cfg_k_values,
        cfg_diffusion_t_values,
        cfg_diffusion_sample_n,
        cfg_seed,
        pd,
        np,
    )
    df_diffusion
    return (df_diffusion,)


@app.cell
def phase3_plot_funcs_cell(plt, sns):
    def plot_diffusion_summary(df_diffusion):
        fig_diffusion, axes_diffusion = plt.subplots(1, 2, figsize=(12, 4))

        sns.lineplot(
            data=df_diffusion,
            x="t",
            y="js_median",
            hue="space",
            style="distance_metric",
            markers=True,
            ax=axes_diffusion[0],
        )
        axes_diffusion[0].set_title("Diffusion JS divergence")

        sns.lineplot(
            data=df_diffusion,
            x="t",
            y="sym_kl_median",
            hue="space",
            style="distance_metric",
            markers=True,
            ax=axes_diffusion[1],
        )
        axes_diffusion[1].set_title("Diffusion symmetric KL")

        plt.tight_layout()
        plt.show()
        return fig_diffusion

    return (plot_diffusion_summary,)


@app.cell
def phase3_plot_cell(df_diffusion, plot_diffusion_summary):
    fig_diffusion = plot_diffusion_summary(df_diffusion)
    return


@app.cell(hide_code=True)
def phase4_header_cell(mo):
    mo.md(r"""
    ---
    ### Phase 4: Unsupervised clustering stability

    Leiden clusters are computed independently in reference and manipulated space, then compared with ARI/NMI.
    """)
    return


@app.cell
def clustering_helper_funcs_cell(
    adjusted_rand_score,
    normalized_mutual_info_score,
    np,
    sc,
):
    def run_leiden_labels(mat, k=30, metric="euclidean", resolution=1.0, seed=42):
        adata_tmp = sc.AnnData(mat)
        sc.pp.neighbors(
            adata_tmp,
            n_neighbors=k,
            metric=metric,
            use_rep="X",
            random_state=seed,
        )
        sc.tl.leiden(
            adata_tmp,
            resolution=resolution,
            random_state=seed,
            key_added="leiden_eval",
        )
        return adata_tmp.obs["leiden_eval"].astype(str).to_numpy()

    def clustering_stability(ref_mat, man_mat, k, metric, resolution, seed):
        ref_clusters = run_leiden_labels(
            ref_mat,
            k=k,
            metric=metric,
            resolution=resolution,
            seed=seed,
        )
        man_clusters = run_leiden_labels(
            man_mat,
            k=k,
            metric=metric,
            resolution=resolution,
            seed=seed,
        )
        return {
            "ari": float(adjusted_rand_score(ref_clusters, man_clusters)),
            "nmi": float(normalized_mutual_info_score(ref_clusters, man_clusters)),
            "n_ref_clusters": int(len(np.unique(ref_clusters))),
            "n_manip_clusters": int(len(np.unique(man_clusters))),
        }

    def compute_cluster_stability_table(
        raw_ref_mat,
        raw_man_mat,
        emb_ref_mat,
        emb_man_mat,
        distance_metrics,
        k_values,
        resolutions,
        seed,
        pd_module,
    ):
        cluster_rows = []
        space_triplets = [
            ("raw", raw_ref_mat, raw_man_mat),
            ("embedding", emb_ref_mat, emb_man_mat),
        ]
        for current_space, current_ref, current_man in space_triplets:
            for current_metric in distance_metrics:
                for current_k in k_values:
                    for current_resolution in resolutions:
                        current_summary = clustering_stability(
                            current_ref,
                            current_man,
                            k=current_k,
                            metric=current_metric,
                            resolution=current_resolution,
                            seed=seed,
                        )
                        cluster_rows.append({
                            "space": current_space,
                            "distance_metric": current_metric,
                            "k": current_k,
                            "resolution": current_resolution,
                            **current_summary,
                        })
        return pd_module.DataFrame(cluster_rows)

    return compute_cluster_stability_table, run_leiden_labels


@app.cell
def phase4_run_cell(
    cfg_distance_metrics,
    cfg_k_values,
    cfg_leiden_resolutions,
    cfg_seed,
    compute_cluster_stability_table,
    emb_man_mat,
    emb_ref_mat,
    pd,
    raw_man_mat,
    raw_ref_mat,
):
    df_cluster_stability = compute_cluster_stability_table(
        raw_ref_mat,
        raw_man_mat,
        emb_ref_mat,
        emb_man_mat,
        cfg_distance_metrics,
        cfg_k_values,
        cfg_leiden_resolutions,
        cfg_seed,
        pd,
    )
    df_cluster_stability
    return (df_cluster_stability,)


@app.cell
def phase4_plot_funcs_cell(plt, sns):
    def plot_cluster_stability(df_cluster_stability):
        fig_cluster, axes_cluster = plt.subplots(1, 2, figsize=(12, 4))

        sns.lineplot(
            data=df_cluster_stability,
            x="k",
            y="ari",
            hue="space",
            style="distance_metric",
            markers=True,
            ax=axes_cluster[0],
        )
        axes_cluster[0].set_title("Leiden ARI: ref vs manipulated")

        sns.lineplot(
            data=df_cluster_stability,
            x="k",
            y="nmi",
            hue="space",
            style="distance_metric",
            markers=True,
            ax=axes_cluster[1],
        )
        axes_cluster[1].set_title("Leiden NMI: ref vs manipulated")

        plt.tight_layout()
        plt.show()
        return fig_cluster

    return (plot_cluster_stability,)


@app.cell
def phase4_plot_cell(df_cluster_stability, plot_cluster_stability):
    fig_cluster = plot_cluster_stability(df_cluster_stability)
    return


@app.cell(hide_code=True)
def phase5_header_cell(mo):
    mo.md(r"""
    ---
    ### Phase 5: Standard label and batch metrics, when metadata exist

    Cell-type metrics are reported when `cfg_cell_type_col` exists. Batch metrics are reported when `cfg_batch_col` exists.
    """)
    return


@app.cell
def metadata_metric_funcs_cell(
    adjusted_rand_score,
    knn_indices,
    normalized_mutual_info_score,
    np,
    run_leiden_labels,
    silhouette_score,
):
    def safe_silhouette(mat, labels, metric="euclidean", sample_size=None, seed=42):
        labels_arr = np.asarray(labels).astype(str)
        if len(np.unique(labels_arr)) < 2:
            return np.nan
        if len(np.unique(labels_arr)) >= len(labels_arr):
            return np.nan
        actual_sample = None if sample_size is None else min(sample_size, mat.shape[0])
        return float(silhouette_score(
            mat,
            labels_arr,
            metric=metric,
            sample_size=actual_sample,
            random_state=seed,
        ))

    def mean_neighbor_same_label_fraction(mat, labels, k=30, metric="euclidean"):
        labels_arr = np.asarray(labels).astype(str)
        _, neighbor_idx = knn_indices(mat, k=k, metric=metric)
        same_vals = labels_arr[neighbor_idx] == labels_arr[:, None]
        return float(np.mean(same_vals))

    def mean_neighbor_label_entropy(mat, labels, k=30, metric="euclidean"):
        labels_arr = np.asarray(labels).astype(str)
        unique_labels, inverse_labels = np.unique(labels_arr, return_inverse=True)
        _, neighbor_idx = knn_indices(mat, k=k, metric=metric)
        entropy_vals = []
        for neighbor_row in neighbor_idx:
            counts = np.bincount(inverse_labels[neighbor_row], minlength=len(unique_labels)).astype(float)
            probs = counts / max(counts.sum(), 1.0)
            nonzero_probs = probs[probs > 0]
            entropy_vals.append(-np.sum(nonzero_probs * np.log(nonzero_probs)))
        max_entropy = np.log(len(unique_labels)) if len(unique_labels) > 1 else 1.0
        return float(np.mean(entropy_vals) / max_entropy)

    def ilisi_like_score(mat, labels, k=30, metric="euclidean"):
        labels_arr = np.asarray(labels).astype(str)
        unique_labels, inverse_labels = np.unique(labels_arr, return_inverse=True)
        _, neighbor_idx = knn_indices(mat, k=k, metric=metric)
        inv_simpson_vals = []
        for neighbor_row in neighbor_idx:
            counts = np.bincount(inverse_labels[neighbor_row], minlength=len(unique_labels)).astype(float)
            probs = counts / max(counts.sum(), 1.0)
            inv_simpson_vals.append(1.0 / np.sum(probs ** 2))
        return float(np.mean(inv_simpson_vals))

    def label_cluster_agreement(mat, labels, k=30, metric="euclidean", resolution=1.0, seed=42):
        labels_arr = np.asarray(labels).astype(str)
        cluster_labels = run_leiden_labels(
            mat,
            k=k,
            metric=metric,
            resolution=resolution,
            seed=seed,
        )
        return {
            "label_cluster_ari": float(adjusted_rand_score(labels_arr, cluster_labels)),
            "label_cluster_nmi": float(normalized_mutual_info_score(labels_arr, cluster_labels)),
        }

    def compute_metadata_metrics_for_space(
        mat,
        obs_df,
        space_name,
        cell_type_col,
        batch_col,
        k_values,
        distance_metrics,
        silhouette_sample_n,
        seed,
    ):
        rows = []
        for current_metric in distance_metrics:
            for current_k in k_values:
                base_row = {
                    "space": space_name,
                    "distance_metric": current_metric,
                    "k": current_k,
                }
                if cell_type_col in obs_df.columns:
                    cell_labels = obs_df[cell_type_col].astype(str).to_numpy()
                    rows.append({
                        **base_row,
                        "metadata_type": "cell_type",
                        "silhouette": safe_silhouette(
                            mat,
                            cell_labels,
                            metric=current_metric,
                            sample_size=silhouette_sample_n,
                            seed=seed,
                        ),
                        "same_label_neighbor_fraction": mean_neighbor_same_label_fraction(
                            mat,
                            cell_labels,
                            k=current_k,
                            metric=current_metric,
                        ),
                        "label_entropy_neighbors": mean_neighbor_label_entropy(
                            mat,
                            cell_labels,
                            k=current_k,
                            metric=current_metric,
                        ),
                        "ilisi_like": ilisi_like_score(
                            mat,
                            cell_labels,
                            k=current_k,
                            metric=current_metric,
                        ),
                        **label_cluster_agreement(
                            mat,
                            cell_labels,
                            k=current_k,
                            metric=current_metric,
                            seed=seed,
                        ),
                    })

                if batch_col in obs_df.columns:
                    batch_labels = obs_df[batch_col].astype(str).to_numpy()
                    rows.append({
                        **base_row,
                        "metadata_type": "batch",
                        "silhouette": safe_silhouette(
                            mat,
                            batch_labels,
                            metric=current_metric,
                            sample_size=silhouette_sample_n,
                            seed=seed,
                        ),
                        "same_label_neighbor_fraction": mean_neighbor_same_label_fraction(
                            mat,
                            batch_labels,
                            k=current_k,
                            metric=current_metric,
                        ),
                        "label_entropy_neighbors": mean_neighbor_label_entropy(
                            mat,
                            batch_labels,
                            k=current_k,
                            metric=current_metric,
                        ),
                        "ilisi_like": ilisi_like_score(
                            mat,
                            batch_labels,
                            k=current_k,
                            metric=current_metric,
                        ),
                        **label_cluster_agreement(
                            mat,
                            batch_labels,
                            k=current_k,
                            metric=current_metric,
                            seed=seed,
                        ),
                    })
        return rows

    def compute_all_metadata_metrics(
        raw_ref_mat,
        raw_man_mat,
        emb_ref_mat,
        emb_man_mat,
        obs_df,
        cell_type_col,
        batch_col,
        k_values,
        distance_metrics,
        silhouette_sample_n,
        seed,
        pd_module,
    ):
        metadata_rows = []
        matrix_items = [
            ("raw_reference", raw_ref_mat),
            ("raw_manipulated", raw_man_mat),
            ("embedding_reference", emb_ref_mat),
            ("embedding_manipulated", emb_man_mat),
        ]
        for current_space, current_mat in matrix_items:
            metadata_rows.extend(compute_metadata_metrics_for_space(
                current_mat,
                obs_df,
                current_space,
                cell_type_col,
                batch_col,
                k_values,
                distance_metrics,
                silhouette_sample_n,
                seed,
            ))
        return pd_module.DataFrame(metadata_rows)

    return (compute_all_metadata_metrics,)


@app.cell
def phase5_run_cell(
    cfg_batch_col,
    cfg_cell_type_col,
    cfg_distance_metrics,
    cfg_k_values,
    cfg_seed,
    cfg_silhouette_sample_n,
    compute_all_metadata_metrics,
    emb_man_mat,
    emb_ref_mat,
    pd,
    raw_man_mat,
    raw_ref_mat,
    raw_ref_obs,
):
    df_metadata_metrics = compute_all_metadata_metrics(
        raw_ref_mat,
        raw_man_mat,
        emb_ref_mat,
        emb_man_mat,
        raw_ref_obs,
        cfg_cell_type_col,
        cfg_batch_col,
        cfg_k_values,
        cfg_distance_metrics,
        cfg_silhouette_sample_n,
        cfg_seed,
        pd,
    )
    df_metadata_metrics
    return (df_metadata_metrics,)


@app.cell
def phase5_plot_funcs_cell(plt, sns):
    def plot_metadata_metrics(df_metadata_metrics):
        if df_metadata_metrics.empty:
            print("No requested metadata columns found; skipping metadata plots.")
            return None

        fig_metadata, axes_metadata = plt.subplots(1, 2, figsize=(12, 4))
        sns.barplot(
            data=df_metadata_metrics,
            x="space",
            y="silhouette",
            hue="metadata_type",
            ax=axes_metadata[0],
        )
        axes_metadata[0].set_title("Metadata silhouette")
        axes_metadata[0].tick_params(axis="x", rotation=30)

        sns.barplot(
            data=df_metadata_metrics,
            x="space",
            y="ilisi_like",
            hue="metadata_type",
            ax=axes_metadata[1],
        )
        axes_metadata[1].set_title("iLISI-like neighbor diversity")
        axes_metadata[1].tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.show()
        return fig_metadata

    return (plot_metadata_metrics,)


@app.cell
def phase5_plot_cell(df_metadata_metrics, plot_metadata_metrics):
    fig_metadata = plot_metadata_metrics(df_metadata_metrics)
    return


@app.cell(hide_code=True)
def phase6_header_cell(mo):
    mo.md(r"""
    ---
    ### Phase 6: Raw-vs-embedding robustness summaries
    """)
    return


@app.cell
def phase6_gain_funcs_cell(pd):
    def make_raw_embedding_gain_table(df_metric, value_cols, id_cols):
        if df_metric.empty:
            return pd.DataFrame()
        raw_df = df_metric[df_metric["space"] == "raw"].copy()
        emb_df = df_metric[df_metric["space"] == "embedding"].copy()
        if raw_df.empty or emb_df.empty:
            return pd.DataFrame()
        merged = emb_df.merge(
            raw_df,
            on=id_cols,
            suffixes=("_embedding", "_raw"),
            how="inner",
        )
        for value_col in value_cols:
            emb_col = f"{value_col}_embedding"
            raw_col = f"{value_col}_raw"
            if emb_col in merged.columns and raw_col in merged.columns:
                merged[f"{value_col}_embedding_minus_raw"] = merged[emb_col] - merged[raw_col]
        return merged

    def compute_all_gain_tables(df_knn_summary, df_diffusion, df_cluster_stability):
        df_knn_gain = make_raw_embedding_gain_table(
            df_knn_summary,
            value_cols=["mean_recall", "median_recall", "mean_jaccard", "median_jaccard"],
            id_cols=["distance_metric", "k"],
        )
        df_diffusion_gain = make_raw_embedding_gain_table(
            df_diffusion,
            value_cols=["sym_kl_mean", "sym_kl_median", "js_mean", "js_median"],
            id_cols=["distance_metric", "k", "t"],
        )
        df_cluster_gain = make_raw_embedding_gain_table(
            df_cluster_stability,
            value_cols=["ari", "nmi"],
            id_cols=["distance_metric", "k", "resolution"],
        )
        return df_knn_gain, df_diffusion_gain, df_cluster_gain

    return (compute_all_gain_tables,)


@app.cell
def phase6_run_cell(
    compute_all_gain_tables,
    df_cluster_stability,
    df_diffusion,
    df_knn_summary,
):
    df_knn_gain, df_diffusion_gain, df_cluster_gain = compute_all_gain_tables(
        df_knn_summary,
        df_diffusion,
        df_cluster_stability,
    )
    return df_cluster_gain, df_diffusion_gain, df_knn_gain


@app.cell(hide_code=True)
def phase6_display_header_cell(mo):
    mo.md(r"""
    #### kNN embedding - raw gain
    """)
    return


@app.cell
def phase6_knn_gain_display_cell(df_knn_gain):
    df_knn_gain
    return


@app.cell(hide_code=True)
def phase6_diffusion_display_header_cell(mo):
    mo.md(r"""
    #### Diffusion embedding - raw gain

    Negative divergence gain means the embedding is more robust than raw space.
    """)
    return


@app.cell
def phase6_diffusion_gain_display_cell(df_diffusion_gain):
    df_diffusion_gain
    return


@app.cell(hide_code=True)
def phase6_cluster_display_header_cell(mo):
    mo.md(r"""
    #### Clustering embedding - raw gain
    """)
    return


@app.cell
def phase6_cluster_gain_display_cell(df_cluster_gain):
    df_cluster_gain
    return


if __name__ == "__main__":
    app.run()
