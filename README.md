# scfm-controlled-manipulations

Evaluating single-cell foundation model representations through controlled cellular interventions.

## Running The Pipeline

The manipulation step reads an input `.h5ad`, writes embedding-ready inputs under
`results_dir/manipulations`, and applies every configured intervention.

```bash
make manipulate
```

The **evaluate** step compares each manipulation to the reference in both **raw** (manipulated
`.h5ad` under `results_dir/manipulations`) and **embedding** space
(`embeddings_root/{model}/{model}_{intervention_id}.h5ad`), and writes one consolidated CSV per model
under `results_dir/evaluation/`. Run this after embeddings exist for each manipulation variant.

Expected layout per dataset (e.g. `processed/arterial/`):

```text
embeddings/
  pca/pca_reference.h5ad
  pca/pca_{intervention_id}.h5ad
  geneformer/geneformer_reference.h5ad
  ...
results/
  manipulations/reference.h5ad
  manipulations/{intervention_id}.h5ad
  manipulations/hvg.txt
  evaluation/{model}_metrics.csv    # written by evaluate
```

```bash
make evaluate
```

| Command | Purpose | Main outputs |
|---------|---------|--------------|
| `make manipulate` | Run interventions on `input_h5ad` | `results_dir/manipulations/*.h5ad`, `reference.h5ad`, `hvg.txt` |
| `make evaluate` | Structure metrics (stats, shift, kNN+diffusion, clustering, cell/batch) | `results_dir/evaluation/{model}_metrics.csv` |

Both use the same `CONFIG` (default `configs/default.yaml`). Evaluation hyperparameters live under
the top-level `evaluation:` key (see `configs/default.yaml`). Diffusion transitions are cached under
`results_dir/evaluation_cache/`.

Set `evaluation.evaluation_workers` to parallelize across interventions (e.g. `80` on a large node).
Each worker is a **process** pinned to **one** BLAS/sklearn thread; limits are applied in
`scripts/lib/eval_runtime_env.sh` and in Python before numpy loads. On Linux, workers default to `fork` and share read-only reference matrices via copy-on-write.
Each fork worker runs neighbors+Leiden in a nested `spawn` subprocess so OpenMP from kNN does
not trigger unsafe `fork()` in scanpy/igraph. Set `evaluation_mp_start_method: spawn` if issues
persist. Diffusion pickles
live under `evaluation_cache/` with file locking.

**One atlas in the background:**

```bash
nohup scripts/run_one_evaluation.sh lung > run_logs/batch_eval_lung.log 2>&1 &
```

**Non-default venv** (once per machine): `echo .venv-05 > .python-env` or `export SCFM_UV_ENV=.venv-05`.

### Evaluation output schema

Each model writes `results_dir/evaluation/{model}_metrics.csv`. Every row includes
`dataset_id`, `model`, `intervention_id`, `intervention_name`, `metric_category`, `metric_name`,
`space`, `value_mean`, `value_median`, `value_std`, `value_min`, `value_max`, `value_q05`,
`value_q25`, `value_q75`, `value_q95`, `null_value` (when applicable), `n_cells`, and `seed`.
Additional columns depend on the category (`distance_metric`, `k`, `diffusion_t`, `resolution`,
`metadata_type`, etc.).

**Summary columns:** For distribution-based metrics, the `value_*` columns summarize a per-cell (or
per-dimension / per-pair) array: mean, median, sample std, min, max, and quantiles (5/25/75/95%).
Global metrics (silhouette, Leiden ARI) set quantiles and often `value_std` to `NaN`.
Classifier metrics use `value_mean` / `value_std` as CV mean ± std.

**Gain rows:** Categories `embedding_shift_gain`, `knn_metrics_gain`, and `clustering_metrics_gain`
append embedding-minus-raw differences for all summary columns except `value_std` (left `NaN`).

**Permutation nulls:** `knn_recall`, `diffusion_sym_kl`, `diffusion_js`, and
`classifier_roc_auc_ovr_macro_cv_mean` include broken-pairing nulls in `null_value` (mean over
cells; optional multi-shuffle average via `knn_n_null_permutations`). Diffusion uses a PHATE-style
alpha-decay kNN kernel (`knn_alpha`, default 10); set `knn_alpha: 2` for a Gaussian-like kernel.
Clear `results/evaluation_cache/` after changing diffusion kernel settings.

### Evaluation metrics (by category)

| Category | Space(s) | Metric | Description |
|----------|----------|--------|-------------|
| `embedding_stats` | `raw`, `embedding` | `mean_row_l2_norm_ref` / `_man` | Per-cell L2 norm distribution |
| | | `col_mean_ref` / `_man` | Per-gene / per-dim mean distribution |
| | | `col_variance_ref` / `_man` | Per-gene / per-dim variance distribution |
| `embedding_shift` | `raw`, `embedding` | `paired_cell_l2_norm` | Per-cell \|\|man − ref\|\|₂ (all aligned cells) |
| | | `shift_pairwise_cosine` | Subsampled cos(shift_i, shift_j) for shift_i = man_i − ref_i |
| | | `within_ref_pairwise_l2` | Subsampled all-pairs spread in reference |
| | | `within_man_pairwise_l2` | Same cell subset, all-pairs spread in manipulation |
| `knn_metrics` | `raw`, `embedding` | `knn_recall` | kNN neighborhood recall vs reference (+ permutation null) |
| | `embedding` | `diffusion_sym_kl` | Symmetric KL between PHATE-style kNN random-walk transitions (+ null) |
| | | `diffusion_js` | Jensen–Shannon divergence between transitions (+ null) |
| `clustering_metrics` | `embedding` | `leiden_ari` | ARI between independent Leiden clusterings (ref vs manip) |
| `cell_type_and_batch_metrics` | `raw_reference`, `raw_manipulated`, `embedding_reference`, `embedding_manipulated` | `silhouette` | Label silhouette (global; std `NaN`) |
| | | `neighbor_same_label_fraction` | Fraction of kNN neighbors sharing label (per-cell dist.) |
| | | `neighbor_label_entropy_norm` | Normalized neighbor-label entropy (cell types only) |
| | | `ilisi_like_inverse_simpson` | Inverse Simpson index of neighbor labels (cell types only) |
| | | `label_vs_leiden_ari` | Agreement between labels and Leiden clusters (**embedding** only) |
| | | `classifier_roc_auc_ovr_macro_cv_mean` | OVR macro ROC-AUC (3-fold CV mean ± std; permutation null) |
| | | `classifier_ap_macro_cv_mean` | OVR macro average precision (3-fold CV mean ± std) |

Configurable under `evaluation:`: `k_values`, `distance_metrics`, `diffusion_t_values`,
`knn_alpha`, `knn_bandwidth_k`, `knn_n_null_permutations`, `leiden_resolutions`,
`leiden_resolution_cell_batch`, `cell_type_col`, `batch_col`, `dataset_id`,
`stats_shift_pairwise_cell_subsample_n`, `stats_shift_pairwise_max_pairs`.

To run a different config:

```bash
make manipulate CONFIG=configs/my-run.yaml
make evaluate CONFIG=configs/my-run.yaml
```

Interventions are configured as YAML entries with a registry `name` and optional `kwargs`:

```yaml
interventions:
  - name: gene_shuffle
    kwargs:
      variant: stratified
      n_strata: 10
  - name: downsample
    kwargs:
      fraction: 0.5
```

List-valued kwargs are expanded as Cartesian sweeps. This example writes one manipulation for each
gene-shuffle variant and one local-smoothing manipulation for each `k` value:

```yaml
interventions:
  - name: gene_shuffle
    kwargs:
      variant: [random, stratified, chromosome, chromosome_control]
      n_strata: 10
  - name: local_smoothing
    kwargs:
      k: [5, 10, 20, 50, 100]
      n_pcs: 50
```

You can also place sweep-only parameters under `sweep`; values in `sweep` override same-named
entries in `kwargs` before expansion.

Each output stores intervention provenance in `adata.uns["scfm_intervention"][name]`, including the
seed and operation-specific metadata.

The manipulation directory also includes:

- `reference.h5ad`: the prepared, slimmed reference AnnData from the input dataset.
- `hvg.txt`: one gene symbol per line for the top `hvg_n_top_genes` highly variable genes. When
  `adata.raw` is present and matches the main matrix shape, HVGs are computed from `adata.raw.X`
  (counts); otherwise from `adata.X`.

The CLI uses Python logging and defaults to `log_level: INFO`. Set `log_level: DEBUG`, `WARNING`,
or another standard logging level in the config to adjust verbosity.

If `manipulation_workers` is omitted, manipulations run sequentially. Set it to a bounded value to
process multiple variants in parallel; each worker loads one copy of the input AnnData, so choose
this based on available memory.

Before each manipulated `.h5ad` is saved, the pipeline refreshes count-derived metadata with
`scanpy.pp.calculate_qc_metrics(..., percent_top=None)`. This updates standard Scanpy QC columns
such as `total_counts` and `n_genes_by_counts`, and also writes compatibility aliases `n_counts`
and `n_genes`.

The pipeline also ensures every saved file has `adata.var["gene_name"]` and
`adata.var["ensembl_id"]`. It uses common aliases when present, falls back to `var_names` for
`gene_name`, and uses `var_names` for `ensembl_id` only when they look like Ensembl gene IDs.
Before writing, `adata.var_names` is set to `adata.var["gene_name"]` so embedding inputs use gene
symbols as feature names while retaining Ensembl IDs in `.var`.

By default, manipulated outputs are slimmed before writing: `layers`, `obsm`, `varm`, `obsp`,
`varp`, `raw`, and unrelated `uns` entries are removed while preserving intervention provenance.
`h5ad_compression` controls file compression, and `overwrite_manipulations: false` skips variants
whose output files already exist.

## Interventions

### `gene_shuffle`

Shuffles gene annotations while keeping the count matrix fixed, so model inputs see the same counts
assigned to perturbed gene identities.

Parameters:

- `variant`: one of `random`, `stratified`, `chromosome`, or `chromosome_control`. `random`
  permutes all genes globally, `stratified` permutes genes within mean-expression bins,
  `chromosome` permutes genes only within their chromosome, and `chromosome_control` builds random
  size-matched groups from chromosome group sizes before permuting within those groups.
- `n_strata`: number of expression strata for `variant: stratified`; defaults to `10`.
- `ensembl_id_column`: `adata.var` column containing Ensembl gene IDs for chromosome-aware
  variants; defaults to `ensembl_id`.
- `species`: MyGene species used for chromosome lookup; defaults to `human`.
- `chromosome_cache_path`: optional path for the cached Ensembl-to-chromosome CSV. By default this
  is stored under `~/.cache/scfm-controlled-manipulations/`.

Chromosome-aware variants require every gene to have an Ensembl ID in `adata.var[ensembl_id_column]`.
Missing chromosome mappings are downloaded with MyGene and cached for future runs. If only some genes
remain unmapped after lookup, they are left in their original positions and excluded from
chromosome-aware shuffling. If no genes can be mapped, the intervention raises an error.

Metadata includes `variant`, `n_strata`, `ensembl_id_column`, `species`, `chromosome_cache_path`,
chromosome-aware `group_sizes`, unmapped gene metadata, `seed`, and the gene `permutation`.

### `downsample`

Applies binomial subsampling to non-zero count entries, modeling reduced sequencing depth while
preserving the original matrix shape.

Parameters:

- `fraction`: sampling probability for each count; must satisfy `0 < fraction <= 1`.

Metadata includes `fraction`, `seed`, and median per-cell counts before and after downsampling.

### `gene_dropout`

Randomly removes a fraction of non-zero expression entries with Bernoulli dropout.

Parameters:

- `dropout_rate`: probability of dropping each non-zero entry; must satisfy
  `0 <= dropout_rate < 1` and defaults to `0.3`.

Metadata includes `dropout_rate`, `actual_fraction_dropped`, and `seed`.

### `poisson_resampling`

Repeatedly Poisson-resamples non-zero counts with rate equal to the previously sampled count. One
iteration simulates a single extra stochastic count-sampling step; larger values chain the process so
iteration `i + 1` samples from the counts produced by iteration `i`.

Parameters:

- `iterations`: number of chained Poisson resampling steps; must be at least `1` and defaults to
  `1`.

Metadata includes `iterations`, the fixed `rate_multiplier`, `seed`, and total counts before and
after each resampling step.

### `local_smoothing`

Builds a k-nearest-neighbor graph from PCA coordinates of log-normalized expression and averages raw
counts across each cell's neighborhood. The smoothed counts are rounded to non-negative integers.

Parameters:

- `k`: number of neighbors used for smoothing; must be at least `2` and defaults to `15`.
- `n_pcs`: requested number of PCA components for neighbor finding; defaults to `50`.

For small datasets, the effective `k` and `n_pcs` are capped by the available observations and
variables. Metadata includes requested and effective parameter values, `seed`, and the sparse
smoothing operator components.

