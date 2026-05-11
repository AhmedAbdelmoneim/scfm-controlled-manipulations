# scfm-controlled-manipulations

Evaluating single-cell foundation model representations through controlled cellular interventions.

## Running The Pipeline

The manipulation step reads an input `.h5ad`, applies every configured intervention, and writes
one manipulated AnnData file per intervention under `results_dir/manipulations`.

```bash
make manipulate
```

The analysis step expects one embedding `.h5ad` per model and intervention ID under
`embeddings_root/{model}/{intervention_id}.h5ad`, computes configured metrics, and writes parquet
tables under `results_dir/metrics`.

```bash
make analyze
```

Both targets use `configs/default.yaml` by default. To run a different config:

```bash
make manipulate CONFIG=configs/my-run.yaml
make analyze CONFIG=configs/my-run.yaml
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

