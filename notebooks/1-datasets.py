import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Datasets for evaluation
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import re

    import marimo as mo
    import mygene
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp

    DATA_DIR = Path("/vault/amoneim/scfm-controlled-manipulations/raw_datasets")
    OUTPUT_DIR = Path("/vault/amoneim/scfm-controlled-manipulations/raw_datasets_normalized")
    ENSEMBL_MAPPING_CACHE = OUTPUT_DIR / "gene_symbol_to_human_ensembl.csv"
    return DATA_DIR, ENSEMBL_MAPPING_CACHE, OUTPUT_DIR, mo, mygene, np, pd, re, sc, sp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load from raw directory
    """)
    return


@app.cell
def _(DATA_DIR, pd, sc):
    def load_h5ads(dataset_dir, recursive=False):
        paths = dataset_dir.rglob("*.h5ad") if recursive else dataset_dir.glob("*.h5ad")
        return {
            path.relative_to(dataset_dir).with_suffix("").as_posix().replace("/", "__"): sc.read_h5ad(path, backed="r")
            for path in sorted(paths)
            if "_downloads" not in path.parts
        }

    ATLASES = load_h5ads(DATA_DIR / "atlases")
    SCEVAL = load_h5ads(DATA_DIR / "sceval")
    SCFM_EVAL = load_h5ads(DATA_DIR / "scfm_eval", recursive=True)

    manifest_path = DATA_DIR / "scfm_eval" / "manifest.csv"
    if manifest_path.is_file():
        scfm_eval_manifest = pd.read_csv(manifest_path)
        scfm_eval_manifest = scfm_eval_manifest[
            scfm_eval_manifest["status"].isin(["downloaded", "converted"])
        ].copy()
        scfm_eval_manifest["dataset_name"] = (
            scfm_eval_manifest["source"].astype(str)
            + "__"
            + scfm_eval_manifest["output_path"].map(
                lambda value: str(value).rsplit("/", 1)[-1].removesuffix(".h5ad")
            )
        )
        SCFM_EVAL_METADATA = scfm_eval_manifest.set_index("dataset_name").to_dict("index")
    else:
        SCFM_EVAL_METADATA = {}
    return ATLASES, SCEVAL, SCFM_EVAL, SCFM_EVAL_METADATA


@app.cell
def _(ATLASES, SCEVAL, SCFM_EVAL, pd):


    CELL_TYPE_COLUMNS = (
        "cell_type",
        "celltype",
        "cell_type_label",
        "cell_type_annotation",
        "cell_annotation",
        "annotation",
        "cell_label",
        "cell_ontology_class",
        "cell_type_ontology_term_id",
    )
    BATCH_COLUMNS = (
        "batch",
        "batch_id",
        "sample",
        "sample_id",
        "donor",
        "donor_id",
        "patient",
        "patient_id",
        "individual",
        "study",
        "dataset",
        "orig.ident",
    )

    def best_column(obs, candidates):
        normalized = {column.lower(): column for column in obs.columns}
        for candidate in candidates:
            if candidate.lower() in normalized:
                return normalized[candidate.lower()]
        return None

    def summarize_dataset(name, adata):
        cell_type_column = best_column(adata.obs, CELL_TYPE_COLUMNS)
        batch_column = best_column(adata.obs, BATCH_COLUMNS)
        return {
            "dataset_name": name,
            "cell_type_column": cell_type_column or "",
            "batch_column": batch_column or "",
            "n_cells": adata.n_obs,
            "n_cell_types": (
                adata.obs[cell_type_column].nunique(dropna=True) if cell_type_column else pd.NA
            ),
            "n_batches": adata.obs[batch_column].nunique(dropna=True) if batch_column else pd.NA,
        }

    dataset_summary = pd.DataFrame(
        [
            summarize_dataset(name, adata)
            for datasets in (ATLASES, SCEVAL, SCFM_EVAL)
            for name, adata in datasets.items()
        ]
    )
    dataset_summary
    return BATCH_COLUMNS, CELL_TYPE_COLUMNS, best_column


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Normalize cell type and batch columns
    """)
    return


@app.cell
def _(
    ATLASES,
    BATCH_COLUMNS,
    CELL_TYPE_COLUMNS,
    SCEVAL,
    SCFM_EVAL,
    best_column,
    pd,
):
    def normalize_obs_columns(dataset_group, datasets):
        rows = []
        for name, adata in datasets.items():
            cell_type_source = best_column(adata.obs, CELL_TYPE_COLUMNS)
            batch_source = best_column(adata.obs, BATCH_COLUMNS)

            if cell_type_source is not None:
                adata.obs["cell_type"] = adata.obs[cell_type_source]
            if batch_source is not None:
                adata.obs["batch"] = adata.obs[batch_source]

            rows.append(
                {
                    "dataset_group": dataset_group,
                    "dataset_name": name,
                    "cell_type_source": cell_type_source or "",
                    "batch_source": batch_source or "",
                    "has_cell_type": "cell_type" in adata.obs.columns,
                    "has_batch": "batch" in adata.obs.columns,
                }
            )
        return rows

    normalization_summary = pd.DataFrame(
        normalize_obs_columns("atlases", ATLASES)
        + normalize_obs_columns("sceval", SCEVAL)
        + normalize_obs_columns("scfm_eval", SCFM_EVAL)
    )
    normalization_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Filter genes
    """)
    return


@app.cell
def _(ATLASES, SCEVAL, SCFM_EVAL, best_column, pd):
    BIOTYPE_COLUMNS = (
        "feature_type",
        "feature_biotype",
        "gene_biotype",
        "biotype",
        "Biotype",
        "gene_type",
    )
    TARGET_BIOTYPE = "protein_coding"

    def infer_biotype_column(adata):
        return best_column(adata.var, BIOTYPE_COLUMNS)

    def protein_coding_filter_info(adata):
        biotype_column = infer_biotype_column(adata)
        if biotype_column is None:
            return {
                "mask": None,
                "biotype_column": "",
                "n_genes_before": adata.n_vars,
                "n_genes_after": adata.n_vars,
                "filter_applied": False,
            }

        biotypes = adata.var[biotype_column].astype(str).str.strip()
        mask = biotypes == TARGET_BIOTYPE
        n_after = int(mask.sum())
        if n_after == 0:
            return {
                "mask": None,
                "biotype_column": biotype_column,
                "n_genes_before": adata.n_vars,
                "n_genes_after": adata.n_vars,
                "filter_applied": False,
            }
        return {
            "mask": mask.to_numpy(),
            "biotype_column": biotype_column,
            "n_genes_before": adata.n_vars,
            "n_genes_after": n_after,
            "filter_applied": True,
        }

    def summarize_gene_filtering_plan(dataset_group, datasets):
        rows = []
        for name, adata in datasets.items():
            info = protein_coding_filter_info(adata)
            rows.append(
                {
                    "dataset_group": dataset_group,
                    "dataset_name": name,
                    "biotype_column": info["biotype_column"],
                    "filter_applied": info["filter_applied"],
                    "n_genes_before": info["n_genes_before"],
                    "n_genes_after": info["n_genes_after"],
                    "n_genes_removed": info["n_genes_before"] - info["n_genes_after"],
                }
            )
        return rows

    gene_filtering_plan = pd.DataFrame(
        summarize_gene_filtering_plan("atlases", ATLASES)
        + summarize_gene_filtering_plan("sceval", SCEVAL)
        + summarize_gene_filtering_plan("scfm_eval", SCFM_EVAL)
    )
    gene_filtering_plan
    return TARGET_BIOTYPE, protein_coding_filter_info


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Normalize embedding inputs
    """)
    return


@app.cell
def _(
    ATLASES,
    BATCH_COLUMNS,
    CELL_TYPE_COLUMNS,
    ENSEMBL_MAPPING_CACHE,
    SCEVAL,
    SCFM_EVAL,
    TARGET_BIOTYPE,
    best_column,
    mygene,
    np,
    pd,
    protein_coding_filter_info,
    re,
    sp,
):
    GENE_NAME_COLUMNS = (
        "feature_name",
        "gene_name",
        "gene_symbol",
        "gene_symbols",
        "original_gene_symbols",
        "original_symbol",
        "Gene",
        "symbol",
        "name",
    )
    ENSEMBL_ID_COLUMNS = (
        "ensembl_id",
        "ensemble_id",
        "ensembl_gene_id",
        "gene_id",
        "gene_ids",
        "feature_id",
        "id",
    )
    COUNT_LAYER_CANDIDATES = (
        "counts",
        "raw_counts",
        "decontXcounts",
        "soupX",
    )
    ROUND_INTEGER_LIKE_THRESHOLD = 0.99
    ENSEMBL_ID_PATTERN = re.compile(r"^ENS[A-Z]*G\d+(?:\.\d+)?$")

    def looks_like_ensembl_id(value):
        return bool(ENSEMBL_ID_PATTERN.match(str(value).strip()))

    def valid_ensembl_fraction(values):
        nonempty_values = [
            str(value).strip()
            for value in values
            if str(value).strip() and str(value).strip().lower() != "nan"
        ]
        if not nonempty_values:
            return 0.0
        return sum(looks_like_ensembl_id(value) for value in nonempty_values) / len(nonempty_values)

    def nonempty_text(value, fallback=""):
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return str(fallback).strip()
        return text

    def make_unique(values):
        counts = {}
        used = set()
        unique = []
        for value in values:
            base = nonempty_text(value, fallback="gene")
            index = counts.get(base, 0)
            candidate = base if index == 0 else f"{base}-{index}"
            while candidate in used:
                index += 1
                candidate = f"{base}-{index}"
            counts[base] = index + 1
            used.add(candidate)
            unique.append(candidate)
        return unique

    def infer_gene_names(adata):
        source_column = best_column(adata.var, GENE_NAME_COLUMNS)
        values = adata.var[source_column].tolist() if source_column else adata.var_names.tolist()
        gene_names = [
            nonempty_text(value, fallback=fallback)
            for value, fallback in zip(values, adata.var_names.astype(str), strict=False)
        ]
        return make_unique(gene_names), source_column or "var_names"

    def infer_ensembl_ids(adata):
        source_column = best_column(adata.var, ENSEMBL_ID_COLUMNS)
        if source_column:
            ids = [
                nonempty_text(value, fallback=fallback)
                for value, fallback in zip(
                    adata.var[source_column].tolist(),
                    adata.var_names.astype(str),
                    strict=False,
                )
            ]
            if valid_ensembl_fraction(ids) > 0.5:
                return ids, source_column

        var_names = adata.var_names.astype(str).tolist()
        ensembl_fraction = sum(looks_like_ensembl_id(value) for value in var_names) / len(var_names)
        if var_names and ensembl_fraction > 0.5:
            return var_names, "var_names"

        return [""] * adata.n_vars, ""

    def load_ensembl_mapping_cache():
        if not ENSEMBL_MAPPING_CACHE.is_file():
            return {}
        cached = pd.read_csv(ENSEMBL_MAPPING_CACHE).fillna("")
        return dict(zip(cached["symbol"].astype(str), cached["ensembl_id"].astype(str), strict=False))

    def save_ensembl_mapping_cache(mapping):
        ENSEMBL_MAPPING_CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"symbol": symbol, "ensembl_id": ensembl_id} for symbol, ensembl_id in sorted(mapping.items())]
        ).to_csv(ENSEMBL_MAPPING_CACHE, index=False)

    def extract_ensembl_id(query_result):
        ensembl = query_result.get("ensembl")
        if isinstance(ensembl, list):
            candidates = ensembl
        elif isinstance(ensembl, dict):
            candidates = [ensembl]
        else:
            candidates = []
        for candidate in candidates:
            gene_id = str(candidate.get("gene", "")).strip()
            if gene_id.startswith("ENSG"):
                return gene_id
        return ""

    def fetch_human_ensembl_ids(symbols):
        if not symbols:
            return {}
        mg = mygene.MyGeneInfo()
        fetched = {}
        for start in range(0, len(symbols), 1000):
            batch = symbols[start : start + 1000]
            results = mg.querymany(
                batch,
                scopes="symbol,alias",
                fields="ensembl.gene",
                species="human",
                as_dataframe=False,
                verbose=False,
            )
            for result in results:
                query = str(result.get("query", "")).strip()
                fetched[query] = extract_ensembl_id(result)
        return fetched

    def map_symbols_to_human_ensembl_ids(gene_names):
        cache = load_ensembl_mapping_cache()
        query_symbols = sorted({str(symbol).strip() for symbol in gene_names if str(symbol).strip()})
        query_symbols += [
            symbol.upper()
            for symbol in query_symbols
            if symbol.upper() != symbol and symbol.upper() not in cache
        ]
        missing = [symbol for symbol in query_symbols if symbol not in cache]
        if missing:
            cache.update(fetch_human_ensembl_ids(missing))
            save_ensembl_mapping_cache(cache)
        return [
            cache.get(symbol, "") or cache.get(symbol.upper(), "")
            for symbol in gene_names
        ]

    def infer_count_source(adata):
        if adata.raw is not None and adata.raw.shape == adata.shape:
            return "raw.X"
        for layer in COUNT_LAYER_CANDIDATES:
            if layer in adata.layers and adata.layers[layer].shape == adata.shape:
                return f"layers[{layer}]"
        return "X"

    def copy_matrix(matrix):
        if hasattr(matrix, "to_memory"):
            return matrix.to_memory().copy()
        if hasattr(matrix, "copy"):
            return matrix.copy()
        if hasattr(matrix, "toarray"):
            return sp.csr_matrix(matrix.toarray())
        return np.asarray(matrix).copy()

    def count_matrix_for_embedding(adata):
        count_source = infer_count_source(adata)
        if count_source == "raw.X":
            return copy_matrix(adata.raw.X), count_source
        if count_source.startswith("layers["):
            layer = count_source.removeprefix("layers[").removesuffix("]")
            return copy_matrix(adata.layers[layer]), count_source
        return copy_matrix(adata.X), count_source

    def sample_matrix_values(matrix, max_values=200_000):
        values = matrix.data if sp.issparse(matrix) else np.asarray(matrix).ravel()
        if values.size <= max_values:
            return values.astype(np.float64, copy=False)
        rng = np.random.default_rng(0)
        idx = rng.choice(values.size, size=max_values, replace=False)
        return values[idx].astype(np.float64, copy=False)

    def count_like_rounding_decision(matrix):
        values = sample_matrix_values(matrix)
        if values.size == 0:
            return {
                "round_counts": False,
                "integer_like_fraction": 0.0,
                "fractional_positive_fraction": 0.0,
                "reason": "empty_matrix",
            }
        if np.any(values < 0):
            return {
                "round_counts": False,
                "integer_like_fraction": 0.0,
                "fractional_positive_fraction": 0.0,
                "reason": "negative_values",
            }

        rounded = np.round(values)
        integer_like_fraction = float(np.mean(np.isclose(values, rounded, rtol=0, atol=1e-5)))
        positive = values[values > 0]
        fractional_positive_fraction = (
            float(np.mean((positive % 1) > 1e-5)) if positive.size else 0.0
        )
        should_round = integer_like_fraction >= ROUND_INTEGER_LIKE_THRESHOLD
        return {
            "round_counts": should_round,
            "integer_like_fraction": integer_like_fraction,
            "fractional_positive_fraction": fractional_positive_fraction,
            "reason": "count_like" if should_round else "not_count_like",
        }

    def round_count_like_matrix(matrix):
        decision = count_like_rounding_decision(matrix)
        if not decision["round_counts"]:
            return matrix, decision

        if sp.issparse(matrix):
            rounded = matrix.copy()
            rounded.data = np.round(rounded.data)
            rounded.eliminate_zeros()
            return rounded, decision

        return np.round(matrix), decision

    def normalize_for_embedding(adata, dataset_group="", dataset_name="", dataset_metadata=None):
        normalized = adata.to_memory() if hasattr(adata, "to_memory") else adata.copy()
        filter_info = protein_coding_filter_info(normalized)
        count_matrix, count_source = count_matrix_for_embedding(normalized)
        if filter_info["filter_applied"]:
            count_matrix = count_matrix[:, filter_info["mask"]].copy()
            normalized = normalized[:, filter_info["mask"]].copy()
        count_matrix, rounding_decision = round_count_like_matrix(count_matrix)
        gene_names, gene_name_source = infer_gene_names(normalized)
        ensembl_ids, ensembl_id_source = infer_ensembl_ids(normalized)
        n_genes_before_ensembl_filter = normalized.n_vars
        n_genes_after_ensembl_filter = normalized.n_vars
        if valid_ensembl_fraction(ensembl_ids) <= 0.5:
            ensembl_ids = map_symbols_to_human_ensembl_ids(gene_names)
            ensembl_id_source = "mygene_human_symbol"
            ensembl_mask = np.array([bool(value) for value in ensembl_ids])
            if ensembl_mask.any() and not ensembl_mask.all():
                count_matrix = count_matrix[:, ensembl_mask].copy()
                normalized = normalized[:, ensembl_mask].copy()
                gene_names = [
                    gene_name for gene_name, keep in zip(gene_names, ensembl_mask, strict=False) if keep
                ]
                ensembl_ids = [
                    ensembl_id
                    for ensembl_id, keep in zip(ensembl_ids, ensembl_mask, strict=False)
                    if keep
                ]
                n_genes_after_ensembl_filter = len(gene_names)
        cell_type_source = best_column(normalized.obs, CELL_TYPE_COLUMNS)
        batch_source = best_column(normalized.obs, BATCH_COLUMNS)

        normalized.X = count_matrix
        normalized.raw = None
        for layer in list(normalized.layers.keys()):
            del normalized.layers[layer]

        normalized.var["gene_name"] = gene_names
        normalized.var["gene_symbol"] = gene_names
        normalized.var["feature_name"] = gene_names
        normalized.var["ensembl_id"] = ensembl_ids
        normalized.var_names = gene_names

        if cell_type_source is not None:
            normalized.obs["cell_type"] = normalized.obs[cell_type_source]
        if batch_source is not None:
            normalized.obs["batch"] = normalized.obs[batch_source]

        if dataset_metadata:
            metadata = {
                key: value
                for key, value in dataset_metadata.items()
                if not (pd.isna(value) if not isinstance(value, (list, dict)) else False)
            }
            normalized.uns["scfm_eval_manifest"] = metadata
            for key in (
                "d_id",
                "source",
                "arm",
                "disease",
                "confound",
                "provenance",
                "resolution_method",
                "tissue_annotation_suspect",
                "gene_resolution_coverage",
            ):
                if key in metadata:
                    normalized.obs[f"dataset_{key}"] = metadata[key]

        if dataset_group:
            normalized.obs["dataset_group"] = dataset_group
        if dataset_name:
            normalized.obs["dataset_name"] = dataset_name

        normalized.uns["embedding_input_normalization"] = {
            "count_source": count_source,
            "gene_name_source": gene_name_source,
            "ensembl_id_source": ensembl_id_source,
            "cell_type_source": cell_type_source or "",
            "batch_source": batch_source or "",
            "round_counts": rounding_decision["round_counts"],
            "rounding_reason": rounding_decision["reason"],
            "integer_like_fraction": rounding_decision["integer_like_fraction"],
            "fractional_positive_fraction": rounding_decision["fractional_positive_fraction"],
            "gene_filter_target": TARGET_BIOTYPE,
            "gene_filter_source": filter_info["biotype_column"],
            "gene_filter_applied": filter_info["filter_applied"],
            "n_genes_before_filter": filter_info["n_genes_before"],
            "n_genes_after_filter": filter_info["n_genes_after"],
            "n_genes_before_ensembl_filter": n_genes_before_ensembl_filter,
            "n_genes_after_ensembl_filter": n_genes_after_ensembl_filter,
        }
        return normalized

    def normalize_group_for_embedding(datasets, names=None):
        selected_names = names if names is not None else datasets.keys()
        return {
            name: normalize_for_embedding(datasets[name], dataset_name=name)
            for name in selected_names
        }

    def summarize_embedding_normalization_plan(dataset_group, datasets):
        rows = []
        for name, adata in datasets.items():
            filter_info = protein_coding_filter_info(adata)
            count_matrix, _ = count_matrix_for_embedding(adata)
            if filter_info["filter_applied"]:
                count_matrix = count_matrix[:, filter_info["mask"]]
            rounding_decision = count_like_rounding_decision(count_matrix)
            filtered = adata[:, filter_info["mask"]] if filter_info["filter_applied"] else adata
            gene_names, gene_name_source = infer_gene_names(filtered)
            ensembl_ids, ensembl_id_source = infer_ensembl_ids(filtered)
            rows.append(
                {
                    "dataset_group": dataset_group,
                    "dataset_name": name,
                    "gene_filter_source": filter_info["biotype_column"],
                    "n_genes_before_filter": filter_info["n_genes_before"],
                    "n_genes_after_filter": filter_info["n_genes_after"],
                    "count_source": infer_count_source(adata),
                    "gene_name_source": gene_name_source,
                    "ensembl_id_source": ensembl_id_source,
                    "n_duplicate_gene_names_after_normalization": len(gene_names)
                    - len(set(gene_names)),
                    "n_missing_ensembl_ids": sum(not value for value in ensembl_ids),
                    "will_clear_raw": adata.raw is not None,
                    "will_clear_layers": bool(adata.layers),
                    "will_round_counts": rounding_decision["round_counts"],
                    "rounding_reason": rounding_decision["reason"],
                    "integer_like_fraction": rounding_decision["integer_like_fraction"],
                    "fractional_positive_fraction": rounding_decision[
                        "fractional_positive_fraction"
                    ],
                }
            )
        return rows

    embedding_input_normalization_plan = pd.DataFrame(
        summarize_embedding_normalization_plan("atlases", ATLASES)
        + summarize_embedding_normalization_plan("sceval", SCEVAL)
        + summarize_embedding_normalization_plan("scfm_eval", SCFM_EVAL)
    )
    embedding_input_normalization_plan
    return (normalize_for_embedding,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Save normalized datasets
    """)
    return


@app.cell
def _(
    ATLASES,
    OUTPUT_DIR,
    SCEVAL,
    SCFM_EVAL,
    SCFM_EVAL_METADATA,
    normalize_for_embedding,
    pd,
):
    OVERWRITE_OUTPUTS = False

    def save_normalized_group(dataset_group, datasets):
        output_dir = OUTPUT_DIR / dataset_group
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for name, adata in datasets.items():
            output_path = output_dir / f"{name}.h5ad"
            if output_path.exists() and not OVERWRITE_OUTPUTS:
                rows.append(
                    {
                        "dataset_group": dataset_group,
                        "dataset_name": name,
                        "output_path": str(output_path),
                        "status": "skipped_exists",
                    }
                )
                continue

            normalized = normalize_for_embedding(
                adata,
                dataset_group=dataset_group,
                dataset_name=name,
                dataset_metadata=SCFM_EVAL_METADATA.get(name),
            )
            normalized.write_h5ad(output_path, compression="gzip")
            rows.append(
                {
                    "dataset_group": dataset_group,
                    "dataset_name": name,
                    "output_path": str(output_path),
                    "status": "written",
                }
            )
        return rows

    save_summary = pd.DataFrame(
        save_normalized_group("atlases", ATLASES)
        + save_normalized_group("sceval", SCEVAL)
        + save_normalized_group("scfm_eval", SCFM_EVAL)
    )
    save_summary
    return


if __name__ == "__main__":
    app.run()
