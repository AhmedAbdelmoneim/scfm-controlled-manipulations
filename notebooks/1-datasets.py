import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Datasets for evaluation
    """)
    return


@app.cell
def _():
    import gzip
    import shutil
    import tempfile
    import zipfile
    from pathlib import Path
    import re

    import anndata as ad
    import marimo as mo
    import mygene
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scipy.io as sio
    import scipy.sparse as sp

    DATA_DIR = Path("/vault/amoneim/scfm-controlled-manipulations/raw_datasets")
    PROJECT_DIR = Path("/home/amoneim/scfm-controlled-manipulations")
    CURATED_DATASETS_PATH = PROJECT_DIR / "data" / "test-datasets.csv"
    PREPROCESSED_DIR = Path("/vault/amoneim/scfm-controlled-manipulations/2-preprocessed_datasets")
    DOWNSAMPLED_DIR = Path("/vault/amoneim/scfm-controlled-manipulations/3-downsampled-datasets")
    ENSEMBL_MAPPING_CACHE = PREPROCESSED_DIR / "gene_symbol_to_human_ensembl.csv"
    return (
        CURATED_DATASETS_PATH,
        DATA_DIR,
        DOWNSAMPLED_DIR,
        ENSEMBL_MAPPING_CACHE,
        PREPROCESSED_DIR,
        ad,
        gzip,
        mo,
        mygene,
        np,
        pd,
        re,
        sc,
        shutil,
        sio,
        sp,
        tempfile,
        zipfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load from raw directory
    """)
    return


@app.cell
def _(
    CURATED_DATASETS_PATH,
    DATA_DIR,
    ad,
    gzip,
    np,
    pd,
    sc,
    shutil,
    sio,
    sp,
    tempfile,
    zipfile,
):
    def load_h5ads(dataset_dir, recursive=False, exclude_parts=()):
        paths = dataset_dir.rglob("*.h5ad") if recursive else dataset_dir.glob("*.h5ad")
        return {
            path.relative_to(dataset_dir).with_suffix("").as_posix().replace("/", "__"): sc.read_h5ad(path, backed="r")
            for path in sorted(paths)
            if "_downloads" not in path.parts
            and not any(part in path.parts for part in exclude_parts)
        }

    TRAJECTORY_DIR = DATA_DIR / "trajectory_benchmark"
    TRAJECTORY_MIN_GENES_PER_CELL = 200
    TRAJECTORY_MIN_CELLS_PER_GENE = 3

    def make_unique_index(values):
        counts = {}
        unique = []
        for value in map(str, values):
            count = counts.get(value, 0)
            unique.append(value if count == 0 else f"{value}-{count}")
            counts[value] = count + 1
        return pd.Index(unique)

    def apply_trajectory_minimal_qc(adata):
        filtered = adata.copy()
        sc.pp.filter_cells(filtered, min_genes=TRAJECTORY_MIN_GENES_PER_CELL)
        sc.pp.filter_genes(filtered, min_cells=TRAJECTORY_MIN_CELLS_PER_GENE)
        filtered.uns["trajectory_benchmark_minimal_qc"] = {
            "min_genes_per_cell": TRAJECTORY_MIN_GENES_PER_CELL,
            "min_cells_per_gene": TRAJECTORY_MIN_CELLS_PER_GENE,
        }
        return filtered

    def read_gene_by_cell_csv_gz(path, chunksize=512):
        with gzip.open(path, "rt") as handle:
            cell_names = handle.readline().rstrip("\n").split(",")[1:]

        gene_names = []
        chunks = []
        for chunk in pd.read_csv(path, index_col=0, chunksize=chunksize):
            gene_names.extend(chunk.index.astype(str))
            chunks.append(sp.csr_matrix(chunk.to_numpy(dtype=np.float32, copy=False)))

        gene_by_cell = sp.vstack(chunks, format="csr")
        return ad.AnnData(
            X=gene_by_cell.T.tocsr(),
            obs=pd.DataFrame(index=make_unique_index(cell_names)),
            var=pd.DataFrame(index=make_unique_index(gene_names)),
        )

    def read_cell_by_gene_tsv_gz(path, chunksize=512):
        with gzip.open(path, "rt") as handle:
            columns = handle.readline().lstrip("# ").rstrip("\n").split("\t")
        cell_column = columns[0]
        gene_names = columns[1:]

        cell_names = []
        chunks = []
        for chunk in pd.read_csv(
            path,
            sep="\t",
            names=columns,
            skiprows=1,
            chunksize=chunksize,
        ):
            cell_names.extend(chunk[cell_column].astype(str))
            chunks.append(sp.csr_matrix(chunk.iloc[:, 1:].to_numpy(dtype=np.float32, copy=False)))

        return ad.AnnData(
            X=sp.vstack(chunks, format="csr"),
            obs=pd.DataFrame(index=make_unique_index(cell_names)),
            var=pd.DataFrame(index=make_unique_index(gene_names)),
        )

    def align_obs_metadata(adata, metadata, obs_key):
        metadata = metadata.copy()
        metadata[obs_key] = metadata[obs_key].astype(str)
        metadata = metadata.set_index(obs_key)
        shared = adata.obs_names.intersection(metadata.index)
        aligned = adata[shared].copy()
        aligned.obs = metadata.loc[aligned.obs_names].copy()
        return aligned

    def load_emt_trajectory():
        dataset_dir = TRAJECTORY_DIR / "EMT_GSE147405"
        adata = read_gene_by_cell_csv_gz(dataset_dir / "GSE147405_A549_TGFB1_TimeCourse_UMI_matrix.csv.gz")
        metadata = pd.read_csv(dataset_dir / "GSE147405_A549_TGFB1_TimeCourse_metadata.csv.gz", index_col=0)
        metadata.index = metadata.index.astype(str)
        adata = adata[adata.obs_names.intersection(metadata.index)].copy()
        adata.obs = metadata.loc[adata.obs_names].copy()
        adata = adata[~adata.obs["Time"].astype(str).str.endswith("_rm")].copy()
        adata.obs["timepoint"] = adata.obs["Time"].astype(str)
        adata.obs["trajectory_stage"] = adata.obs["timepoint"].replace({"3d": "late", "7d": "late"})
        adata.obs["cell_type"] = adata.obs["Cluster"].astype(str)
        adata.obs["batch"] = adata.obs["Mix"].astype(str)
        adata.var["gene_name"] = adata.var_names.astype(str)
        adata.uns["source"] = "GSE147405_A549_TGFB1_TimeCourse"
        return apply_trajectory_minimal_qc(adata)

    def load_veres_trajectory():
        dataset_dir = TRAJECTORY_DIR / "Veres_GSE114412"
        adata = read_cell_by_gene_tsv_gz(dataset_dir / "GSE114412_Stage_5.all.processed_counts.tsv.gz")
        metadata = pd.read_csv(
            dataset_dir / "GSE114412_Stage_5.endocrine_pseudotime.cell_metadata.tsv.gz",
            sep="\t",
        )
        adata = align_obs_metadata(adata, metadata, "library.barcode")
        adata.obs["stage"] = "Stage 5"
        adata.obs["timepoint"] = "Stage_5_day_" + adata.obs["CellDay"].astype(str)
        adata.obs["cell_type"] = adata.obs["Assigned_cluster"].astype(str)
        adata.obs["batch"] = adata.obs["Lib_prep_batch"].astype(str)
        adata.var["gene_name"] = adata.var_names.astype(str)
        adata.uns["source"] = "GSE114412_Stage_5_endocrine_pseudotime"
        return apply_trajectory_minimal_qc(adata)

    def read_10x_from_zip(zip_handle, sample, label):
        prefix = f"scRNAseq/{sample}/"
        matrix = sio.mmread(zip_handle.open(prefix + "matrix.mtx")).tocsr().T
        genes = pd.read_csv(zip_handle.open(prefix + "genes.tsv"), sep="\t", header=None)
        barcodes = pd.read_csv(zip_handle.open(prefix + "barcodes.tsv"), sep="\t", header=None)[0].astype(str)
        obs = pd.DataFrame(index=make_unique_index([f"{sample}_{barcode}" for barcode in barcodes]))
        obs["sample_id"] = sample
        obs["timepoint"] = label
        obs["batch"] = sample
        var = pd.DataFrame(index=make_unique_index(genes[1].astype(str)))
        var["ensembl_id"] = genes[0].astype(str).to_numpy()
        var["gene_name"] = var.index.astype(str)
        return ad.AnnData(X=matrix, obs=obs, var=var)

    def filter_library_size_sequential_percentiles(adata, lower_percentile=20, upper_percentile=75):
        totals = np.asarray(adata.X.sum(axis=1)).ravel()
        lower_bound = np.percentile(totals, lower_percentile)
        above_lower = totals > lower_bound
        filtered = adata[above_lower].copy()

        filtered_totals = np.asarray(filtered.X.sum(axis=1)).ravel()
        upper_bound = np.percentile(filtered_totals, upper_percentile)
        below_upper = filtered_totals < upper_bound
        filtered = filtered[below_upper].copy()
        filtered.uns["library_size_filter"] = {
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile,
            "lower_bound": float(lower_bound),
            "upper_bound_after_lower_filter": float(upper_bound),
        }
        return filtered

    def load_ebdata_trajectory():
        sample_labels = {
            "T0_1A": "Day 00-03",
            "T2_3B": "Day 06-09",
            "T4_5C": "Day 12-15",
            "T6_7D": "Day 18-21",
            "T8_9E": "Day 24-27",
        }
        zip_path = TRAJECTORY_DIR / "EBdata_Mendeley_v6n743h5ng" / "scRNAseq.zip"
        with zipfile.ZipFile(zip_path) as zip_handle:
            datasets = [
                filter_library_size_sequential_percentiles(read_10x_from_zip(zip_handle, sample, label))
                for sample, label in sample_labels.items()
            ]
        combined = ad.concat(datasets, join="outer", merge="same", index_unique=None)
        combined.uns["source"] = "Mendeley_10.17632_v6n743h5ng.1_scRNAseq"
        return apply_trajectory_minimal_qc(combined)

    def read_h5ad_gz(path):
        with gzip.open(path, "rb") as source, tempfile.NamedTemporaryFile(suffix=".h5ad") as temporary:
            shutil.copyfileobj(source, temporary)
            temporary.flush()
            return sc.read_h5ad(temporary.name)

    def load_hspc_trajectory():
        adata = read_h5ad_gz(
            TRAJECTORY_DIR / "HSPC_GSE226824" / "GSE226824_HSPC-all_filtered.h5ad.gz"
        )
        adata.obs["timepoint"] = adata.obs["time"].astype(str)
        adata.obs["cell_type"] = adata.obs["clusters"].astype(str)
        adata.obs["batch"] = adata.obs["hashtags"].astype(str)
        adata.var["gene_name"] = adata.var_names.astype(str)
        adata.uns["source"] = "GSE226824_HSPC-all_filtered"
        adata.uns["source_note"] = "GEO/paper metadata describe this accession as murine HSPCs."
        return adata

    def load_trajectory_benchmark():
        if not TRAJECTORY_DIR.is_dir():
            return {}
        return {
            "emt": load_emt_trajectory(),
            "veres": load_veres_trajectory(),
            "ebdata": load_ebdata_trajectory(),
            "hspc_ifna": load_hspc_trajectory(),
        }

    ATLASES = load_h5ads(DATA_DIR / "atlases")
    SCEVAL = load_h5ads(DATA_DIR / "sceval")
    HER2ST = load_h5ads(DATA_DIR / "scfm_eval" / "her2st")
    SPATIALLIBD = load_h5ads(DATA_DIR / "scfm_eval" / "spatialLIBD")
    SCFM_EVAL = load_h5ads(
        DATA_DIR / "scfm_eval",
        recursive=True,
        exclude_parts=("her2st", "merfishhprd", "spatialLIBD"),
    )
    TRAJECTORY_BENCHMARK = load_trajectory_benchmark()
    CURATED_DATASETS = pd.read_csv(CURATED_DATASETS_PATH)
    for column in (
        "Downsampled",
        "Cell type task",
        "Batch task",
        "Trajectory task",
        "Perturbation task",
    ):
        CURATED_DATASETS[column] = CURATED_DATASETS[column].astype(str).str.upper().eq("TRUE")



    return (
        ATLASES,
        CURATED_DATASETS,
        HER2ST,
        SCEVAL,
        SCFM_EVAL,
        SPATIALLIBD,
        TRAJECTORY_BENCHMARK,
    )


@app.cell
def _(
    ATLASES,
    CURATED_DATASETS,
    HER2ST,
    SCEVAL,
    SCFM_EVAL,
    SPATIALLIBD,
    TRAJECTORY_BENCHMARK,
    pd,
):
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
        "Assigned_cluster",
        "Assigned_subcluster",
        "Cluster",
        "cluster",
        "clusters",
        "milestone_id",
        "layer_guess_reordered_short",
        "layer_guess_reordered",
        "layer_guess",
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

    DATASET_GROUPS = {
        "atlases": ATLASES,
        "sceval": SCEVAL,
        "scfm_eval": SCFM_EVAL,
        "her2st": HER2ST,
        "spatialLIBD": SPATIALLIBD,
        "trajectory_benchmark": TRAJECTORY_BENCHMARK,
    }

    def find_dataset(dataset_name):
        matches = [
            (dataset_group, datasets[dataset_name])
            for dataset_group, datasets in DATASET_GROUPS.items()
            if dataset_name in datasets
        ]
        if len(matches) != 1:
            raise KeyError(f"{dataset_name} matched {len(matches)} dataset groups")
        return matches[0]

    def curated_dataset_items():
        for row in CURATED_DATASETS.to_dict("records"):
            dataset_name = row["Dataset Name"]
            dataset_group, adata = find_dataset(dataset_name)
            yield dataset_group, dataset_name, adata, row

    def summarize_dataset(dataset_group, name, adata, metadata):
        cell_type_column = best_column(adata.obs, CELL_TYPE_COLUMNS)
        batch_column = best_column(adata.obs, BATCH_COLUMNS)
        return {
            "dataset_group": dataset_group,
            "dataset_name": name,
            "cell_type_task": metadata["Cell type task"],
            "batch_task": metadata["Batch task"],
            "trajectory_task": metadata["Trajectory task"],
            "cell_type_column": cell_type_column or "",
            "batch_column": batch_column or "",
            "n_cells": adata.n_obs,
            "n_cell_types": (
                adata.obs[cell_type_column].nunique(dropna=True) if cell_type_column else pd.NA
            ),
            "n_batches": adata.obs[batch_column].nunique(dropna=True) if batch_column else pd.NA,
        }

    missing_curated_datasets = [
        dataset_name
        for dataset_name in CURATED_DATASETS["Dataset Name"]
        if not any(dataset_name in datasets for datasets in DATASET_GROUPS.values())
    ]
    if missing_curated_datasets:
        raise KeyError(f"Curated datasets missing from loaded groups: {missing_curated_datasets}")

    dataset_summary = pd.DataFrame(
        [
            summarize_dataset(dataset_group, name, adata, metadata)
            for dataset_group, name, adata, metadata in curated_dataset_items()
        ]
    )
    dataset_summary
    return BATCH_COLUMNS, CELL_TYPE_COLUMNS, best_column, curated_dataset_items


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Trajectory inference dataset cleanups
    """)
    return


@app.cell
def _(TRAJECTORY_BENCHMARK, curated_dataset_items, pd):
    TRAJECTORY_SOURCE_COLUMNS = {
        "emt": "trajectory_stage",
        "veres": "timepoint",
        "ebdata": "timepoint",
        "hspc_ifna": "timepoint",
    }

    TRAJECTORY_VALUE_ORDERS = {
        "emt": ["0d", "8h", "1d", "late"],
        "veres": [f"Stage_5_day_{day}" for day in range(8)],
        "ebdata": ["Day 00-03", "Day 06-09", "Day 12-15", "Day 18-21", "Day 24-27"],
        "hspc_ifna": ["control", "3h", "24h", "72h"],
    }

    def set_trajectory_columns(dataset_name, adata, source_column, ordered_values):
        value_to_index = {value: index for index, value in enumerate(ordered_values)}

        labels = adata.obs[source_column].astype(str)
        observed_values = set(labels.dropna().unique())
        unexpected_values = sorted(observed_values - set(ordered_values))
        if unexpected_values:
            raise ValueError(
                f"{dataset_name} has unexpected {source_column} values: {unexpected_values}"
            )

        adata.obs["trajectory_label"] = pd.Categorical(
            labels,
            categories=ordered_values,
            ordered=True,
        )
        adata.obs["trajectory"] = labels.map(value_to_index).astype(int)
        adata.uns["trajectory_cleanup"] = {
            "source_column": source_column,
            "label_column": "trajectory_label",
            "numeric_column": "trajectory",
            "value_order": ordered_values,
            "value_to_index": value_to_index,
        }

    def clean_trajectory_benchmark(dataset_name, adata):
        set_trajectory_columns(
            dataset_name,
            adata,
            TRAJECTORY_SOURCE_COLUMNS[dataset_name],
            TRAJECTORY_VALUE_ORDERS[dataset_name],
        )

    def clean_dynverse_trajectory(dataset_name, adata):
        if "milestone_id" not in adata.obs.columns:
            raise KeyError(f"{dataset_name} is missing dynverse ground-truth milestone_id")
        ordered_values = list(pd.unique(adata.obs["milestone_id"].astype(str)))
        set_trajectory_columns(dataset_name, adata, "milestone_id", ordered_values)

    for dataset_name, adata in TRAJECTORY_BENCHMARK.items():
        clean_trajectory_benchmark(dataset_name, adata)

    for dataset_group, dataset_name, adata, metadata in curated_dataset_items():
        if metadata["Trajectory task"] and dataset_group == "scfm_eval" and dataset_name.startswith("dynverse__"):
            clean_dynverse_trajectory(dataset_name, adata)

    trajectory_cleanup_summary = pd.DataFrame(
        [
            {
                "dataset_name": dataset_name,
                "source_column": adata.uns["trajectory_cleanup"]["source_column"],
                "trajectory_label": label,
                "trajectory": index,
                "n_cells": int((adata.obs["trajectory"] == index).sum()),
            }
            for _, dataset_name, adata, metadata in curated_dataset_items()
            if metadata["Trajectory task"] and "trajectory_cleanup" in adata.uns
            for index, label in enumerate(adata.uns["trajectory_cleanup"]["value_order"])
        ]
    )
    trajectory_cleanup_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Normalize cell type and batch columns
    """)
    return


@app.cell
def _(
    BATCH_COLUMNS,
    CELL_TYPE_COLUMNS,
    best_column,
    curated_dataset_items,
    pd,
):
    def normalize_obs_columns():
        rows = []
        errors = []
        for dataset_group, name, adata, metadata in curated_dataset_items():
            cell_type_source = best_column(adata.obs, CELL_TYPE_COLUMNS)
            batch_source = best_column(adata.obs, BATCH_COLUMNS)

            if cell_type_source is not None:
                adata.obs["cell_type"] = adata.obs[cell_type_source]
            if batch_source is not None:
                adata.obs["batch"] = adata.obs[batch_source]

            if metadata["Cell type task"] and "cell_type" not in adata.obs.columns:
                errors.append(f"{name} has Cell type task=True but no cell_type source column")
            if metadata["Batch task"] and "batch" not in adata.obs.columns:
                errors.append(f"{name} has Batch task=True but no batch source column")

            rows.append(
                {
                    "dataset_group": dataset_group,
                    "dataset_name": name,
                    "cell_type_task": metadata["Cell type task"],
                    "batch_task": metadata["Batch task"],
                    "cell_type_source": cell_type_source or "",
                    "batch_source": batch_source or "",
                    "has_cell_type": "cell_type" in adata.obs.columns,
                    "has_batch": "batch" in adata.obs.columns,
                }
            )
        if errors:
            raise ValueError("; ".join(errors))
        return rows

    normalization_summary = pd.DataFrame(normalize_obs_columns())
    normalization_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Filter genes
    """)
    return


@app.cell
def _(best_column, curated_dataset_items, pd):
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

    def summarize_gene_filtering_plan():
        rows = []
        for dataset_group, name, adata, _ in curated_dataset_items():
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

    gene_filtering_plan = pd.DataFrame(summarize_gene_filtering_plan())
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
    BATCH_COLUMNS,
    CELL_TYPE_COLUMNS,
    ENSEMBL_MAPPING_CACHE,
    TARGET_BIOTYPE,
    best_column,
    curated_dataset_items,
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
        requires_cell_type = bool(dataset_metadata.get("Cell type task")) if dataset_metadata else False
        requires_batch = bool(dataset_metadata.get("Batch task")) if dataset_metadata else False
        requires_trajectory = bool(dataset_metadata.get("Trajectory task")) if dataset_metadata else False

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
        if requires_cell_type and "cell_type" not in normalized.obs.columns:
            raise ValueError(f"{dataset_name} requires cell_type but no source column was found")
        if requires_batch and "batch" not in normalized.obs.columns:
            raise ValueError(f"{dataset_name} requires batch but no source column was found")
        if requires_trajectory:
            if "trajectory" not in normalized.obs.columns or "trajectory_label" not in normalized.obs.columns:
                raise ValueError(f"{dataset_name} requires trajectory but trajectory columns were not normalized")
            normalized.obs["trajectory"] = normalized.obs["trajectory"].astype(int)
            normalized.obs["trajectory_label"] = normalized.obs["trajectory_label"].astype(str)

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
            "requires_cell_type": requires_cell_type,
            "requires_batch": requires_batch,
            "requires_trajectory": requires_trajectory,
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

    def summarize_embedding_normalization_plan():
        rows = []
        for dataset_group, name, adata, metadata in curated_dataset_items():
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
                    "requires_cell_type": metadata["Cell type task"],
                    "has_cell_type": "cell_type" in adata.obs.columns,
                    "requires_batch": metadata["Batch task"],
                    "has_batch": "batch" in adata.obs.columns,
                    "requires_trajectory": metadata["Trajectory task"],
                    "has_trajectory": "trajectory" in adata.obs.columns and "trajectory_label" in adata.obs.columns,
                    "rounding_reason": rounding_decision["reason"],
                    "integer_like_fraction": rounding_decision["integer_like_fraction"],
                    "fractional_positive_fraction": rounding_decision[
                        "fractional_positive_fraction"
                    ],
                }
            )
        return rows

    embedding_input_normalization_plan = pd.DataFrame(summarize_embedding_normalization_plan())
    embedding_input_normalization_plan
    return (normalize_for_embedding,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Save normalized datasets
    """)
    return


@app.cell
def _(PREPROCESSED_DIR, curated_dataset_items, normalize_for_embedding, pd):
    OVERWRITE_PREPROCESSED = False

    def save_preprocessed_datasets(overwrite=OVERWRITE_PREPROCESSED):
        PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        rows = []
        for dataset_group, dataset_name, adata, metadata in curated_dataset_items():
            output_path = PREPROCESSED_DIR / f"{dataset_name}.h5ad"
            if output_path.exists() and not overwrite:
                rows.append(
                    {
                        "dataset_group": dataset_group,
                        "dataset_name": dataset_name,
                        "output_path": str(output_path),
                        "status": "skipped_exists",
                    }
                )
                continue

            normalized = normalize_for_embedding(
                adata,
                dataset_group=dataset_group,
                dataset_name=dataset_name,
                dataset_metadata=metadata,
            )
            normalized.write_h5ad(output_path, compression="gzip")
            rows.append(
                {
                    "dataset_group": dataset_group,
                    "dataset_name": dataset_name,
                    "output_path": str(output_path),
                    "status": "written",
                    "n_obs": normalized.n_obs,
                    "n_vars": normalized.n_vars,
                }
            )
        summary = pd.DataFrame(rows)
        summary.to_csv(PREPROCESSED_DIR / "manifest.csv", index=False)
        return summary

    preprocessed_save_summary = save_preprocessed_datasets()
    preprocessed_save_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Save downsampled datasets
    """)
    return


@app.cell(hide_code=True)
def _(DOWNSAMPLED_DIR, PREPROCESSED_DIR, curated_dataset_items, pd, sc):
    OVERWRITE_DOWNSAMPLED = False
    DOWNSAMPLE_TARGET_CELLS = 5_000
    DOWNSAMPLE_RANDOM_SEED = 0

    def save_downsampled_datasets(overwrite=OVERWRITE_DOWNSAMPLED):
        DOWNSAMPLED_DIR.mkdir(parents=True, exist_ok=True)
        rows = []
        for dataset_group, dataset_name, _, metadata in curated_dataset_items():
            input_path = PREPROCESSED_DIR / f"{dataset_name}.h5ad"
            output_path = DOWNSAMPLED_DIR / f"{dataset_name}.h5ad"
            if output_path.exists() and not overwrite:
                rows.append(
                    {
                        "dataset_group": dataset_group,
                        "dataset_name": dataset_name,
                        "input_path": str(input_path),
                        "output_path": str(output_path),
                        "status": "skipped_exists",
                    }
                )
                continue
            if not input_path.is_file():
                raise FileNotFoundError(f"Missing preprocessed dataset: {input_path}")

            adata = sc.read_h5ad(input_path)
            n_obs_before = adata.n_obs
            target_n_obs = min(DOWNSAMPLE_TARGET_CELLS, n_obs_before)
            if n_obs_before > target_n_obs:
                sc.pp.subsample(
                    adata,
                    n_obs=target_n_obs,
                    random_state=DOWNSAMPLE_RANDOM_SEED,
                    copy=False,
                )
                status = "downsampled"
            else:
                status = "copied"
            adata.uns["downsampling"] = {
                "target_n_obs": DOWNSAMPLE_TARGET_CELLS,
                "random_seed": DOWNSAMPLE_RANDOM_SEED,
                "n_obs_before": n_obs_before,
                "n_obs_after": adata.n_obs,
                "curated_downsampled": bool(metadata["Downsampled"]),
            }
            adata.write_h5ad(output_path, compression="gzip")
            rows.append(
                {
                    "dataset_group": dataset_group,
                    "dataset_name": dataset_name,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "status": status,
                    "n_obs_before": n_obs_before,
                    "n_obs_after": adata.n_obs,
                    "curated_downsampled": bool(metadata["Downsampled"]),
                }
            )
        summary = pd.DataFrame(rows)
        summary.to_csv(DOWNSAMPLED_DIR / "manifest.csv", index=False)
        return summary

    downsampled_save_summary = save_downsampled_datasets()
    downsampled_save_summary
    return


if __name__ == "__main__":
    app.run()
