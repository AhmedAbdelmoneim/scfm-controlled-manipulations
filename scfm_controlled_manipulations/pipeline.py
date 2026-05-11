from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import json
import multiprocessing as mp
from pathlib import Path
import re
from typing import Any

import anndata as ad
import pandas as pd
import scanpy as sc
import yaml

from scfm_controlled_manipulations import interventions
from scfm_controlled_manipulations import metrics as metrics_pkg
from scfm_controlled_manipulations.io import (
    embedding_path,
    intervention_id,
    load_embedding,
    manipulation_path,
    metric_results_path,
)

_WORKER_ADATA: ad.AnnData | None = None


def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_sweep_value(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def _expand_kwargs(kwargs: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not kwargs:
        return [{}]

    keys = list(kwargs)
    values = [
        list(value) if _is_sweep_value(value) else [value]
        for value in (kwargs[key] for key in keys)
    ]
    return [dict(zip(keys, combination, strict=True)) for combination in product(*values)]


def expand_intervention_specs(specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Expand list-valued intervention kwargs into a Cartesian sweep."""
    expanded = []
    for spec in specs:
        name = spec["name"]
        kwargs = dict(spec.get("kwargs") or {})
        kwargs.update(dict(spec.get("sweep") or {}))

        for expanded_kwargs in _expand_kwargs(kwargs):
            expanded.append({"name": name, "kwargs": expanded_kwargs})

    return expanded


def refresh_count_metadata(adata: ad.AnnData) -> None:
    """Refresh count-derived QC metadata after an intervention changes ``adata.X``."""
    sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)
    adata.obs["n_counts"] = adata.obs["total_counts"]
    adata.obs["n_genes"] = adata.obs["n_genes_by_counts"]


def _first_existing_var_column(adata: ad.AnnData, candidates: Sequence[str]) -> str | None:
    return next((column for column in candidates if column in adata.var), None)


def _looks_like_ensembl_ids(values: Sequence[Any]) -> bool:
    pattern = re.compile(r"^ENS[A-Z]*G\d+(?:\.\d+)?$")
    values = [str(value) for value in values]
    return bool(values) and all(pattern.match(value) for value in values)


def ensure_gene_metadata(
    adata: ad.AnnData,
    *,
    gene_name_column: str = "gene_name",
    ensembl_id_column: str = "ensembl_id",
) -> None:
    """Ensure stable gene name and Ensembl ID columns exist in ``adata.var``."""
    if gene_name_column not in adata.var:
        source = _first_existing_var_column(
            adata,
            [
                "gene_name",
                "gene_names",
                "gene_symbol",
                "gene_symbols",
                "symbol",
                "feature_name",
                "name",
            ],
        )
        adata.var[gene_name_column] = (
            adata.var[source].astype(str).to_numpy() if source else adata.var_names.astype(str)
        )

    if ensembl_id_column in adata.var:
        return

    source = _first_existing_var_column(
        adata,
        [
            "ensembl_id",
            "ensembl_gene_id",
            "gene_id",
            "gene_ids",
            "feature_id",
            "id",
        ],
    )
    if source:
        adata.var[ensembl_id_column] = adata.var[source].astype(str).to_numpy()
        return

    if _looks_like_ensembl_ids(adata.var_names):
        adata.var[ensembl_id_column] = adata.var_names.astype(str)
        return

    raise ValueError(
        "Could not infer Ensembl gene IDs. Add an `ensembl_id` column to `adata.var`, "
        "or provide one of: ensembl_gene_id, gene_id, gene_ids, feature_id, id."
    )


def slim_manipulated_adata(adata: ad.AnnData) -> None:
    """Drop large derived containers that are not needed for manipulated count files."""
    preserved_uns = {}
    if "scfm_intervention" in adata.uns:
        preserved_uns["scfm_intervention"] = adata.uns["scfm_intervention"]

    adata.layers.clear()
    adata.obsm.clear()
    adata.varm.clear()
    adata.obsp.clear()
    adata.varp.clear()
    adata.uns.clear()
    adata.uns.update(preserved_uns)
    adata.raw = None


def prepare_manipulated_adata(adata: ad.AnnData, options: Mapping[str, Any]) -> None:
    """Apply final output invariants before writing a manipulated AnnData file."""
    ensure_gene_metadata(
        adata,
        gene_name_column=str(options.get("gene_name_column", "gene_name")),
        ensembl_id_column=str(options.get("ensembl_id_column", "ensembl_id")),
    )
    refresh_count_metadata(adata)

    if options.get("slim_outputs", True):
        slim_manipulated_adata(adata)

    adata.uns["scfm_pipeline"] = {
        "qc_metadata_refreshed": True,
        "slimmed_output": bool(options.get("slim_outputs", True)),
        "gene_name_column": str(options.get("gene_name_column", "gene_name")),
        "ensembl_id_column": str(options.get("ensembl_id_column", "ensembl_id")),
    }


def _output_options(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "slim_outputs": config.get("slim_manipulated_outputs", True),
        "h5ad_compression": config.get("h5ad_compression", "gzip"),
        "gene_name_column": config.get("gene_name_column", "gene_name"),
        "ensembl_id_column": config.get("ensembl_id_column", "ensembl_id"),
        "overwrite": config.get("overwrite_manipulations", True),
    }


def _write_h5ad(adata: ad.AnnData, path: Path, compression: str | None) -> None:
    if compression:
        adata.write_h5ad(path, compression=compression)
    else:
        adata.write_h5ad(path)


def _apply_and_write_manipulation(
    adata_in: ad.AnnData,
    results_dir: Path,
    spec: Mapping[str, Any],
    seed: int | None,
    output_options: Mapping[str, Any],
) -> str:
    name = str(spec["name"])
    kwargs = dict(spec.get("kwargs") or {})
    iid = intervention_id(name, kwargs)
    out_path = manipulation_path(results_dir, iid)
    if out_path.exists() and not output_options.get("overwrite", True):
        return iid

    cls = interventions.REGISTRY[name]
    intervention = cls(**kwargs)
    out = intervention.apply(adata_in, seed=seed)
    prepare_manipulated_adata(out, output_options)
    _write_h5ad(out, out_path, output_options.get("h5ad_compression"))
    return iid


def _init_worker_adata(input_path: str, options: Mapping[str, Any]) -> None:
    global _WORKER_ADATA

    _WORKER_ADATA = ad.read_h5ad(input_path)
    ensure_gene_metadata(
        _WORKER_ADATA,
        gene_name_column=str(options.get("gene_name_column", "gene_name")),
        ensembl_id_column=str(options.get("ensembl_id_column", "ensembl_id")),
    )


def _apply_and_write_manipulation_worker(task: Mapping[str, Any]) -> str:
    if _WORKER_ADATA is None:
        raise RuntimeError("Worker AnnData was not initialized")

    return _apply_and_write_manipulation(
        _WORKER_ADATA,
        Path(task["results_dir"]),
        task["spec"],
        task["seed"],
        task["output_options"],
    )


def run_manipulate(config: dict[str, Any]) -> None:
    """Apply each configured intervention and write ``manipulations/{intervention_id}.h5ad``."""
    input_path = Path(config["input_h5ad"])
    results_dir = Path(config["results_dir"])
    seed = config.get("seed")
    output_options = _output_options(config)
    specs = expand_intervention_specs(config["interventions"])
    workers = max(1, int(config.get("manipulation_workers", 1)))

    manip_dir = results_dir / "manipulations"
    manip_dir.mkdir(parents=True, exist_ok=True)

    if workers == 1:
        adata_in = ad.read_h5ad(input_path)
        ensure_gene_metadata(
            adata_in,
            gene_name_column=str(output_options["gene_name_column"]),
            ensembl_id_column=str(output_options["ensembl_id_column"]),
        )
        for spec in specs:
            _apply_and_write_manipulation(adata_in, results_dir, spec, seed, output_options)
        return

    tasks = [
        {
            "results_dir": str(results_dir),
            "spec": spec,
            "seed": seed,
            "output_options": output_options,
        }
        for spec in specs
    ]
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker_adata,
        initargs=(str(input_path), output_options),
    ) as executor:
        futures = [executor.submit(_apply_and_write_manipulation_worker, task) for task in tasks]
        for future in as_completed(futures):
            future.result()


def _reference_intervention_id(config: dict[str, Any]) -> str:
    if rid := config.get("reference_intervention_id"):
        return str(rid)
    ref = config.get("reference_intervention")
    if not ref:
        raise ValueError(
            "Analyze requires `reference_intervention` {{name, kwargs}} or "
            "`reference_intervention_id` string."
        )
    return intervention_id(ref["name"], dict(ref.get("kwargs") or {}))


def run_analyze(config: dict[str, Any]) -> None:
    """Load external embeddings per model and intervention, compute metrics, write parquet tables."""
    results_dir = Path(config["results_dir"])
    embeddings_root = Path(config["embeddings_root"])
    ref_id = _reference_intervention_id(config)

    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows_by_metric: dict[str, list[dict[str, Any]]] = {m: [] for m in config["metrics"]}

    for model in config["models"]:
        ref_path = embedding_path(embeddings_root, model, ref_id)
        emb_ref = load_embedding(ref_path)

        for spec in expand_intervention_specs(config["interventions"]):
            name = spec["name"]
            kwargs = dict(spec.get("kwargs") or {})
            iid = intervention_id(name, kwargs)
            pert_path = embedding_path(embeddings_root, model, iid)
            emb_pert = load_embedding(pert_path)

            for metric_name in config["metrics"]:
                cls = metrics_pkg.REGISTRY[metric_name]
                metric = cls()
                computed = metric.compute(emb_ref, emb_pert)
                row = {
                    "model": model,
                    "intervention_name": name,
                    "intervention_id": iid,
                    "kwargs_json": json.dumps(kwargs, sort_keys=True),
                    "reference_intervention_id": ref_id,
                    **computed,
                }
                rows_by_metric[metric_name].append(row)

    for metric_name, rows in rows_by_metric.items():
        df = pd.DataFrame(rows)
        out_path = metric_results_path(results_dir, metric_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="SCFM controlled manipulations pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_man = sub.add_parser("manipulate", help="Run interventions on input_h5ad")
    p_man.add_argument("--config", type=Path, required=True)

    p_an = sub.add_parser("analyze", help="Compute metrics from precomputed embeddings")
    p_an.add_argument("--config", type=Path, required=True)

    args = parser.parse_args()
    cfg = _load_config(args.config)

    if args.command == "manipulate":
        run_manipulate(cfg)
    elif args.command == "analyze":
        run_analyze(cfg)


if __name__ == "__main__":
    main()
