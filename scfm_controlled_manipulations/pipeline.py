from __future__ import annotations

# Pin BLAS/OpenMP to 1 thread before numpy/scipy load (via anndata/scanpy).
from scfm_controlled_manipulations.compute_env import apply_thread_limits

apply_thread_limits(threads_per_process=1)

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import logging
import multiprocessing as mp
from pathlib import Path
import re
from typing import Any

import anndata as ad
import scanpy as sc
import yaml

from scfm_controlled_manipulations import interventions
from scfm_controlled_manipulations.evaluation.run import run_evaluate
from scfm_controlled_manipulations.io import (
    intervention_id,
    manipulation_path,
)
from scfm_controlled_manipulations.sweep_config import expand_intervention_specs

logger = logging.getLogger(__name__)
_WORKER_ADATA: ad.AnnData | None = None


def _configure_logging(level: str = "INFO") -> None:
    class _FlushingStreamHandler(logging.StreamHandler):
        def emit(self, record: logging.LogRecord) -> None:
            super().emit(record)
            self.flush()

    handler = _FlushingStreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s")
    )
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[handler],
        force=True,
    )
    for dependency_logger in ("httpx", "httpcore", "biothings", "biothings.client", "mygene"):
        logging.getLogger(dependency_logger).setLevel(logging.WARNING)


def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


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
        logger.info(
            "Added var[%s] from %s",
            gene_name_column,
            f"var[{source}]" if source else "var_names",
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
        logger.info("Added var[%s] from var[%s]", ensembl_id_column, source)
        return

    if _looks_like_ensembl_ids(adata.var_names):
        adata.var[ensembl_id_column] = adata.var_names.astype(str)
        logger.info("Added var[%s] from var_names", ensembl_id_column)
        return

    raise ValueError(
        "Could not infer Ensembl gene IDs. Add an `ensembl_id` column to `adata.var`, "
        "or provide one of: ensembl_gene_id, gene_id, gene_ids, feature_id, id."
    )


def set_var_names_to_gene_names(
    adata: ad.AnnData,
    *,
    gene_name_column: str = "gene_name",
) -> None:
    """Use gene symbols as AnnData variable names for embedding-ready outputs."""
    if gene_name_column not in adata.var:
        raise ValueError(f"Cannot set var_names: adata.var['{gene_name_column}'] is missing")

    gene_names = adata.var[gene_name_column].astype(str)
    invalid = adata.var[gene_name_column].isna() | (gene_names.str.strip() == "")
    if invalid.any():
        raise ValueError(
            f"Cannot set var_names: adata.var['{gene_name_column}'] contains "
            f"{int(invalid.sum())} missing or empty gene names"
        )

    adata.var_names = gene_names.to_numpy()


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
    set_var_names_to_gene_names(
        adata,
        gene_name_column=str(options.get("gene_name_column", "gene_name")),
    )
    refresh_count_metadata(adata)

    if options.get("slim_outputs", True):
        slim_manipulated_adata(adata)

    adata.uns["scfm_pipeline"] = {
        "qc_metadata_refreshed": True,
        "slimmed_output": bool(options.get("slim_outputs", True)),
        "gene_name_column": str(options.get("gene_name_column", "gene_name")),
        "ensembl_id_column": str(options.get("ensembl_id_column", "ensembl_id")),
        "var_names": str(options.get("gene_name_column", "gene_name")),
    }


def _output_options(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "slim_outputs": config.get("slim_manipulated_outputs", True),
        "h5ad_compression": config.get("h5ad_compression", "gzip"),
        "gene_name_column": config.get("gene_name_column", "gene_name"),
        "ensembl_id_column": config.get("ensembl_id_column", "ensembl_id"),
        "hvg_n_top_genes": config.get("hvg_n_top_genes", 2000),
        "overwrite": config.get("overwrite_manipulations", True),
        "log_level": config.get("log_level", "INFO"),
    }


def _write_h5ad(adata: ad.AnnData, path: Path, compression: str | None) -> None:
    if compression:
        adata.write_h5ad(path, compression=compression)
    else:
        adata.write_h5ad(path)


def _format_spec(spec: Mapping[str, Any]) -> str:
    return f"{spec['name']} {dict(spec.get('kwargs') or {})}"


def _write_reference_h5ad(
    adata_in: ad.AnnData,
    manip_dir: Path,
    output_options: Mapping[str, Any],
) -> None:
    out_path = manip_dir / "reference.h5ad"
    if out_path.exists() and not output_options.get("overwrite", True):
        logger.info("Skipping existing reference.h5ad at %s", out_path)
        return

    reference = adata_in.copy()
    prepare_manipulated_adata(reference, output_options)
    reference.uns["scfm_reference"] = {"source": "input_h5ad"}
    _write_h5ad(reference, out_path, output_options.get("h5ad_compression"))
    logger.info("Wrote reference h5ad to %s", out_path)


def _write_hvg_file(
    adata_in: ad.AnnData,
    manip_dir: Path,
    output_options: Mapping[str, Any],
) -> None:
    out_path = manip_dir / "hvg.txt"
    if out_path.exists() and not output_options.get("overwrite", True):
        logger.info("Skipping existing hvg.txt at %s", out_path)
        return

    gene_name_column = str(output_options.get("gene_name_column", "gene_name"))
    n_top_genes = min(int(output_options.get("hvg_n_top_genes", 2000)), adata_in.n_vars)
    if n_top_genes < 1:
        raise ValueError("Cannot compute HVGs for an AnnData object with zero genes")

    if (
        adata_in.raw is not None
        and adata_in.raw.shape == adata_in.shape
        and adata_in.raw.n_vars == adata_in.n_vars
    ):
        x_hvg = adata_in.raw.X
        logger.info("HVG selection using adata.raw.X (counts layer)")
    else:
        x_hvg = adata_in.X
        logger.info("HVG selection using adata.X")

    hvg_adata = ad.AnnData(
        X=x_hvg,
        var=adata_in.var[[gene_name_column]].copy(),
    )
    sc.pp.highly_variable_genes(
        hvg_adata,
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
        inplace=True,
    )

    hvg_var = hvg_adata.var[hvg_adata.var["highly_variable"]].copy()
    if "highly_variable_rank" in hvg_var:
        hvg_var = hvg_var.sort_values("highly_variable_rank")
    gene_names = hvg_var[gene_name_column].astype(str).tolist()
    out_path.write_text("\n".join(gene_names) + "\n", encoding="utf-8")
    logger.info("Wrote %d seurat_v3 HVGs to %s", len(gene_names), out_path)


def _write_embedding_inputs(
    adata_in: ad.AnnData,
    manip_dir: Path,
    output_options: Mapping[str, Any],
) -> None:
    _write_reference_h5ad(adata_in, manip_dir, output_options)
    _write_hvg_file(adata_in, manip_dir, output_options)


def _apply_and_write_manipulation(
    adata_in: ad.AnnData,
    results_dir: Path,
    spec: Mapping[str, Any],
    seed: int | None,
    output_options: Mapping[str, Any],
) -> dict[str, Any]:
    name = str(spec["name"])
    kwargs = dict(spec.get("kwargs") or {})
    iid = intervention_id(name, kwargs)
    out_path = manipulation_path(results_dir, iid)
    if out_path.exists() and not output_options.get("overwrite", True):
        logger.debug("Skipping existing manipulation %s at %s", iid, out_path)
        return {
            "intervention_id": iid,
            "name": name,
            "kwargs": kwargs,
            "path": str(out_path),
            "status": "skipped",
        }

    logger.debug("Applying intervention %s kwargs=%s", name, kwargs)
    cls = interventions.REGISTRY[name]
    intervention = cls(**kwargs)
    out = intervention.apply(adata_in, seed=seed)
    prepare_manipulated_adata(out, output_options)
    _write_h5ad(out, out_path, output_options.get("h5ad_compression"))
    logger.debug("Wrote manipulation %s to %s", iid, out_path)
    return {
        "intervention_id": iid,
        "name": name,
        "kwargs": kwargs,
        "path": str(out_path),
        "status": "written",
    }


def _init_worker_adata(input_path: str, options: Mapping[str, Any]) -> None:
    global _WORKER_ADATA

    apply_thread_limits(threads_per_process=1)
    _configure_logging(str(options.get("log_level", "INFO")))
    logger.debug("Worker loading input AnnData from %s", input_path)
    _WORKER_ADATA = ad.read_h5ad(input_path)
    ensure_gene_metadata(
        _WORKER_ADATA,
        gene_name_column=str(options.get("gene_name_column", "gene_name")),
        ensembl_id_column=str(options.get("ensembl_id_column", "ensembl_id")),
    )
    logger.debug(
        "Worker loaded input AnnData with %d cells and %d genes",
        _WORKER_ADATA.n_obs,
        _WORKER_ADATA.n_vars,
    )


def _apply_and_write_manipulation_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    if _WORKER_ADATA is None:
        raise RuntimeError("Worker AnnData was not initialized")

    return _apply_and_write_manipulation(
        _WORKER_ADATA,
        Path(task["results_dir"]),
        task["spec"],
        task["seed"],
        task["output_options"],
    )


def _log_manipulation_progress(done: int, total: int, result: Mapping[str, Any]) -> None:
    logger.info(
        "Completed %d/%d: %s %s %s -> %s",
        done,
        total,
        result["status"],
        result["intervention_id"],
        _format_spec({"name": result["name"], "kwargs": result["kwargs"]}),
        result["path"],
    )


def _prepare_reference_phase(
    input_path: Path,
    manip_dir: Path,
    output_options: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    *,
    prewarm_caches: bool,
) -> ad.AnnData:
    """Load input, ensure gene metadata, write reference + HVG, optionally prewarm caches."""
    logger.info("Loading input AnnData from %s", input_path)
    adata_in = ad.read_h5ad(input_path)
    ensure_gene_metadata(
        adata_in,
        gene_name_column=str(output_options["gene_name_column"]),
        ensembl_id_column=str(output_options["ensembl_id_column"]),
    )
    logger.info("Loaded input AnnData with %d cells and %d genes", adata_in.n_obs, adata_in.n_vars)
    _write_embedding_inputs(adata_in, manip_dir, output_options)
    if prewarm_caches:
        _prewarm_intervention_caches(adata_in, specs, output_options)
    return adata_in


def _prewarm_intervention_caches(
    adata_in: ad.AnnData,
    specs: Sequence[Mapping[str, Any]],
    output_options: Mapping[str, Any],
) -> None:
    cache_warmers = []
    for spec in specs:
        cls = interventions.REGISTRY[str(spec["name"])]
        intervention = cls(**dict(spec.get("kwargs") or {}))
        if hasattr(intervention, "warm_cache"):
            cache_warmers.append(intervention)

    if not cache_warmers:
        return

    logger.info("Prewarming caches for %d chromosome-aware variants", len(cache_warmers))
    for intervention in cache_warmers:
        intervention.warm_cache(adata_in)


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

    logger.info(
        "Starting manipulation run: %d total variants with %d workers",
        len(specs),
        workers,
    )
    logger.info(
        "Input=%s results_dir=%s",
        input_path,
        results_dir,
    )
    if workers == 1:
        adata_in = _prepare_reference_phase(
            input_path,
            manip_dir,
            output_options,
            specs,
            prewarm_caches=False,
        )
        for index, spec in enumerate(specs, start=1):
            result = _apply_and_write_manipulation(
                adata_in, results_dir, spec, seed, output_options
            )
            _log_manipulation_progress(index, len(specs), result)
        logger.info("Finished manipulation run: %d/%d variants complete", len(specs), len(specs))
        return

    logger.info("Launching %d manipulation workers", workers)
    adata_in = _prepare_reference_phase(
        input_path,
        manip_dir,
        output_options,
        specs,
        prewarm_caches=True,
    )
    try:
        del adata_in
    finally:
        gc.collect()

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
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            _log_manipulation_progress(done, len(specs), result)
    logger.info("Finished manipulation run: %d/%d variants complete", len(specs), len(specs))


def main() -> None:
    parser = argparse.ArgumentParser(description="SCFM controlled manipulations pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_man = sub.add_parser("manipulate", help="Run interventions on input_h5ad")
    p_man.add_argument("--config", type=Path, required=True)

    p_ev = sub.add_parser(
        "evaluate",
        help="Structure metrics (raw + embedding) vs reference for each manipulation",
    )
    p_ev.add_argument("--config", type=Path, required=True)

    args = parser.parse_args()
    cfg = _load_config(args.config)
    _configure_logging(str(cfg.get("log_level", "INFO")))
    logger.info("Loaded config from %s", args.config)

    if args.command == "manipulate":
        run_manipulate(cfg)
    elif args.command == "evaluate":
        run_evaluate(cfg)


if __name__ == "__main__":
    main()
