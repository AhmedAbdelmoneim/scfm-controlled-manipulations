"""Resolve gene symbols to stable Ensembl IDs and modern scFM-compatible symbols."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import pickle
import re

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_RELEASES: tuple[int, ...] = (75, 98, 103, 111)
_DEFAULT_MODERN_RELEASE: int = 111

_CACHE_DIR = Path(
    os.environ.get(
        "FM_GENE_RESOLVER_CACHE",
        Path.home() / ".cache" / "fm_gene_resolver",
    )
)

_ENSEMBL_ID_PATTERN = re.compile(r"^ENS[A-Z]{0,4}G\d{6,}(?:\.\d+)?$", re.IGNORECASE)
_DEDUP_SUFFIX = re.compile(r"_dup_\d+$")

_SYMBOL_CANDIDATE_COLS = (
    "gene_symbol",
    "gene_name",
    "gene_names",
    "symbol",
    "feature_name",
    "name",
    "Gene",
)
_ENSEMBL_CANDIDATE_COLS = (
    "ensembl_id",
    "ensembl_gene_id",
    "gene_id",
    "gene_ids",
    "feature_id",
    "id",
)


def _looks_like_ensembl_id(value) -> bool:
    return bool(_ENSEMBL_ID_PATTERN.match(str(value).strip()))


def _normalize_ensembl_id(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return ""
    return s.split(".", maxsplit=1)[0]


def _strip_dedup_suffix(s) -> str:
    return _DEDUP_SUFFIX.sub("", str(s))


def _first_var_column(adata: ad.AnnData, candidates: tuple[str, ...]) -> str | None:
    return next((c for c in candidates if c in adata.var.columns), None)


def _ensure_release(release: int):
    from pyensembl import EnsemblRelease

    data = EnsemblRelease(release)
    try:
        _ = data.gene_names()[:1]
        return data
    except Exception:
        pass

    logger.warning(
        "pyensembl release %d not installed; downloading and indexing now "
        "(one-time cost, may take a few minutes)...",
        release,
    )
    data.download()
    data.index()
    return data


def _build_symbol_to_id_table(releases: tuple[int, ...]) -> dict[str, str]:
    table: dict[str, str] = {}
    for release in sorted(releases):
        logger.info("Indexing symbol -> ENSG mappings from Ensembl release %d ...", release)
        data = _ensure_release(release)
        for name in data.gene_names():
            try:
                ids = data.gene_ids_of_gene_name(name)
            except Exception:
                continue
            if not ids:
                continue
            table[str(name)] = _normalize_ensembl_id(ids[0])
    return table


def _build_id_to_modern_symbol(modern_release: int) -> dict[str, str]:
    logger.info(
        "Building ENSG -> modern symbol table from Ensembl release %d ...",
        modern_release,
    )
    modern = _ensure_release(modern_release)
    table: dict[str, str] = {}
    for gene in modern.genes():
        ensg = _normalize_ensembl_id(gene.gene_id)
        if not ensg:
            continue
        name = (gene.gene_name or "").strip()
        if not name or _looks_like_ensembl_id(name):
            name = ensg
        table[ensg] = name
    return table


def _cache_path(name: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / name


def _load_or_build(path: Path, build_fn):
    if path.exists():
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except Exception:
            logger.warning("Cached table at %s is corrupt; rebuilding.", path)
    obj = build_fn()
    with path.open("wb") as f:
        pickle.dump(obj, f)
    return obj


def _load_or_build_symbol_table(releases: tuple[int, ...]) -> dict[str, str]:
    tag = "_".join(str(r) for r in sorted(releases))
    path = _cache_path(f"symbol_to_ensg_releases_{tag}.pkl")
    return _load_or_build(path, lambda: _build_symbol_to_id_table(releases))


def _load_or_build_modern_table(modern_release: int) -> dict[str, str]:
    path = _cache_path(f"ensg_to_modern_symbol_release_{modern_release}.pkl")
    return _load_or_build(path, lambda: _build_id_to_modern_symbol(modern_release))


def _mygene_resolve(missing: list[str]) -> dict[str, str]:
    if not missing:
        return {}
    try:
        import mygene
    except ImportError:
        logger.warning("mygene not installed; skipping fallback.")
        return {}

    mg = mygene.MyGeneInfo()
    out: dict[str, str] = {}
    chunk = 1000
    for i in range(0, len(missing), chunk):
        batch = missing[i : i + chunk]
        try:
            hits = mg.querymany(
                batch,
                scopes="symbol,alias,name,prev_symbol",
                fields="ensembl.gene",
                species="human",
            )
        except Exception as e:
            logger.warning("mygene chunk failed: %s", e)
            continue
        for hit in hits:
            if hit.get("notfound"):
                continue
            q = str(hit.get("query", "")).strip()
            ens = hit.get("ensembl")
            if isinstance(ens, list) and ens:
                ens = ens[0]
            if isinstance(ens, dict):
                gid = ens.get("gene", "")
            else:
                gid = ens or ""
            if isinstance(gid, list) and gid:
                gid = gid[0]
            gid = _normalize_ensembl_id(gid)
            if q and gid and q not in out:
                out[q] = gid
    return out


def sync_var_index_for_write(adata: ad.AnnData) -> ad.AnnData:
    """Align ``gene_symbol`` / ``gene_name`` with ``var_names`` and clear index name for h5ad I/O."""
    names = adata.var_names.astype(str).to_numpy()
    if "gene_symbol" in adata.var.columns:
        adata.var["gene_symbol"] = names
    if "gene_name" in adata.var.columns:
        adata.var["gene_name"] = names
    adata.var.index.name = None
    return adata


def warmup_caches(
    releases: tuple[int, ...] = _DEFAULT_RELEASES,
    modern_release: int = _DEFAULT_MODERN_RELEASE,
) -> None:
    """Pre-build Ensembl union + modern symbol caches (run once before batch jobs)."""
    _load_or_build_symbol_table(tuple(sorted(releases)))
    _load_or_build_modern_table(modern_release)


def prepare_gene_metadata_for_pipeline(
    adata: ad.AnnData,
    *,
    releases: tuple[int, ...] = _DEFAULT_RELEASES,
    modern_release: int = _DEFAULT_MODERN_RELEASE,
    use_mygene_fallback: bool = True,
) -> ad.AnnData:
    """Resolve Ensembl IDs and modern symbols; set uniquified ``var_names``."""
    symbol_col = _first_var_column(adata, _SYMBOL_CANDIDATE_COLS)
    ensembl_col = _first_var_column(adata, _ENSEMBL_CANDIDATE_COLS)

    raw_var_names = adata.var_names.astype(str).map(_strip_dedup_suffix).to_numpy()
    sample = raw_var_names[: min(200, len(raw_var_names))]
    var_names_are_ensembl = (
        sample.size > 0 and float(np.mean([_looks_like_ensembl_id(n) for n in sample])) > 0.5
    )

    if ensembl_col is not None:
        input_ensembl = adata.var[ensembl_col].map(_normalize_ensembl_id).astype(str).to_numpy()
    elif var_names_are_ensembl:
        input_ensembl = np.array([_normalize_ensembl_id(n) for n in raw_var_names], dtype=object)
    else:
        input_ensembl = np.array([""] * adata.n_vars, dtype=object)

    if symbol_col is not None:
        input_symbols = adata.var[symbol_col].astype(str).to_numpy()
    elif not var_names_are_ensembl:
        input_symbols = raw_var_names.copy().astype(object)
    else:
        input_symbols = np.array([""] * adata.n_vars, dtype=object)

    input_symbols = np.array([str(s) for s in input_symbols], dtype=object)

    needs_id_mask = pd.Series(input_ensembl).eq("").to_numpy()
    if needs_id_mask.any():
        sym_to_id = _load_or_build_symbol_table(tuple(sorted(releases)))
        resolved = np.array(
            [sym_to_id.get(_strip_dedup_suffix(s), "") for s in input_symbols],
            dtype=object,
        )
        input_ensembl = np.where(needs_id_mask, resolved, input_ensembl)

        if use_mygene_fallback:
            still_missing_mask = (pd.Series(input_ensembl) == "").to_numpy()
            if still_missing_mask.any():
                missing_syms = sorted(
                    {
                        _strip_dedup_suffix(s)
                        for s, m in zip(input_symbols, still_missing_mask, strict=True)
                        if m and str(s).strip()
                    }
                )
                logger.info(
                    "Falling back to mygene for %d unresolved symbols.",
                    len(missing_syms),
                )
                mygene_map = _mygene_resolve(missing_syms)
                if mygene_map:
                    input_ensembl = np.array(
                        [
                            mygene_map.get(_strip_dedup_suffix(s), e) if e == "" else e
                            for s, e in zip(input_symbols, input_ensembl, strict=True)
                        ],
                        dtype=object,
                    )

    id_to_modern = _load_or_build_modern_table(modern_release)
    modern_symbols = np.array(
        [id_to_modern.get(e, "") for e in input_ensembl],
        dtype=object,
    )
    final_symbols = np.where(
        modern_symbols == "",
        input_symbols.astype(object),
        modern_symbols,
    )

    adata.var["original_symbol"] = input_symbols
    adata.var["ensembl_id"] = input_ensembl
    adata.var["gene_symbol"] = final_symbols
    adata.var["gene_name"] = final_symbols

    adata.var_names = pd.Index(adata.var["gene_symbol"].astype(str), name=None)
    if not adata.var_names.is_unique:
        adata.var_names_make_unique(join="_dup_")
    sync_var_index_for_write(adata)
    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique(join="_dup_")

    n_total = adata.n_vars
    n_with_ensg = int(np.sum(input_ensembl != ""))
    n_with_modern = int(np.sum(modern_symbols != ""))
    logger.info(
        "Resolved %d / %d (%.1f%%) to Ensembl IDs; %d / %d (%.1f%%) to modern symbols.",
        n_with_ensg,
        n_total,
        100.0 * n_with_ensg / max(n_total, 1),
        n_with_modern,
        n_total,
        100.0 * n_with_modern / max(n_total, 1),
    )

    return adata
