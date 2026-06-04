"""Validate manipulation h5ad files for transcriptomic-fms embedding."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scfm_controlled_manipulations.fm_gene_resolver import _looks_like_ensembl_id

_ENSEMBL_COL_CANDIDATES = (
    "ensembl_id",
    "ensembl_gene_id",
    "ensembl ids",
    "ensembl_ids",
    "gene_id",
    "gene_ids",
)
_SYMBOL_COL_CANDIDATES = (
    "gene_name",
    "gene_symbol",
    "gene_symbols",
    "gene_names",
    "feature_name",
    "symbol",
)
_EMBED_OUTPUT_SUFFIXES = ("_embeddings.h5ad", "_emb.h5ad")

_ENSEMBL_MODELS = ("geneformer", "scconcept")


class Level(str, Enum):
    OK = "OK"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass
class Finding:
    level: Level
    message: str


@dataclass
class FileReport:
    path: Path
    findings: list[Finding] = field(default_factory=list)

    @property
    def worst_level(self) -> Level:
        if any(f.level == Level.ERROR for f in self.findings):
            return Level.ERROR
        if any(f.level == Level.WARN for f in self.findings):
            return Level.WARN
        return Level.OK


def _add(report: FileReport, level: Level, message: str) -> None:
    report.findings.append(Finding(level=level, message=message))


def _first_var_column(adata: ad.AnnData, candidates: Sequence[str]) -> str | None:
    return next((c for c in candidates if c in adata.var.columns), None)


def _sample_matrix_values(x: sp.spmatrix | np.ndarray, max_values: int = 200_000) -> np.ndarray:
    if sp.issparse(x):
        data = x.data
    else:
        data = np.asarray(x).ravel()
    if data.size == 0:
        return data.astype(np.float64)
    if data.size <= max_values:
        return data.astype(np.float64)
    rng = np.random.default_rng(0)
    idx = rng.choice(data.size, size=max_values, replace=False)
    return data[idx].astype(np.float64)


def check_raw_counts_in_x(x: sp.spmatrix | np.ndarray) -> list[Finding]:
    """Heuristic: manipulated files should store raw counts in ``adata.X``."""
    findings: list[Finding] = []
    values = _sample_matrix_values(x)
    if values.size == 0:
        findings.append(Finding(Level.ERROR, "X is empty"))
        return findings

    if np.any(values < 0):
        findings.append(Finding(Level.ERROR, "X contains negative values (expected raw counts)"))
        return findings

    rounded = np.round(values)
    integer_like = float(np.mean(np.isclose(values, rounded, rtol=0, atol=1e-5)))
    if integer_like < 0.9:
        findings.append(
            Finding(
                Level.ERROR,
                f"X does not look like integer raw counts "
                f"(integer-like fraction={integer_like:.3f})",
            )
        )
    elif integer_like < 0.99:
        findings.append(
            Finding(
                Level.WARN,
                f"X is mostly integer-like but not fully "
                f"(integer-like fraction={integer_like:.3f})",
            )
        )

    positive = values[values > 0]
    if positive.size > 0:
        frac_fractional = float(np.mean((positive % 1) > 1e-5))
        if frac_fractional > 0.05:
            findings.append(
                Finding(
                    Level.WARN,
                    f"Many non-zero X entries are non-integer (fraction={frac_fractional:.3f})",
                )
            )
        max_val = float(np.max(positive))
        if (
            integer_like < 0.95
            and max_val < 25
            and float(np.mean(positive < 15)) > 0.8
        ):
            findings.append(
                Finding(
                    Level.WARN,
                    "X value range resembles log-normalized data (max<25, most values<15)",
                )
            )

    return findings


def extract_gene_symbols(adata: ad.AnnData) -> list[str]:
    """Mirror transcriptomic_fms BaseEmbeddingModel._get_gene_symbols priority."""
    col = _first_var_column(adata, _SYMBOL_COL_CANDIDATES)
    if col is not None:
        return adata.var[col].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def extract_ensembl_ids(adata: ad.AnnData, *, ensembl_col: str = "ensembl_id") -> list[str]:
    if ensembl_col in adata.var.columns:
        return adata.var[ensembl_col].astype(str).tolist()
    col = _first_var_column(adata, _ENSEMBL_COL_CANDIDATES)
    if col is not None:
        return adata.var[col].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def _normalize_gene_set(genes: Iterable[str], *, uppercase: bool = False) -> set[str]:
    out: set[str] = set()
    for g in genes:
        s = str(g).strip()
        if not s or s.lower() == "nan":
            continue
        out.add(s.upper() if uppercase else s)
    return out


def load_gene_list_file(path: Path) -> list[str]:
    """Load a model gene list (one symbol/ID per line, or TSV with gene_name)."""
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return [str(k) for k in data]
        if isinstance(data, list):
            return [str(x) for x in data]
        raise ValueError(f"Unsupported JSON gene list format: {path}")

    if path.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t")
        if "gene_name" in df.columns:
            return df["gene_name"].astype(str).tolist()

    lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return lines


def discover_fm_gene_lists(
    fm_repo: Path,
    *,
    extra: Mapping[str, Path] | None = None,
) -> dict[str, Path]:
    """Find known model gene-list files under a transcriptomic-fms checkout."""
    fm_repo = fm_repo.expanduser().resolve()
    found: dict[str, Path] = dict(extra or {})

    scf_index = fm_repo / "OS_scRNA_gene_index.19264.tsv"
    if scf_index.is_file():
        found.setdefault("scfoundation", scf_index)
    for path in fm_repo.rglob("OS_scRNA_gene_index.19264.tsv"):
        found.setdefault("scfoundation", path)
        break

    for path in fm_repo.rglob("vocab.json"):
        lower = str(path).lower()
        if "scgpt" in lower or path.parent.name.lower() in {"scgpt_human", "scgpt"}:
            found.setdefault("scgpt", path)
            break

    models_root = fm_repo / "models"
    if models_root.is_dir():
        for path in models_root.rglob("vocab.json"):
            found.setdefault("scgpt", path)
            break
        for path in models_root.rglob("OS_scRNA_gene_index.19264.tsv"):
            found.setdefault("scfoundation", path)
            break

    return found


def parse_gene_list_arg(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected MODEL=path, got: {spec!r}")
    name, raw_path = spec.split("=", maxsplit=1)
    name = name.strip().lower()
    if not name:
        raise ValueError(f"Empty model name in: {spec!r}")
    return name, Path(raw_path.strip())


def check_structure(adata: ad.AnnData) -> list[Finding]:
    findings: list[Finding] = []
    n_obs, n_vars = adata.shape
    if adata.n_obs != n_obs or adata.n_vars != n_vars:
        findings.append(Finding(Level.ERROR, "AnnData internal shape inconsistent"))
    if len(adata.obs_names) != n_obs:
        findings.append(Finding(Level.ERROR, "obs index length does not match n_obs"))
    if len(adata.var_names) != n_vars:
        findings.append(Finding(Level.ERROR, "var index length does not match n_vars"))

    if not adata.obs_names.is_unique:
        n_dup = int(adata.obs_names.duplicated().sum())
        findings.append(Finding(Level.ERROR, f"obs_names are not unique ({n_dup} duplicates)"))
    if not adata.var_names.is_unique:
        n_dup = int(adata.var_names.duplicated().sum())
        findings.append(Finding(Level.ERROR, f"var_names are not unique ({n_dup} duplicates)"))

    if adata.raw is not None:
        findings.append(
            Finding(
                Level.WARN,
                "adata.raw is set (manipulation outputs are usually slimmed without .raw)",
            )
        )
    if adata.layers:
        findings.append(
            Finding(
                Level.WARN,
                f"adata.layers present ({', '.join(adata.layers.keys())}); "
                "embed pipeline expects counts in .X",
            )
        )
    return findings


def check_gene_metadata(
    adata: ad.AnnData,
    *,
    gene_name_column: str = "gene_name",
    ensembl_id_column: str = "ensembl_id",
) -> list[Finding]:
    findings: list[Finding] = []
    symbols = adata.var_names.astype(str)
    sample = symbols[: min(200, len(symbols))]
    ensembl_frac = (
        float(np.mean([_looks_like_ensembl_id(v) for v in sample])) if sample.size else 0.0
    )
    if ensembl_frac > 0.5:
        findings.append(
            Finding(
                Level.ERROR,
                "var_names look like Ensembl IDs; expected gene symbols in var_names",
            )
        )

    if gene_name_column not in adata.var.columns:
        findings.append(
            Finding(Level.ERROR, f"Missing adata.var['{gene_name_column}'] (pipeline output)")
        )
    else:
        gene_names = adata.var[gene_name_column].astype(str)
        empty = int(
            ((gene_names.str.strip() == "") | adata.var[gene_name_column].isna()).sum()
        )
        if empty:
            findings.append(
                Finding(
                    Level.ERROR,
                    f"adata.var['{gene_name_column}'] has {empty} missing/empty entries",
                )
            )
        mism = int((gene_names.to_numpy() != symbols.to_numpy()).sum())
        if mism:
            findings.append(
                Finding(
                    Level.ERROR,
                    f"var_names disagree with var['{gene_name_column}'] on {mism} genes",
                )
            )

    ens_col = ensembl_id_column if ensembl_id_column in adata.var.columns else None
    if ens_col is None:
        ens_col = _first_var_column(adata, _ENSEMBL_COL_CANDIDATES)
    if ens_col is None:
        findings.append(
            Finding(
                Level.ERROR,
                f"Missing Ensembl ID column (expected var['{ensembl_id_column}'])",
            )
        )
    else:
        ensembl = adata.var[ens_col].astype(str).map(lambda v: str(v).strip())
        empty_ens = int(((ensembl == "") | (ensembl.str.lower() == "nan")).sum())
        if empty_ens:
            findings.append(
                Finding(
                    Level.ERROR,
                    f"var['{ens_col}'] has {empty_ens} missing Ensembl IDs",
                )
            )
        sample_ens = ensembl.head(min(200, len(ensembl)))
        valid = float(np.mean([_looks_like_ensembl_id(v) for v in sample_ens]))
        if valid < 0.8:
            findings.append(
                Finding(
                    Level.WARN,
                    f"var['{ens_col}'] entries often do not match Ensembl ID pattern "
                    f"(valid fraction in sample={valid:.3f})",
                )
            )

    return findings


def check_model_gene_overlap(
    adata: ad.AnnData,
    model_gene_lists: Mapping[str, Sequence[str]],
    *,
    min_symbol_overlap: float = 0.1,
    min_ensembl_overlap: float = 0.1,
) -> list[Finding]:
    """Report overlap between file genes and optional per-model reference lists."""
    findings: list[Finding] = []
    if not model_gene_lists:
        return findings

    file_symbols = _normalize_gene_set(extract_gene_symbols(adata))
    file_symbols_upper = _normalize_gene_set(file_symbols, uppercase=True)
    file_ensembl = {
        str(v).split(".", maxsplit=1)[0]
        for v in extract_ensembl_ids(adata)
        if str(v).strip() and str(v).lower() != "nan"
    }

    for model, genes in model_gene_lists.items():
        model = model.lower()
        ref = _normalize_gene_set(genes)
        if not ref:
            findings.append(Finding(Level.WARN, f"{model}: gene list is empty"))
            continue

        if model in _ENSEMBL_MODELS:
            ref_ens = {str(g).split(".", maxsplit=1)[0] for g in ref}
            overlap = len(file_ensembl & ref_ens) / len(ref_ens)
            metric = "Ensembl IDs"
            threshold = min_ensembl_overlap
        else:
            ref_upper = _normalize_gene_set(ref, uppercase=True)
            overlap = len(file_symbols_upper & ref_upper) / len(ref_upper)
            metric = "gene symbols"
            threshold = min_symbol_overlap

        pct = 100.0 * overlap
        if overlap < threshold:
            findings.append(
                Finding(
                    Level.ERROR,
                    f"{model}: only {pct:.1f}% of model {metric} found in file "
                    f"({len(ref)} reference genes)",
                )
            )
        elif overlap < 0.5:
            findings.append(
                Finding(
                    Level.WARN,
                    f"{model}: {pct:.1f}% of model {metric} found in file",
                )
            )
        else:
            findings.append(
                Finding(
                    Level.OK,
                    f"{model}: {pct:.1f}% of model {metric} found in file",
                )
            )
    return findings


def validate_h5ad_file(
    path: Path,
    *,
    gene_name_column: str = "gene_name",
    ensembl_id_column: str = "ensembl_id",
    model_gene_lists: Mapping[str, Sequence[str]] | None = None,
    min_symbol_overlap: float = 0.1,
    min_ensembl_overlap: float = 0.1,
) -> FileReport:
    report = FileReport(path=path)
    try:
        adata = ad.read_h5ad(path)
    except Exception as exc:
        _add(report, Level.ERROR, f"Failed to read h5ad: {exc}")
        return report

    for finding in check_structure(adata):
        _add(report, finding.level, finding.message)
    for finding in check_raw_counts_in_x(adata.X):
        _add(report, finding.level, finding.message)
    for finding in check_gene_metadata(
        adata,
        gene_name_column=gene_name_column,
        ensembl_id_column=ensembl_id_column,
    ):
        _add(report, finding.level, finding.message)

    try:
        symbols = extract_gene_symbols(adata)
        if len(set(symbols)) < len(symbols):
            _add(
                report,
                Level.WARN,
                "Duplicate gene symbols in feature metadata (var_names should be unique)",
            )
    except Exception as exc:
        _add(report, Level.ERROR, f"Cannot extract gene symbols: {exc}")

    if model_gene_lists:
        for finding in check_model_gene_overlap(
            adata,
            model_gene_lists,
            min_symbol_overlap=min_symbol_overlap,
            min_ensembl_overlap=min_ensembl_overlap,
        ):
            if finding.level != Level.OK:
                _add(report, finding.level, finding.message)
            else:
                _add(report, Level.OK, finding.message)

    if not any(f.level != Level.OK for f in report.findings):
        _add(report, Level.OK, "All checks passed")
    return report


def iter_h5ad_paths(
    directory: Path,
    *,
    skip_embed_outputs: bool = True,
) -> list[Path]:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    paths = sorted(directory.glob("*.h5ad"))
    if not skip_embed_outputs:
        return paths
    out: list[Path] = []
    for path in paths:
        name = path.name
        if any(name.endswith(suffix) for suffix in _EMBED_OUTPUT_SUFFIXES):
            continue
        out.append(path)
    return out


def load_model_gene_lists(
    *,
    fm_repo: Path | None = None,
    gene_list_specs: Sequence[str] = (),
    manipulations_dir: Path | None = None,
) -> dict[str, list[str]]:
    lists: dict[str, list[str]] = {}

    for spec in gene_list_specs:
        name, path = parse_gene_list_arg(spec)
        lists[name] = load_gene_list_file(path)

    if fm_repo is not None:
        for model, path in discover_fm_gene_lists(fm_repo).items():
            if model not in lists:
                lists[model] = load_gene_list_file(path)

    if manipulations_dir is not None:
        hvg = manipulations_dir / "hvg.txt"
        if hvg.is_file() and "hvg" not in lists:
            lists["hvg"] = load_gene_list_file(hvg)

    return lists


def format_report(reports: Sequence[FileReport], *, directory: Path) -> str:
    lines = [
        f"=== embed input validation: {directory} ({len(reports)} files) ===",
        "",
    ]
    n_err = n_warn = n_ok = 0
    for report in reports:
        level = report.worst_level
        if level == Level.ERROR:
            n_err += 1
        elif level == Level.WARN:
            n_warn += 1
        else:
            n_ok += 1
        lines.append(f"{level.value:5}  {report.path.name}")
        shown = [f for f in report.findings if f.level != Level.OK or len(report.findings) == 1]
        for finding in shown:
            if finding.level == Level.OK and finding.message == "All checks passed":
                continue
            prefix = finding.level.value
            lines.append(f"       [{prefix}] {finding.message}")
        lines.append("")

    lines.append(f"Summary: {n_ok} ok, {n_warn} warn, {n_err} error (of {len(reports)} files)")
    return "\n".join(lines)


def _reference_path(paths: Sequence[Path]) -> Path:
    for candidate in paths:
        if candidate.name == "reference.h5ad":
            return candidate
    return paths[0]


def check_cross_file_consistency(
    directory: Path,
    paths: Sequence[Path],
) -> list[Finding]:
    """Ensure manipulations share the same cell and gene indices as the reference file."""
    if len(paths) < 2:
        return []

    ref_path = _reference_path(paths)
    try:
        ref = ad.read_h5ad(ref_path, backed="r")
        ref_obs = list(ref.obs_names)
        ref_var = list(ref.var_names)
        ref.close()
    except Exception as exc:
        return [Finding(Level.ERROR, f"Cannot read reference {ref_path.name}: {exc}")]

    findings: list[Finding] = []
    ref_obs_set = set(ref_obs)
    ref_var_set = set(ref_var)

    for path in paths:
        if path == ref_path:
            continue
        try:
            adata = ad.read_h5ad(path, backed="r")
            obs = list(adata.obs_names)
            var = list(adata.var_names)
            adata.file.close()
        except Exception as exc:
            findings.append(Finding(Level.ERROR, f"{path.name}: cannot read for cross-check: {exc}"))
            continue

        if obs != ref_obs:
            missing_obs = len(ref_obs_set - set(obs))
            extra_obs = len(set(obs) - ref_obs_set)
            findings.append(
                Finding(
                    Level.ERROR,
                    f"{path.name}: obs_names differ from {ref_path.name} "
                    f"(missing={missing_obs}, extra={extra_obs}, order_mismatch={obs != ref_obs})",
                )
            )
        if var != ref_var:
            missing_var = len(ref_var_set - set(var))
            extra_var = len(set(var) - ref_var_set)
            findings.append(
                Finding(
                    Level.ERROR,
                    f"{path.name}: var_names differ from {ref_path.name} "
                    f"(missing={missing_var}, extra={extra_var}, order_mismatch={var != ref_var})",
                )
            )

    if not findings:
        findings.append(
            Finding(
                Level.OK,
                f"All {len(paths) - 1} files match {ref_path.name} obs/var names and order",
            )
        )
    return findings


def validate_directory(
    directory: Path,
    *,
    gene_name_column: str = "gene_name",
    ensembl_id_column: str = "ensembl_id",
    model_gene_lists: Mapping[str, Sequence[str]] | None = None,
    min_symbol_overlap: float = 0.1,
    min_ensembl_overlap: float = 0.1,
) -> tuple[list[FileReport], int]:
    paths = iter_h5ad_paths(directory)
    if not paths:
        raise FileNotFoundError(f"No .h5ad files found in {directory}")

    reports = [
        validate_h5ad_file(
            path,
            gene_name_column=gene_name_column,
            ensembl_id_column=ensembl_id_column,
            model_gene_lists=model_gene_lists,
            min_symbol_overlap=min_symbol_overlap,
            min_ensembl_overlap=min_ensembl_overlap,
        )
        for path in paths
    ]

    cross = check_cross_file_consistency(directory, paths)
    if cross:
        summary = FileReport(path=directory / "[cross-file]")
        for finding in cross:
            if finding.level != Level.OK:
                _add(summary, finding.level, finding.message)
            elif len(paths) > 1:
                _add(summary, Level.OK, finding.message)
        if summary.findings:
            reports.append(summary)

    exit_code = 1 if any(r.worst_level == Level.ERROR for r in reports) else 0
    return reports, exit_code
