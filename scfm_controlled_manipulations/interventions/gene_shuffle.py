from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from scfm_controlled_manipulations.base import Intervention

logger = logging.getLogger(__name__)


class GeneShuffle(Intervention):
    name = "gene_shuffle"
    variants = ("random", "stratified", "chromosome", "chromosome_control")
    cache_dir = Path.home() / ".cache" / "scfm-controlled-manipulations"

    def __init__(
        self,
        variant: str = "random",
        n_strata: int = 10,
        ensembl_id_column: str = "ensembl_id",
        species: str = "human",
        chromosome_cache_path: str | None = None,
    ):
        if variant not in self.variants:
            raise ValueError(f"variant must be one of: {', '.join(self.variants)}")
        if n_strata < 1:
            raise ValueError("n_strata must be at least 1")
        self.variant = variant
        self.n_strata = n_strata
        self.ensembl_id_column = ensembl_id_column
        self.species = species
        self.chromosome_cache_path = (
            Path(chromosome_cache_path).expanduser()
            if chromosome_cache_path
            else self.cache_dir / f"gene_chromosomes_{species}.csv"
        )

    @staticmethod
    def _normalize_ensembl_id(ensembl_id: Any) -> str:
        if pd.isna(ensembl_id):
            return ""
        return str(ensembl_id).split(".", maxsplit=1)[0]

    @staticmethod
    def _normalize_chromosome(chromosome: Any) -> str | None:
        if isinstance(chromosome, list | tuple):
            chromosome = chromosome[0] if chromosome else None
        if chromosome is None or pd.isna(chromosome):
            return None

        normalized = str(chromosome).strip()
        if not normalized:
            return None
        if normalized.lower().startswith("chr"):
            normalized = normalized[3:]
        return normalized.upper()

    @staticmethod
    def _shuffle_within_groups(
        groups: np.ndarray,
        rng: np.random.Generator,
        fixed_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        perm = np.arange(len(groups))
        eligible_mask = np.ones(len(groups), dtype=bool) if fixed_mask is None else ~fixed_mask
        for group in np.unique(groups[eligible_mask]):
            idx = np.where((groups == group) & eligible_mask)[0]
            if len(idx) > 1:
                perm[idx] = rng.permutation(idx)
        return perm

    @staticmethod
    def _group_sizes(groups: np.ndarray, include_mask: np.ndarray | None = None) -> dict[str, int]:
        included_groups = groups if include_mask is None else groups[include_mask]
        return {
            str(group): int(np.sum(included_groups == group))
            for group in np.unique(included_groups)
        }

    @staticmethod
    def _size_matched_control_groups(
        chromosome_labels: np.ndarray, mapped_mask: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        mapped_chromosome_labels = chromosome_labels[mapped_mask]
        group_sizes = [
            len(np.where(mapped_chromosome_labels == chromosome)[0])
            for chromosome in np.unique(mapped_chromosome_labels)
        ]
        shuffled_indices = rng.permutation(np.where(mapped_mask)[0])
        control_groups = np.empty(len(chromosome_labels), dtype=object)

        start = 0
        for group_idx, group_size in enumerate(group_sizes):
            idx = shuffled_indices[start : start + group_size]
            control_groups[idx] = f"control_{group_idx}"
            start += group_size

        return control_groups

    def _read_chromosome_cache(self) -> dict[str, str]:
        if not self.chromosome_cache_path.exists():
            logger.info("Chromosome cache not found at %s", self.chromosome_cache_path)
            return {}

        cache = pd.read_csv(self.chromosome_cache_path, dtype=str)
        required_columns = {"ensembl_id", "chromosome"}
        if not required_columns.issubset(cache.columns):
            raise ValueError(
                f"Chromosome cache {self.chromosome_cache_path} must contain columns: "
                f"{', '.join(sorted(required_columns))}"
            )

        chromosome_map = {}
        for row in cache.itertuples(index=False):
            ensembl_id = self._normalize_ensembl_id(row.ensembl_id)
            chromosome = self._normalize_chromosome(row.chromosome)
            if ensembl_id and chromosome:
                chromosome_map[ensembl_id] = chromosome
        logger.info(
            "Loaded %d chromosome mappings from %s",
            len(chromosome_map),
            self.chromosome_cache_path,
        )
        return chromosome_map

    def _write_chromosome_cache(self, chromosome_map: dict[str, str]) -> None:
        self.chromosome_cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache = pd.DataFrame(
            sorted(chromosome_map.items()),
            columns=["ensembl_id", "chromosome"],
        )
        cache.to_csv(self.chromosome_cache_path, index=False)
        logger.info(
            "Wrote %d chromosome mappings to %s",
            len(chromosome_map),
            self.chromosome_cache_path,
        )

    def _download_chromosomes(self, ensembl_ids: list[str]) -> dict[str, str]:
        import mygene

        logger.info(
            "Downloading chromosome labels for %d Ensembl IDs using species=%s",
            len(ensembl_ids),
            self.species,
        )
        mg = mygene.MyGeneInfo()
        records = mg.querymany(
            ensembl_ids,
            scopes="ensembl.gene",
            fields="chromosome",
            species=self.species,
            as_dataframe=False,
            verbose=False,
        )

        chromosome_map = {}
        for record in records:
            if record.get("notfound"):
                continue
            ensembl_id = self._normalize_ensembl_id(record["query"])
            chromosome = self._normalize_chromosome(record.get("chromosome"))
            if chromosome:
                chromosome_map[ensembl_id] = chromosome

        logger.info("Downloaded %d chromosome labels", len(chromosome_map))
        return chromosome_map

    def _chromosome_labels(
        self, adata: ad.AnnData
    ) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
        if self.ensembl_id_column not in adata.var:
            raise ValueError(
                f"gene_shuffle variant '{self.variant}' requires "
                f"adata.var['{self.ensembl_id_column}'] with Ensembl gene IDs"
            )

        ensembl_ids = [
            self._normalize_ensembl_id(ensembl_id)
            for ensembl_id in adata.var[self.ensembl_id_column]
        ]
        missing_ids = [idx for idx, ensembl_id in enumerate(ensembl_ids) if not ensembl_id]
        if missing_ids:
            raise ValueError(
                f"adata.var['{self.ensembl_id_column}'] contains empty Ensembl IDs at "
                f"{len(missing_ids)} positions"
            )

        unique_ids = list(dict.fromkeys(ensembl_ids))
        chromosome_map = self._read_chromosome_cache()
        missing_from_cache = [
            ensembl_id for ensembl_id in unique_ids if ensembl_id not in chromosome_map
        ]
        if missing_from_cache:
            chromosome_map.update(self._download_chromosomes(missing_from_cache))
            self._write_chromosome_cache(chromosome_map)

        chromosome_labels = np.asarray(
            [chromosome_map.get(ensembl_id) for ensembl_id in ensembl_ids], dtype=object
        )
        mapped_mask = np.asarray(
            [chromosome is not None for chromosome in chromosome_labels], dtype=bool
        )
        unmapped = [ensembl_id for ensembl_id in unique_ids if ensembl_id not in chromosome_map]
        if not mapped_mask.any():
            preview = ", ".join(unmapped[:5])
            raise ValueError(
                "Could not find chromosome labels for any genes "
                f"using species='{self.species}'. First missing IDs: {preview}"
            )
        if unmapped:
            logger.info(
                "Leaving %d unmapped genes fixed during %s shuffle",
                len(unmapped),
                self.variant,
            )

        return chromosome_labels, ensembl_ids, mapped_mask, unmapped

    def apply(self, adata: ad.AnnData, seed: int | None = None) -> ad.AnnData:
        rng = np.random.default_rng(seed)
        n = adata.n_vars
        group_sizes = {}
        unmapped_ensembl_ids = []

        if self.variant == "random":
            perm = rng.permutation(n)
        elif self.variant == "stratified":
            # Bin by mean expression of non-zero entries to avoid
            # zero-inflation dominating the low bin
            X = adata.X
            with np.errstate(divide="ignore", invalid="ignore"):
                col_sum = np.asarray(X.sum(axis=0)).flatten()
                col_nnz = np.asarray((X > 0).sum(axis=0)).flatten()
                mean_nz = np.where(col_nnz > 0, col_sum / col_nnz, 0)
            ranks = np.argsort(np.argsort(mean_nz))
            bin_size = max(1, n // self.n_strata)
            bins = np.minimum(ranks // bin_size, self.n_strata - 1)
            perm = np.arange(n)
            for b in range(self.n_strata):
                idx = np.where(bins == b)[0]
                if len(idx) > 1:
                    perm[idx] = rng.permutation(idx)
        else:
            chromosome_labels, _, mapped_mask, unmapped_ensembl_ids = self._chromosome_labels(
                adata
            )
            groups = (
                chromosome_labels
                if self.variant == "chromosome"
                else self._size_matched_control_groups(chromosome_labels, mapped_mask, rng)
            )
            group_sizes = self._group_sizes(groups, include_mask=mapped_mask)
            perm = self._shuffle_within_groups(groups, rng, fixed_mask=~mapped_mask)

        out = adata.copy()
        # Keep X fixed and shuffle gene annotations so models see counts under perturbed gene IDs.
        out.var = adata.var.take(perm).copy()
        out.uns["scfm_intervention"] = {
            self.name: {
                "variant": self.variant,
                "n_strata": self.n_strata,
                "ensembl_id_column": self.ensembl_id_column,
                "species": self.species,
                "chromosome_cache_path": str(self.chromosome_cache_path),
                "group_sizes": group_sizes,
                "unmapped_gene_count": len(unmapped_ensembl_ids),
                "unmapped_ensembl_ids": unmapped_ensembl_ids,
                "seed": seed,
                "permutation": perm.tolist(),
            }
        }
        return out
