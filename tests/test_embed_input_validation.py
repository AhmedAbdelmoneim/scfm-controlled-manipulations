"""Tests for embed-ready h5ad validation."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import anndata as ad
import numpy as np
import scipy.sparse as sp

from scfm_controlled_manipulations.embed_input_validation import (
    Level,
    check_raw_counts_in_x,
    load_gene_list_file,
    validate_directory,
    validate_h5ad_file,
)


def _make_valid_adata(n_obs: int = 5, n_vars: int = 8) -> ad.AnnData:
    rng = np.random.default_rng(0)
    x = rng.poisson(3, size=(n_obs, n_vars)).astype(np.float32)
    symbols = [f"GENE{i}" for i in range(n_vars)]
    ensembl = [f"ENSG0000000000{i:02d}" for i in range(n_vars)]
    adata = ad.AnnData(
        X=sp.csr_matrix(x),
        obs={"batch": ["a"] * n_obs},
        var={"gene_name": symbols, "ensembl_id": ensembl},
    )
    adata.var_names = symbols
    adata.obs_names = [f"cell{i}" for i in range(n_obs)]
    return adata


class TestEmbedInputValidation(unittest.TestCase):
    def test_raw_counts_check_passes_integers(self) -> None:
        adata = _make_valid_adata()
        findings = check_raw_counts_in_x(adata.X)
        self.assertFalse(any(f.level == Level.ERROR for f in findings))

    def test_raw_counts_check_fails_floats(self) -> None:
        x = np.array([[0.1, 1.5], [2.2, 0.0]], dtype=np.float32)
        findings = check_raw_counts_in_x(x)
        self.assertTrue(any(f.level == Level.ERROR for f in findings))

    def test_validate_good_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.h5ad"
            _make_valid_adata().write_h5ad(path)
            report = validate_h5ad_file(path)
            self.assertEqual(report.worst_level, Level.OK)

    def test_validate_detects_ensembl_var_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adata = _make_valid_adata()
            adata.var_names = adata.var["ensembl_id"].astype(str)
            path = Path(tmp) / "bad.h5ad"
            adata.write_h5ad(path)
            report = validate_h5ad_file(path)
            self.assertEqual(report.worst_level, Level.ERROR)

    def test_model_overlap_gene_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.h5ad"
            _make_valid_adata().write_h5ad(path)
            gene_list_path = Path(tmp) / "genes.txt"
            gene_list_path.write_text("GENE0\nGENE1\nMISSING\n")
            report = validate_h5ad_file(
                path,
                model_gene_lists={"scgpt": load_gene_list_file(gene_list_path)},
                min_symbol_overlap=0.9,
            )
            self.assertEqual(report.worst_level, Level.ERROR)

    def test_validate_directory_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_valid_adata().write_h5ad(root / "a.h5ad")
            reports, code = validate_directory(root)
            self.assertEqual(len(reports), 1)
            self.assertEqual(code, 0)

    def test_cross_file_var_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = _make_valid_adata()
            ref.write_h5ad(root / "reference.h5ad")
            other = _make_valid_adata(n_vars=6)
            other.write_h5ad(root / "other.h5ad")
            reports, code = validate_directory(root)
            self.assertEqual(code, 1)
            cross = [r for r in reports if r.path.name == "[cross-file]"]
            self.assertEqual(len(cross), 1)
            self.assertEqual(cross[0].worst_level, Level.ERROR)


if __name__ == "__main__":
    unittest.main()
