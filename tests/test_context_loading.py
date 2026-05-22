"""Integration-style tests for evaluation data loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scfm_controlled_manipulations.evaluation.context import (
    load_dataset_context,
    load_intervention_bundle,
    load_model_context,
)
from scfm_controlled_manipulations.evaluation.data import dense_embedding_aligned_to_obs
from scfm_controlled_manipulations.io import embedding_path


class ContextLoadingTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[str, list[str]]:
        obs_names = [f"cell_{i}" for i in range(8)]
        obs = {"cell_type": ["a", "b", "a", "b", "a", "b", "a", "b"]}
        var = {"gene_name": [f"g{i}" for i in range(5)]}
        rng = np.random.default_rng(0)

        manip_dir = root / "results" / "manipulations"
        manip_dir.mkdir(parents=True)
        emb_root = root / "embeddings" / "pca"
        emb_root.mkdir(parents=True)

        ref_x = sp.csr_matrix(rng.poisson(1, size=(8, 5)).astype(np.float64))
        ad_ref = ad.AnnData(X=ref_x, obs=obs, var=var)
        ad_ref.obs_names = obs_names
        ad_ref.write_h5ad(manip_dir / "reference.h5ad")

        shuffled_obs = list(reversed(obs_names))
        ad_man = ad.AnnData(X=ref_x.copy(), obs=obs, var=var)
        ad_man.obs_names = obs_names
        ad_man.write_h5ad(manip_dir / "downsample_abc123.h5ad")

        ref_emb = rng.standard_normal((8, 3)).astype(np.float32)
        ad_emb_ref = ad.AnnData(X=ref_emb)
        ad_emb_ref.obs_names = obs_names
        ad_emb_ref.write_h5ad(emb_root / "pca_reference.h5ad")

        man_emb = rng.standard_normal((8, 3)).astype(np.float32)
        ad_emb_man = ad.AnnData(X=man_emb)
        ad_emb_man.obs_names = shuffled_obs
        ad_emb_man.write_h5ad(emb_root / "pca_downsample_abc123.h5ad")
        return "downsample_abc123", obs_names

    def test_dense_embedding_aligned_to_obs_reorders(self) -> None:
        adata = ad.AnnData(
            X=np.arange(6, dtype=np.float32).reshape(3, 2),
            obs={"id": ["b", "a", "c"]},
        )
        adata.obs_names = ["b", "a", "c"]
        aligned = dense_embedding_aligned_to_obs(adata, pd.Index(["a", "b", "c"]), label="emb")
        np.testing.assert_allclose(aligned[:, 0], [2, 0, 4])

    def test_load_intervention_bundle_aligns_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            iid, obs_names = self._write_fixture(root)
            dataset_ctx = load_dataset_context(root / "results")
            model_ctx = load_model_context(
                root / "embeddings",
                "pca",
                "reference",
                target_obs=dataset_ctx.obs.index,
            )
            bundle = load_intervention_bundle(
                dataset_ctx=dataset_ctx,
                model_ctx=model_ctx,
                results_dir=root / "results",
                embeddings_root=root / "embeddings",
                model="pca",
                intervention_id=iid,
            )
            self.assertEqual(bundle.raw_ref.shape[0], 8)
            self.assertEqual(bundle.emb_man.shape, (8, 3))
            self.assertTrue(np.all(dataset_ctx.obs.index == obs_names))
            self.assertEqual(
                embedding_path(root / "embeddings", "pca", iid).name,
                "pca_downsample_abc123.h5ad",
            )


if __name__ == "__main__":
    unittest.main()
