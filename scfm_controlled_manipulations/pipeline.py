from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd
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


def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_manipulate(config: dict[str, Any]) -> None:
    """Apply each configured intervention and write ``manipulations/{intervention_id}.h5ad``."""
    input_path = Path(config["input_h5ad"])
    results_dir = Path(config["results_dir"])
    seed = config.get("seed")

    adata_in = ad.read_h5ad(input_path)

    manip_dir = results_dir / "manipulations"
    manip_dir.mkdir(parents=True, exist_ok=True)

    for spec in config["interventions"]:
        name = spec["name"]
        kwargs = dict(spec.get("kwargs") or {})
        iid = intervention_id(name, kwargs)
        cls = interventions.REGISTRY[name]
        intervention = cls(**kwargs)
        out = intervention.apply(adata_in, seed=seed)
        out_path = manipulation_path(results_dir, iid)
        out.write_h5ad(out_path)


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

        for spec in config["interventions"]:
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
