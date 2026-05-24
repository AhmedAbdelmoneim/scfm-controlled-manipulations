"""Discover datasets and models from checked-in Parquet bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from metrics_dashboard.bundle import (
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    SUMMARY_FILENAME,
    is_bundle_dataset_dir,
)
from metrics_dashboard.config import MODEL_ORDER, bundle_root


@dataclass(frozen=True)
class DatasetStatus:
    dataset_id: str
    n_models: int
    models: tuple[str, ...]
    last_modified: datetime | None
    n_cells: int | None = None


def discover_datasets(root: Path | None = None) -> list[str]:
    base = root or bundle_root()
    if not base.is_dir():
        return []
    return sorted(
        p.name
        for p in base.iterdir()
        if p.is_dir() and not p.name.startswith(".") and is_bundle_dataset_dir(p)
    )


def discover_models(dataset_id: str, root: Path | None = None) -> list[str]:
    base = root or bundle_root()
    manifest_path = base / dataset_id / MANIFEST_FILENAME
    if manifest_path.is_file():
        data = json.loads(manifest_path.read_text())
        uniq = [str(m) for m in data.get("models", [])]
    else:
        metrics_path = base / dataset_id / METRICS_FILENAME
        if not metrics_path.is_file():
            return []
        uniq = sorted(
            pd.read_parquet(metrics_path, columns=["model"])["model"].astype(str).unique()
        )
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(uniq, key=lambda m: order.get(m, len(MODEL_ORDER)))


def dataset_status(dataset_id: str, root: Path | None = None) -> DatasetStatus:
    base = root or bundle_root()
    ds_path = base / dataset_id
    if not is_bundle_dataset_dir(ds_path):
        return DatasetStatus(dataset_id, 0, (), None)

    metrics_path = ds_path / METRICS_FILENAME
    mtime = datetime.fromtimestamp(metrics_path.stat().st_mtime, tz=timezone.utc)
    models = tuple(discover_models(dataset_id, base))
    n_cells: int | None = None
    summary_path = ds_path / SUMMARY_FILENAME
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        n_cells = summary.get("n_cells")

    return DatasetStatus(
        dataset_id=dataset_id,
        n_models=len(models),
        models=models,
        last_modified=mtime,
        n_cells=n_cells,
    )


def catalog_table(root: Path | None = None) -> list[DatasetStatus]:
    return [dataset_status(ds, root) for ds in discover_datasets(root)]
