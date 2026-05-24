"""Discover datasets and models from dashboard bundles or legacy SCEval trees."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from metrics_dashboard.bundle import (
    METRICS_FILENAME,
    discover_datasets_at_root,
    is_bundle_dataset_dir,
    is_legacy_dataset_dir,
)
from metrics_dashboard.config import MODEL_ORDER, artifacts_root


@dataclass(frozen=True)
class DatasetStatus:
    dataset_id: str
    n_metric_csvs: int
    models: tuple[str, ...]
    last_modified: datetime | None
    has_evaluation: bool


def discover_datasets(root: Path | None = None) -> list[str]:
    base = root or artifacts_root()
    if not base.is_dir():
        return []
    return discover_datasets_at_root(base)


def discover_models(dataset_dir: Path) -> list[str]:
    """Model names from ``metrics.parquet`` or legacy ``*_metrics.csv`` files."""
    if (dataset_dir / METRICS_FILENAME).is_file():
        models = pd.read_parquet(dataset_dir / METRICS_FILENAME, columns=["model"])["model"]
        uniq = sorted(models.astype(str).unique())
    else:
        ev = dataset_dir / "results" / "evaluation"
        if not ev.is_dir():
            return []
        uniq = sorted(p.stem.removesuffix("_metrics") for p in ev.glob("*_metrics.csv"))
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(uniq, key=lambda m: order.get(m, len(MODEL_ORDER)))


def dataset_status(dataset_id: str, root: Path | None = None) -> DatasetStatus:
    base = root or artifacts_root()
    ds_path = base / dataset_id

    if is_bundle_dataset_dir(ds_path):
        metrics_path = ds_path / METRICS_FILENAME
        mtime = datetime.fromtimestamp(metrics_path.stat().st_mtime, tz=timezone.utc)
        models = tuple(discover_models(ds_path))
        return DatasetStatus(
            dataset_id=dataset_id,
            n_metric_csvs=len(models),
            models=models,
            last_modified=mtime,
            has_evaluation=True,
        )

    if is_legacy_dataset_dir(ds_path):
        ev = ds_path / "results" / "evaluation"
        csvs = list(ev.glob("*_metrics.csv"))
        mtimes = [p.stat().st_mtime for p in csvs if p.is_file()]
        last_mod = (
            datetime.fromtimestamp(max(mtimes), tz=timezone.utc) if mtimes else None
        )
        models = tuple(discover_models(ds_path))
        return DatasetStatus(
            dataset_id=dataset_id,
            n_metric_csvs=len(csvs),
            models=models,
            last_modified=last_mod,
            has_evaluation=len(csvs) > 0,
        )

    return DatasetStatus(
        dataset_id=dataset_id,
        n_metric_csvs=0,
        models=(),
        last_modified=None,
        has_evaluation=False,
    )


def catalog_table(root: Path | None = None) -> list[DatasetStatus]:
    return [dataset_status(ds, root) for ds in discover_datasets(root)]
