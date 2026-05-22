"""Discover datasets and models from evaluation artifacts on disk."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from metrics_dashboard.config import MODEL_ORDER, artifacts_root, evaluation_dir


@dataclass(frozen=True)
class DatasetStatus:
    dataset_id: str
    n_metric_csvs: int
    models: tuple[str, ...]
    last_modified: datetime | None
    has_evaluation: bool


def discover_datasets(root: Path | None = None) -> list[str]:
    """Dataset IDs under artifacts root (directories, sorted)."""
    base = root or artifacts_root()
    if not base.is_dir():
        return []
    return sorted(
        p.name
        for p in base.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def discover_models(eval_path: Path) -> list[str]:
    """Model names with ``{model}_metrics.csv`` in an evaluation directory."""
    if not eval_path.is_dir():
        return []
    models = sorted(p.stem.removesuffix("_metrics") for p in eval_path.glob("*_metrics.csv"))
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(models, key=lambda m: order.get(m, len(MODEL_ORDER)))


def dataset_status(dataset_id: str, root: Path | None = None) -> DatasetStatus:
    ev = evaluation_dir(dataset_id, root)
    if not ev.is_dir():
        return DatasetStatus(
            dataset_id=dataset_id,
            n_metric_csvs=0,
            models=(),
            last_modified=None,
            has_evaluation=False,
        )
    csvs = list(ev.glob("*_metrics.csv"))
    mtimes = [p.stat().st_mtime for p in csvs if p.is_file()]
    last_mod = None
    if mtimes:
        last_mod = datetime.fromtimestamp(max(mtimes), tz=timezone.utc)
    models = tuple(discover_models(ev))
    return DatasetStatus(
        dataset_id=dataset_id,
        n_metric_csvs=len(csvs),
        models=models,
        last_modified=last_mod,
        has_evaluation=len(csvs) > 0,
    )


def catalog_table(root: Path | None = None) -> list[DatasetStatus]:
    return [dataset_status(ds, root) for ds in discover_datasets(root)]
