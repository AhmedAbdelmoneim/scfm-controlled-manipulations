"""ViScore structure preservation, Székely distance correlation, and intrinsic dimension."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import skdim
import viscore

from scfm_controlled_manipulations.evaluation.metrics_common import (
    NAN_SUMMARY,
    make_metric_row,
    scalar_summary,
)

logger = logging.getLogger(__name__)

_SPACE = "embedding"
_MIN_TWONN_CELLS = 20


def distcorr_from_distances(d_ref: np.ndarray, d_man: np.ndarray) -> float:
    """Székely distance correlation from two square distance matrices."""

    def _center(d: np.ndarray) -> np.ndarray:
        return d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + d.mean()

    a, b = _center(d_ref), _center(d_man)
    n = a.shape[0]
    norm_a = np.sqrt((a * a).sum()) / n
    norm_b = np.sqrt((b * b).sum()) / n
    den = norm_a * norm_b
    return float((a * b).sum() / (n * n) / den) if den > 0.0 else 0.0


def _distcorr_pair(ref: np.ndarray, man: np.ndarray) -> float:
    d_ref = squareform(pdist(ref.astype(np.float64)))
    d_man = squareform(pdist(man.astype(np.float64)))
    return distcorr_from_distances(d_ref, d_man)


def _rnx_curve_json(rnx: np.ndarray, *, n_cells: int) -> str:
    n_emb = int(n_cells)
    k = np.arange(1, len(rnx) + 1, dtype=int)
    u = np.arange(1, n_emb - 1, dtype=np.float64) / n_emb
    return json.dumps({"k": k.tolist(), "u": u.tolist(), "rnx": np.asarray(rnx).tolist()})


def _estimate_twonn(mat: np.ndarray) -> float:
    if mat.shape[0] < _MIN_TWONN_CELLS:
        return float("nan")
    try:
        est = skdim.id.TwoNN()
        est.fit(mat.astype(np.float64, copy=False))
        return float(est.dimension_)
    except Exception as err:
        logger.warning("TwoNN failed (n=%d, d=%d): %s", mat.shape[0], mat.shape[1], err)
        return float("nan")


def _estimate_participation_ratio(mat: np.ndarray) -> float:
    if mat.shape[0] < 2:
        return float("nan")
    try:
        est = skdim.id.lPCA(ver="participation_ratio", verbose=False)
        est.fit(mat.astype(np.float64, copy=False))
        return float(est.dimension_)
    except Exception as err:
        logger.warning(
            "Participation ratio failed (n=%d, d=%d): %s", mat.shape[0], mat.shape[1], err
        )
        return float("nan")


def _append_viscore_rows(
    rows: list[dict[str, Any]],
    *,
    ref: np.ndarray,
    man: np.ndarray,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    seed: int,
    n_cells: int,
) -> None:
    score = viscore.score(ref, man)
    sl = float(score["Sl"])
    sg = float(score["Sg"])
    rnx = np.asarray(score["RNX"], dtype=np.float64)

    base = dict(
        dataset_id=dataset_id,
        model=model,
        intervention_id=intervention_id,
        intervention_name=intervention_name,
        category="structure_metrics",
        space=_SPACE,
        n_cells=n_cells,
        seed=seed,
    )
    for metric_name, value in (
        ("viscore_local_sp", sl),
        ("viscore_global_sp", sg),
    ):
        rows.append(
            make_metric_row(
                dataset_id=base["dataset_id"],
                model=base["model"],
                intervention_id=base["intervention_id"],
                intervention_name=base["intervention_name"],
                metric_category=base["category"],
                metric_name=metric_name,
                space=base["space"],
                summary=scalar_summary(value),
                n_cells=base["n_cells"],
                seed=base["seed"],
            )
        )
    rows.append(
        make_metric_row(
            dataset_id=base["dataset_id"],
            model=base["model"],
            intervention_id=base["intervention_id"],
            intervention_name=base["intervention_name"],
            metric_category=base["category"],
            metric_name="rnx_curve",
            space=base["space"],
            summary=NAN_SUMMARY,
            n_cells=base["n_cells"],
            seed=base["seed"],
            extra={"rnx_curve_json": _rnx_curve_json(rnx, n_cells=n_cells)},
        )
    )


def _append_distcorr_row(
    rows: list[dict[str, Any]],
    *,
    ref: np.ndarray,
    man: np.ndarray,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    seed: int,
    n_cells: int,
) -> None:
    value = _distcorr_pair(ref, man)
    rows.append(
        make_metric_row(
            dataset_id=dataset_id,
            model=model,
            intervention_id=intervention_id,
            intervention_name=intervention_name,
            metric_category="structure_metrics",
            metric_name="distcorr",
            space=_SPACE,
            summary=scalar_summary(value),
            n_cells=n_cells,
            seed=seed,
        )
    )


def _append_intrinsic_dim_rows(
    rows: list[dict[str, Any]],
    *,
    ref: np.ndarray,
    man: np.ndarray,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    seed: int,
    n_cells: int,
) -> None:
    for side, mat in (("ref", ref), ("man", man)):
        twonn = _estimate_twonn(mat)
        pr = _estimate_participation_ratio(mat)
        for metric_name, value in (
            ("intrinsic_dim_twonn", twonn),
            ("intrinsic_dim_participation_ratio", pr),
        ):
            rows.append(
                make_metric_row(
                    dataset_id=dataset_id,
                    model=model,
                    intervention_id=intervention_id,
                    intervention_name=intervention_name,
                    metric_category="intrinsic_dimension",
                    metric_name=metric_name,
                    space=_SPACE,
                    summary=scalar_summary(value),
                    n_cells=n_cells,
                    seed=seed,
                    extra={"side": side},
                )
            )


def compute_structure_metrics(
    *,
    bundle: Any,
    dataset_id: str,
    model: str,
    intervention_id: str,
    intervention_name: str,
    seed: int,
) -> pd.DataFrame:
    """ViScore SP/RNX, Székely distcorr, and skdim TwoNN / participation ratio on embeddings."""
    rows: list[dict[str, Any]] = []
    n_cells = int(bundle.emb_ref.shape[0])
    ref = np.asarray(bundle.emb_ref, dtype=np.float64)
    man = np.asarray(bundle.emb_man, dtype=np.float64)

    logger.info(
        "structure_metrics: intervention=%s n_cells=%d",
        intervention_id,
        n_cells,
    )

    _append_viscore_rows(
        rows,
        ref=ref,
        man=man,
        dataset_id=dataset_id,
        model=model,
        intervention_id=intervention_id,
        intervention_name=intervention_name,
        seed=seed,
        n_cells=n_cells,
    )
    _append_distcorr_row(
        rows,
        ref=ref,
        man=man,
        dataset_id=dataset_id,
        model=model,
        intervention_id=intervention_id,
        intervention_name=intervention_name,
        seed=seed,
        n_cells=n_cells,
    )
    _append_intrinsic_dim_rows(
        rows,
        ref=ref,
        man=man,
        dataset_id=dataset_id,
        model=model,
        intervention_id=intervention_id,
        intervention_name=intervention_name,
        seed=seed,
        n_cells=n_cells,
    )

    return pd.DataFrame(rows)
