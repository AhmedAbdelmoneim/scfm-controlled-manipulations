from __future__ import annotations

from typing import Any

import numpy as np

from scfm_controlled_manipulations.base import Metric


class Distortion(Metric):
    name = "distortion.mean_l2"

    def compute(self, emb_ref: Any, emb_pert: Any) -> dict:
        a = np.asarray(emb_ref, dtype=float)
        b = np.asarray(emb_pert, dtype=float)
        if hasattr(a, "toarray"):
            a = a.toarray()
        if hasattr(b, "toarray"):
            b = b.toarray()
        d = np.linalg.norm(a - b, axis=-1)
        return {"mean_l2": float(np.mean(d)), "std_l2": float(np.std(d))}
