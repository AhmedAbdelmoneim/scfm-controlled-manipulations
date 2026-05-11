from __future__ import annotations

from .downsample import Downsample
from .gene_dropout import GeneDropout
from .gene_shuffle import GeneShuffle
from .local_smoothing import LocalSmoothing
from .poisson_resampling import PoissonResampling

REGISTRY = {
    cls.name: cls
    for cls in [
        Downsample,
        GeneDropout,
        GeneShuffle,
        LocalSmoothing,
        PoissonResampling,
    ]
}
