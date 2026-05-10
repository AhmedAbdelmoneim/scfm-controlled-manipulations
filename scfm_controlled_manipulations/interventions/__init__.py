from __future__ import annotations

from .gene_shuffle import GeneShuffle

REGISTRY = {cls.name: cls for cls in [GeneShuffle]}
