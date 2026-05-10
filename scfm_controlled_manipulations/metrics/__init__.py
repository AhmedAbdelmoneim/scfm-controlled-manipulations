from __future__ import annotations

from .distortion import Distortion

REGISTRY = {cls.name: cls for cls in [Distortion]}
