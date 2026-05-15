"""Intervention sweep expansion and reference ID resolution (shared by pipeline + evaluation)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

from scfm_controlled_manipulations.io import intervention_id


def is_sweep_value(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def expand_kwargs(kwargs: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not kwargs:
        return [{}]

    keys = list(kwargs)
    values = [
        list(value) if is_sweep_value(value) else [value]
        for value in (kwargs[key] for key in keys)
    ]
    return [dict(zip(keys, combination, strict=True)) for combination in product(*values)]


def expand_intervention_specs(specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Expand list-valued intervention kwargs into a Cartesian sweep."""
    expanded = []
    for spec in specs:
        name = spec["name"]
        kwargs = dict(spec.get("kwargs") or {})
        kwargs.update(dict(spec.get("sweep") or {}))

        for expanded_kwargs in expand_kwargs(kwargs):
            expanded.append({"name": name, "kwargs": expanded_kwargs})

    return expanded


def reference_intervention_id(config: dict[str, Any]) -> str:
    """Resolve reference intervention id from config (string or legacy dict)."""
    if rid := config.get("reference_intervention_id"):
        return str(rid)
    ref = config.get("reference_intervention")
    if not ref:
        raise ValueError(
            "Config requires `reference_intervention_id` string or legacy "
            "`reference_intervention` {{name, kwargs}}."
        )
    return intervention_id(ref["name"], dict(ref.get("kwargs") or {}))
