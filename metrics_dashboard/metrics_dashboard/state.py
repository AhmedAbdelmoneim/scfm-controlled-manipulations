"""Query-parameter helpers for shareable dashboard URLs."""

from __future__ import annotations

import streamlit as st


def get_param(key: str, default: str | None = None) -> str | None:
    val = st.query_params.get(key)
    if val is None or val == "":
        return default
    if isinstance(val, list):
        return val[0] if val else default
    return str(val)


def get_param_list(key: str) -> list[str]:
    val = st.query_params.get(key)
    if val is None or val == "":
        return []
    if isinstance(val, list):
        raw = ",".join(val)
    else:
        raw = str(val)
    return [x.strip() for x in raw.split(",") if x.strip()]


def set_params(**kwargs: str | None) -> None:
    for key, value in kwargs.items():
        if value is None:
            if key in st.query_params:
                del st.query_params[key]
        else:
            st.query_params[key] = value


def set_param_list(key: str, values: list[str]) -> None:
    if values:
        st.query_params[key] = ",".join(values)
    elif key in st.query_params:
        del st.query_params[key]


def prefixed_param(prefix: str, key: str, default: str | None = None) -> str | None:
    return get_param(f"{prefix}{key}", default)


def prefixed_param_list(prefix: str, key: str) -> list[str]:
    return get_param_list(f"{prefix}{key}")
