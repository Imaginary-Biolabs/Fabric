"""Feature extraction helpers for collaters."""

from __future__ import annotations

from collections.abc import Callable

import grumpy as gr
from grumpy import GrumpyArray, GrumpyDataFrame

from fabric.core.data import Data
from fabric.utils.constants import SCHEMA
from fabric.utils.errors import CollateError


def schema_levels() -> list[str]:
    """Return Fabric schema level names in nesting order."""
    levels: list[str] = []
    for entry in SCHEMA:
        if isinstance(entry, tuple):
            levels.extend(entry)
        else:
            levels.append(entry)
    return levels


def column_levels() -> dict[str, str]:
    """Map dataframe column prefixes to schema levels."""
    mapping: dict[str, str] = {}
    for level in schema_levels():
        mapping[level] = level
        mapping[f"{level}_id"] = level
        mapping[f"{level}_label"] = level
        mapping[f"{level}_number"] = level
        mapping[f"{level}_pos"] = level
    mapping["atom_pos"] = "atom"
    return mapping


_WIDE_INCOMPATIBLE_LEVELS = frozenset({"atom", "residue", "group", "chain"})

_BUILTIN_FEATURES: dict[str, Callable[[GrumpyDataFrame], list[float]]] = {
    "residue_count": lambda df: _count_at_level(df, "residue"),
    "atom_count": lambda df: _count_at_level(df, "atom"),
    "chain_count": lambda df: _count_at_level(df, "chain"),
    "molecule_count": lambda df: _count_at_level(df, "molecule"),
}


def _count_at_level(df: GrumpyDataFrame, level: str) -> list[float]:
    """Count entities at a schema level for each scene row."""
    try:
        counts = df.shape(dim=level).flatten()
    except Exception as exc:
        raise CollateError(
            f"Cannot compute '{level}_count' from batch; "
            f"shape(dim={level!r}) failed: {exc}"
        ) from exc
    return [float(value) for value in counts.to_list()]


def available_feature_keys(df: GrumpyDataFrame) -> set[str]:
    """Return builtin and column feature names available for a batch."""
    keys = set(_BUILTIN_FEATURES)
    keys.update(df.to_dict().keys())
    return keys


def extract_feature_column(df: GrumpyDataFrame, name: str) -> list[float]:
    """Extract one scalar feature per scene from a dataframe column."""
    if name in _BUILTIN_FEATURES:
        return _BUILTIN_FEATURES[name](df)
    if name not in df.to_dict():
        available = ", ".join(sorted(available_feature_keys(df)))
        raise CollateError(f"Unknown collater feature '{name}'; available: {available}")
    level = column_levels().get(name)
    if level in _WIDE_INCOMPATIBLE_LEVELS:
        raise CollateError(
            f"WideCollater cannot pad nested feature '{name}'; "
            "use LongCollater for variable-length batches"
        )
    column = getattr(df, name)
    values = column.to_list()
    if not values:
        return []
    if isinstance(values[0], list):
        raise CollateError(
            f"WideCollater cannot pad nested feature '{name}'; "
            "use LongCollater for variable-length batches"
        )
    return [float(value) for value in values]


def extract_feature_matrix(data: Data, features: list[str]) -> GrumpyArray:
    """Build a rectangular ``(batch, n_features)`` matrix."""
    df = data.data
    if not features:
        raise CollateError("Collater requires at least one feature")
    rows = [extract_feature_column(df, name) for name in features]
    if any(len(row) != len(rows[0]) for row in rows):
        raise CollateError("Feature columns must share the same batch length")
    batch_size = len(rows[0])
    matrix = gr.array(
        [[rows[col][row] for col in range(len(rows))] for row in range(batch_size)],
        dtype=gr.float32,
    )
    if not gr.is_rectangular(matrix):
        raise CollateError("Collater expected a 2D feature matrix")
    return matrix
