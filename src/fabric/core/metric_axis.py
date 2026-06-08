"""Schema-aware axis grouping helpers for metrics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from grumpy import GrumpyArray

from fabric.utils.constants import SCHEMA
from fabric.utils.errors import MetricError

ReductionFn = Callable[[GrumpyArray], GrumpyArray | float]
"""Callable that collapses a Grumpy array to a scalar or pass-through array."""

REDUCTIONS: tuple[str, ...] = ("mean", "sum", "min", "max", "median")


def schema_levels() -> list[str]:
    """Return Fabric biological schema level names in nesting order."""
    levels: list[str] = []
    for entry in SCHEMA:
        if isinstance(entry, tuple):
            levels.extend(entry)
        else:
            levels.append(entry)
    return levels


def level_index(level: str) -> int:
    """Map a schema level name to its position in :func:`schema_levels`."""
    levels = schema_levels()
    if level not in levels:
        available = ", ".join(levels)
        raise MetricError(f"Unknown schema level '{level}'; expected one of: {available}")
    return levels.index(level)


def grumpy_ndim(values: GrumpyArray) -> int:
    """Infer list-chain depth from a Grumpy array."""

    def depth(node: Any) -> int:
        if not isinstance(node, list):
            return 0
        if not node:
            return 1
        child_depths = [depth(child) for child in node]
        if all(child == 0 for child in child_depths):
            return 1
        return 1 + max(child_depths)

    return depth(values.to_list())


def resolve_dim(level: str | int, *, ndim: int) -> int:
    """Normalize a schema level or axis index to a non-negative dim."""
    if isinstance(level, int):
        return level if level >= 0 else ndim + level
    raise MetricError(
        f"Cannot resolve schema level '{level}' without nesting context; "
        "pass an integer dim index instead."
    )


def _reduce_dim(values: GrumpyArray, dim: int, op: str) -> GrumpyArray:
    """Apply a named reduction along one ragged axis."""
    if op == "mean":
        return values.mean(dim=dim)
    if op == "sum":
        return values.sum(dim=dim)
    if op == "min":
        return values.min(dim=dim)
    if op == "max":
        return values.max(dim=dim)
    if op == "median":
        return values.median(dim=dim)
    raise MetricError(f"Unknown group reduction '{op}'")


def group_values(
    values: GrumpyArray,
    *,
    on: str | int | None,
    per: str | int | None,
    group_reduction: str,
) -> GrumpyArray:
    """Reduce elementwise metric values from ``on`` granularity to ``per`` groups.

    Args:
        values: Elementwise metric values aligned with ``y_pred`` / ``y_true``.
        on: Innermost schema level or dim index where values are defined.
        per: Outer schema level or dim index to group results under.
        group_reduction: Reduction applied along axes between ``per`` and ``on``.

    Returns:
        Values nested at the ``per`` level, or unchanged when grouping is disabled.
    """
    if on is None:
        return values

    ndim = grumpy_ndim(values)
    if ndim < 1:
        raise MetricError("Metric 'on' requires nested Grumpy arrays; got flat data")

    if isinstance(on, int):
        on_dim = resolve_dim(on, ndim=ndim)
        on_level = None
    else:
        on_level = on
        on_dim = ndim - 1

    if per is None:
        return values

    if isinstance(per, int):
        per_dim = resolve_dim(per, ndim=ndim)
    else:
        if on_level is None:
            raise MetricError(
                "Metric 'per' as a schema level requires 'on' to name a schema level too"
            )
        if level_index(per) >= level_index(on_level):
            raise MetricError(
                f"Metric 'per' ({per}) must be an outer schema level relative to 'on' ({on_level})"
            )
        per_dim = 0

    if per_dim >= on_dim:
        raise MetricError(
            f"Metric 'per' ({per}) must be an outer axis relative to 'on' ({on}); "
            f"got per_dim={per_dim}, on_dim={on_dim}"
        )

    grouped = values
    for dim in range(on_dim, per_dim, -1):
        grouped = _reduce_dim(grouped, dim, group_reduction)
    return grouped


def _collapse_scalar(result: GrumpyArray | float) -> GrumpyArray | float:
    """Normalize Grumpy outputs that collapsed to Python scalars."""
    if isinstance(result, (int, float)):
        return float(result)
    return result


def apply_reduction(values: GrumpyArray, reduction: str) -> GrumpyArray | float:
    """Apply a final reduction across all remaining axes."""
    if reduction == "none":
        return values
    if reduction not in REDUCTIONS:
        available = ", ".join(["none", *REDUCTIONS])
        raise MetricError(f"Unknown reduction '{reduction}'; expected one of: {available}")

    current = values
    while grumpy_ndim(current) > 1:
        current = _reduce_dim(current, 0, reduction)
    if grumpy_ndim(current) == 1:
        return _collapse_scalar(_reduce_dim(current, 0, reduction))
    return _collapse_scalar(current)


def metric_value_to_python(value: GrumpyArray | float) -> float | list[Any]:
    """Convert a computed metric value into JSON-friendly Python data."""
    if isinstance(value, GrumpyArray):
        payload = value.to_list()
        if isinstance(payload, (int, float)):
            return float(payload)
        return payload
    return float(value)
