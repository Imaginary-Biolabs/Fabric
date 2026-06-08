"""Metric protocol for benchmark evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import grumpy as gr
from grumpy import GrumpyArray

from fabric.utils.constants import SCHEMA
from fabric.utils.errors import MetricError

_REDUCTIONS = frozenset({"mean", "sum", "min", "max", "median"})


def _schema_levels() -> list[str]:
    levels: list[str] = []
    for entry in SCHEMA:
        if isinstance(entry, tuple):
            levels.extend(entry)
        else:
            levels.append(entry)
    return levels


def _level_index(level: str) -> int:
    levels = _schema_levels()
    if level not in levels:
        available = ", ".join(levels)
        raise MetricError(f"Unknown schema level '{level}'; expected one of: {available}")
    return levels.index(level)


def _grumpy_ndim(values: GrumpyArray) -> int:
    """Infer list-chain depth from nested list structure."""

    def depth(node: Any) -> int:
        if not isinstance(node, list):
            return 0
        if not node:
            return 1
        child_depths = [depth(child) for child in node]
        return 1 if all(d == 0 for d in child_depths) else 1 + max(child_depths)

    return depth(values.to_list())


def _resolve_dim(level: str | int, *, ndim: int) -> int:
    if isinstance(level, int):
        return level if level >= 0 else ndim + level
    raise MetricError(
        f"Cannot resolve schema level '{level}' without nesting context; "
        "pass an integer dim index instead."
    )


def _reduce_dim(values: GrumpyArray, dim: int, op: str) -> GrumpyArray | float:
    if op not in _REDUCTIONS:
        raise MetricError(f"Unknown group reduction '{op}'")
    return getattr(values, op)(dim=dim)


def _group_values(
    values: GrumpyArray,
    *,
    on: str | int | None,
    per: str | int | None,
    group_reduction: str,
) -> GrumpyArray:
    if on is None or per is None:
        return values

    ndim = _grumpy_ndim(values)
    if ndim < 1:
        raise MetricError("Metric 'on' requires nested Grumpy arrays; got flat data")

    if isinstance(on, str):
        on_dim = ndim - 1
        on_level = on
    else:
        on_dim = _resolve_dim(on, ndim=ndim)
        on_level = None

    if isinstance(per, str):
        if on_level is None:
            raise MetricError(
                "Metric 'per' as a schema level requires 'on' to name a schema level too"
            )
        if _level_index(per) >= _level_index(on_level):
            raise MetricError(
                f"Metric 'per' ({per}) must be an outer schema level relative to 'on' ({on_level})"
            )
        per_dim = 0
    else:
        per_dim = _resolve_dim(per, ndim=ndim)

    if per_dim >= on_dim:
        raise MetricError(
            f"Metric 'per' ({per}) must be an outer axis relative to 'on' ({on}); "
            f"got per_dim={per_dim}, on_dim={on_dim}"
        )

    grouped = values
    for dim in range(on_dim, per_dim, -1):
        grouped = _reduce_dim(grouped, dim, group_reduction)
    return grouped


def _apply_reduction(values: GrumpyArray, reduction: str) -> GrumpyArray | float:
    if reduction == "none":
        return values
    if reduction not in _REDUCTIONS:
        available = ", ".join(["none", *_REDUCTIONS])
        raise MetricError(f"Unknown reduction '{reduction}'; expected one of: {available}")

    current = values
    while _grumpy_ndim(current) > 1:
        current = _reduce_dim(current, 0, reduction)
    if _grumpy_ndim(current) == 1:
        current = _reduce_dim(current, 0, reduction)
    return float(current) if isinstance(current, (int, float)) else current


def _metric_value_to_python(value: GrumpyArray | float) -> float | list[Any]:
    if isinstance(value, GrumpyArray):
        payload = value.to_list()
        return float(payload) if isinstance(payload, (int, float)) else payload
    return float(value)


class Metric(ABC):
    """Accumulate predictions and compute evaluation scores on Grumpy arrays.

    Metrics can be computed on a nested biological axis (``on``), optionally
    grouped to an outer level (``per``), then reduced to a scalar or returned
    unchanged when ``reduction='none'``.

    Args:
        on: Schema level or dim index where elementwise values are defined.
        per: Schema level or dim index to group values under before the final
            reduction.
        reduction: Final reduction across grouped values (``'none'`` returns a
            Grumpy array).
        group_reduction: Reduction applied along axes between ``per`` and ``on``.

    Example:
        >>> from fabric.core.metrics import MAE
        >>> metric = MAE(on="residue", per="molecule", reduction="none")
        >>> metric.update(y_pred, y_true)
        >>> metric.compute()
    """

    name: str = "Metric"

    def __init__(
        self,
        *,
        on: str | int | None = None,
        per: str | int | None = None,
        reduction: str = "mean",
        group_reduction: str = "mean",
    ) -> None:
        self.on = on
        self.per = per
        self.reduction = str(reduction)
        self.group_reduction = str(group_reduction)
        self._values: list[GrumpyArray] = []
        self._computed = False

    @abstractmethod
    def _elementwise(self, y_pred: GrumpyArray, y_true: GrumpyArray) -> GrumpyArray:
        """Compute per-element metric values before grouping/reduction."""
        raise NotImplementedError

    def _evaluate_batch(self, y_pred: GrumpyArray, y_true: GrumpyArray) -> GrumpyArray:
        """Return grouped metric values for one batch."""
        values = self._elementwise(y_pred, y_true)
        if self.on is None and self.per is None and _grumpy_ndim(values) <= 1:
            return values
        return _group_values(
            values,
            on=self.on,
            per=self.per,
            group_reduction=self.group_reduction,
        )

    def update(self, y_pred: GrumpyArray, y_true: GrumpyArray) -> None:
        """Accumulate one batch of predictions and targets.

        Args:
            y_pred: Model predictions.
            y_true: Ground-truth targets.
        """
        self._values.append(self._evaluate_batch(y_pred, y_true))
        self._computed = False

    def compute(self) -> float | GrumpyArray:
        """Return the metric value aggregated over all :meth:`update` calls.

        Raises:
            MetricError: If :meth:`update` has not been called yet.
        """
        if not self._values:
            raise MetricError(f"{self.name} has no accumulated batches; call update() first")
        combined = self._values[0] if len(self._values) == 1 else gr.cat(self._values, dim=0)
        result = _apply_reduction(combined, self.reduction)
        self._computed = True
        return result

    def compute_value(self) -> float | list[Any]:
        """Return :meth:`compute` output as plain Python scalars or nested lists."""
        return _metric_value_to_python(self.compute())

    def reset(self) -> None:
        """Clear accumulated batches."""
        self._values.clear()
        self._computed = False

    def hash_params(self) -> dict[str, Any]:
        """Return constructor parameters for provenance hashing."""
        return {
            "on": self.on,
            "per": self.per,
            "reduction": self.reduction,
            "group_reduction": self.group_reduction,
        }


class MultiMetric(Metric):
    """Evaluate multiple metrics and return a combined score mapping.

    Args:
        metrics: Metric instances to update and compute together.
    """

    name = "MultiMetric"

    def __init__(self, metrics: list[Metric]) -> None:
        super().__init__()
        if not metrics:
            raise MetricError("MultiMetric requires at least one metric")
        self.metrics = list(metrics)

    def _elementwise(self, y_pred: GrumpyArray, y_true: GrumpyArray) -> GrumpyArray:
        raise MetricError("MultiMetric does not support single-batch _elementwise")

    def update(self, y_pred: GrumpyArray, y_true: GrumpyArray) -> None:
        for metric in self.metrics:
            metric.update(y_pred, y_true)

    def compute(self) -> float | GrumpyArray:
        raise MetricError("MultiMetric.compute() is unsupported; use compute_all()")

    def compute_all(self) -> dict[str, float | list[Any]]:
        """Compute every child metric.

        Returns:
            Mapping of metric name to scalar or nested-list values.
        """
        return {metric.name: metric.compute_value() for metric in self.metrics}

    def reset(self) -> None:
        super().reset()
        for metric in self.metrics:
            metric.reset()
