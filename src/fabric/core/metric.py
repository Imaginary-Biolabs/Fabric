"""Metric protocol for benchmark evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import grumpy as gr
from grumpy import GrumpyArray

from fabric.core.metric_axis import (
    apply_reduction,
    group_values,
    grumpy_ndim,
    metric_value_to_python,
)
from fabric.utils.errors import MetricError


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
        if self.on is None and self.per is None and grumpy_ndim(values) <= 1:
            return values
        return group_values(
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
        result = apply_reduction(combined, self.reduction)
        self._computed = True
        return result

    def compute_value(self) -> float | list[Any]:
        """Return :meth:`compute` output as plain Python scalars or nested lists."""
        return metric_value_to_python(self.compute())

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
