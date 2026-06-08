"""Metric registry and YAML factory helpers."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from fabric.core.metric import Metric, MultiMetric
from fabric.core.metrics.mae import MAE
from fabric.utils.errors import MetricError

METRIC_REGISTRY: dict[str, type[Metric]] = {
    "MAE": MAE,
}


def build_metrics(config: list[Any] | ListConfig | None) -> MultiMetric:
    """Instantiate metrics from benchmark YAML.

    Args:
        config: List of single-key mappings such as ``[{"MAE": {}}]``.

    Returns:
        :class:`~fabric.core.metric.MultiMetric` wrapping all configured metrics.

    Raises:
        MetricError: If the config is malformed or references an unknown metric.
    """
    if not config:
        raise MetricError("Benchmark config requires a non-empty 'metrics' list")
    metrics: list[Metric] = []
    for entry in config:
        if isinstance(entry, (dict, DictConfig)):
            items = list(entry.items())
        else:
            raise MetricError(f"Invalid metric entry: {entry!r}")
        if len(items) != 1:
            raise MetricError(f"Metric entry must have exactly one key: {entry!r}")
        name, params = items[0]
        cls = METRIC_REGISTRY.get(str(name))
        if cls is None:
            available = ", ".join(sorted(METRIC_REGISTRY))
            raise MetricError(f"Unknown metric '{name}'. Available: {available}")
        if isinstance(params, dict):
            kwargs = params
        elif params is not None:
            kwargs = OmegaConf.to_container(params, resolve=True) or {}
        else:
            kwargs = {}
        if not isinstance(kwargs, dict):
            raise MetricError(f"Metric '{name}' params must be a mapping")
        metrics.append(cls(**kwargs))
    return MultiMetric(metrics)
