"""Task registry and YAML factory helpers."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from fabric.core.task import Task
from fabric.core.tasks.property import PropertyPredictionTask
from fabric.utils.errors import TaskError

TASK_REGISTRY: dict[str, type[Task]] = {
    "PropertyPredictionTask": PropertyPredictionTask,
}


def build_task(config: dict[str, Any] | DictConfig | ListConfig | None) -> Task:
    """Instantiate a task from benchmark YAML.

    Args:
        config: Single-key mapping such as
            ``{"PropertyPredictionTask": {"target": "stability"}}``.

    Returns:
        Configured :class:`~fabric.core.task.Task` instance.

    Raises:
        TaskError: If the config shape is invalid or the task is unknown.
    """
    if config is None:
        raise TaskError("Benchmark config requires a 'task' section")
    if isinstance(config, (dict, DictConfig)):
        items = list(config.items())
    else:
        raise TaskError(f"Invalid task config: {config!r}")
    if len(items) != 1:
        raise TaskError(f"Task config must contain exactly one entry: {config!r}")
    name, params = items[0]
    cls = TASK_REGISTRY.get(str(name))
    if cls is None:
        available = ", ".join(sorted(TASK_REGISTRY))
        raise TaskError(f"Unknown task '{name}'. Available: {available}")
    kwargs = OmegaConf.to_container(params, resolve=True) if params is not None else {}
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise TaskError(f"Task '{name}' params must be a mapping")
    return cls(**kwargs)
