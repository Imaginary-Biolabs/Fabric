"""Property prediction tasks for supervised learning.

Extract structural inputs and scalar targets from dataset assets.
"""

from typing import Any

import grumpy as gr

from fabric.core.data import Data
from fabric.core.dataset import Dataset
from fabric.core.task import Task, TaskResult
from fabric.utils.errors import TaskError


class PropertyPredictionTask(Task):
    """Predict a scalar property stored in dataset assets.

    Args:
        target: Asset key holding per-scene target values (list or index mapping).

    Example:
        >>> task = PropertyPredictionTask(target="stability")
        >>> X, y = task.extract(dataset, [0, 1])
    """

    name = "PropertyPredictionTask"

    def __init__(self, target: str) -> None:
        self.target = str(target)

    def hash_params(self) -> dict[str, Any]:
        return {"target": self.target}

    def extract(self, dataset: Dataset, indices: list[int]) -> TaskResult:
        """Return structural inputs and scalar targets for scene indices."""
        if self.target not in dataset.assets.data:
            available = ", ".join(sorted(dataset.assets.data))
            raise TaskError(
                f"Task target '{self.target}' not found in dataset assets; "
                f"available asset keys: {available or '(none)'}"
            )
        values = dataset.assets.data[self.target]
        try:
            targets = [float(values[index]) for index in indices]
        except (TypeError, KeyError, IndexError) as exc:
            raise TaskError(
                f"Task target '{self.target}' must provide one value per scene index; "
                f"could not resolve indices {indices}"
            ) from exc

        frame = dataset.data.data
        batch = Data(frame[indices])
        return (batch,), gr.array(targets, dtype=gr.float64)
