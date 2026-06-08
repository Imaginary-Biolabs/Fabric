"""Task protocol for benchmark (X, y) extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from grumpy import GrumpyArray

from fabric.core.data import Data
from fabric.core.dataset import Dataset

TaskResult = tuple[tuple[Data, ...], GrumpyArray]
"""Return type of :meth:`Task.extract`."""


class Task(ABC):
    """Convert sampled scene indices into model inputs and targets.

    Example:
        >>> from fabric.core.tasks import PropertyPredictionTask
        >>> task = PropertyPredictionTask(target="stability")
        >>> X, y = task.extract(dataset, [0, 1])
    """

    name: str = "Task"

    @abstractmethod
    def extract(self, dataset: Dataset, indices: list[int]) -> TaskResult:
        """Build ``(X, y)`` for one batch of scene indices.

        Args:
            dataset: Released dataset with structural data and assets.
            indices: Scene-level dataframe indices from the sampler.

        Returns:
            ``(X, y)`` where ``X`` is a tuple of :class:`~fabric.core.data.Data`
            payloads and ``y`` is a Grumpy target array.

        Raises:
            TaskError: If required targets or columns are missing.
        """
        raise NotImplementedError

    def hash_params(self) -> dict[str, Any]:
        """Return constructor parameters for provenance hashing."""
        return {}
