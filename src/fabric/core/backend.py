"""Backend protocol for deep learning frameworks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from fabric.core.collater import CollatedBatch


class Backend(ABC):
    """Transfer collated batches to framework tensors and run train/eval steps."""

    name: str = "Backend"

    @abstractmethod
    def setup(self, model: Any) -> Any:
        """Prepare a user-configured model for training."""
        raise NotImplementedError

    @abstractmethod
    def to_tensor(self, array: np.ndarray) -> Any:
        """Convert a NumPy array to a framework tensor."""
        raise NotImplementedError

    @abstractmethod
    def to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert a framework tensor back to NumPy."""
        raise NotImplementedError

    @abstractmethod
    def train_step(self, model: Any, batch: CollatedBatch) -> float:
        """Execute one training step and return the scalar loss."""
        raise NotImplementedError

    @abstractmethod
    def eval_step(self, model: Any, batch: CollatedBatch) -> tuple[float, np.ndarray]:
        """Execute one evaluation step and return loss and predictions."""
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(
        self,
        path: str,
        *,
        model: Any,
        epoch: int,
        step: int,
    ) -> None:
        """Persist model and optimizer state from ``model.optimizer``."""
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, path: str, *, model: Any) -> dict[str, int]:
        """Restore model and optimizer state onto ``model.optimizer``."""
        raise NotImplementedError
