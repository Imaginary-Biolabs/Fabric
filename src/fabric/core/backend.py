"""Backend protocol for deep learning frameworks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from grumpy import GrumpyArray

from fabric.core.collater import CollatedBatch


class Backend(ABC):
    """Transfer collated batches to framework tensors and run train/eval steps."""

    name: str = "Backend"

    @abstractmethod
    def setup(self, model: Any) -> Any:
        """Prepare a user-configured model for training."""
        raise NotImplementedError

    @abstractmethod
    def to_tensor(self, array: GrumpyArray) -> Any:
        """Convert a Grumpy array to a framework tensor."""
        raise NotImplementedError

    @abstractmethod
    def to_grumpy(self, tensor: Any) -> GrumpyArray:
        """Convert a framework tensor back to a Grumpy array."""
        raise NotImplementedError

    @abstractmethod
    def train_step(self, model: Any, batch: CollatedBatch) -> float:
        """Execute one training step and return the scalar loss."""
        raise NotImplementedError

    @abstractmethod
    def eval_step(self, model: Any, batch: CollatedBatch) -> tuple[float, GrumpyArray]:
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
