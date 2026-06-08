"""Objective protocol for model training paradigms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from fabric.core.collater import CollatedBatch


class Objective(ABC):
    """Compute named losses from scaffold outputs and batch targets."""

    name: str = "Objective"

    @abstractmethod
    def loss(
        self,
        outputs: dict[str, Any],
        batch: CollatedBatch,
        *,
        tensors: dict[str, Any],
        backend: Any,
    ) -> dict[str, Any]:
        """Return a mapping that includes a ``'loss'`` tensor."""
        raise NotImplementedError
