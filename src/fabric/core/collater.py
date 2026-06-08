"""Collater protocol for benchmark batches."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import grumpy as gr
from grumpy import GrumpyArray

from fabric.core.data import Data


@dataclass
class CollatedBatch:
    """Framework-agnostic rectangular batch for deep learning backends.

    Attributes:
        features: Feature matrix with shape ``(batch, n_features)`` or a nested
            long-layout payload stored under ``meta``.
        y: Target vector with shape ``(batch,)``.
        scene_index: Optional scene indices for long-layout batches.
        meta: Additional collater-specific metadata.
    """

    features: GrumpyArray
    y: GrumpyArray
    scene_index: GrumpyArray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def uncollate_y(self, y_pred: GrumpyArray) -> GrumpyArray:
        """Map per-row predictions back to a Grumpy target array."""
        return y_pred.flatten().astype(gr.float64)


class Collater(ABC):
    """Convert benchmark ``(X, y)`` batches into backend-ready arrays."""

    name: str = "Collater"

    @abstractmethod
    def collate(self, X: tuple[Data, ...], y: GrumpyArray | None) -> CollatedBatch:
        """Collate one loader batch.

        Args:
            X: Task inputs (typically a single :class:`~fabric.core.data.Data` batch).
            y: Ground-truth targets from the benchmark task.

        Returns:
            Collated rectangular or long-layout batch.
        """
        raise NotImplementedError

    def __call__(
        self,
        X: tuple[Data, ...],
        y: GrumpyArray | None,
    ) -> CollatedBatch:
        """Alias for :meth:`collate`."""
        return self.collate(X, y)
