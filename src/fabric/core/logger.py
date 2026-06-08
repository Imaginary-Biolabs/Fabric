"""Logger protocol for trainer runs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Logger(ABC):
    """Record training losses and evaluation metrics."""

    name: str = "Logger"

    @abstractmethod
    def log_loss(self, loss: float, *, mode: str = "train") -> None:
        """Record one scalar loss value."""
        raise NotImplementedError

    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], *, mode: str = "eval") -> None:
        """Record a metric mapping."""
        raise NotImplementedError
