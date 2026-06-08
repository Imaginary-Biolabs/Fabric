"""Scaffold protocol and shared model-training types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from grumpy import GrumpyArray


@dataclass(frozen=True)
class CollaterSpec:
    """Layout contract between collaters and models."""

    layout: str
    slots: tuple[str, ...] = ("features",)


@dataclass
class StepResult:
    """Output of one model train or validation step."""

    loss: float
    predictions: GrumpyArray | None = None
    logs: dict[str, float] = field(default_factory=dict)


class Scaffold(ABC):
    """Wire layers into a compute graph with named tensor slots."""

    name: str = "Scaffold"
    input_slots: ClassVar[tuple[str, ...]] = ("features",)
    output_slots: ClassVar[tuple[str, ...]] = ("predictions",)

    @abstractmethod
    def build(self, *, input_dim: int, output_dim: int = 1) -> Any:
        """Return a framework module consuming and producing slot dictionaries."""
        raise NotImplementedError
