"""Layer protocol for model building blocks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar


class Layer(ABC):
    """One differentiable map used by scaffolds to build ``nn.Module`` graphs.

    Layers declare which named tensor slots they read and write. Scaffolds wire
    slot names; collaters populate initial slots on
    :class:`~fabric.core.collater.CollatedBatch`.
    """

    name: str = "Layer"
    reads: ClassVar[tuple[str, ...]] = ()
    writes: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def build(self, **kwargs: Any) -> Any:
        """Return a framework module for this layer."""
        raise NotImplementedError
