from __future__ import annotations

from typing import Any

from fabric.core.layer import Layer
from fabric.utils.errors import ModelError


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ModelError(
            "Layers require PyTorch: pip install 'imaginary-fabric[torch]'"
        ) from exc
    return torch


class ReLULayer(Layer):
    """ReLU activation on the ``features`` slot."""

    name = "ReLU"
    reads = ("features",)
    writes = ("features",)

    def build(self, **kwargs: Any) -> Any:
        torch = _require_torch()
        return torch.nn.ReLU()
