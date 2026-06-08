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


class LinearLayer(Layer):
    """Dense affine map on the ``features`` slot."""

    name = "Linear"
    reads = ("features",)
    writes = ("features",)

    def build(self, **kwargs: Any) -> Any:
        torch = _require_torch()
        in_features = kwargs.get("in_features")
        out_features = kwargs.get("out_features")
        if in_features is None or out_features is None:
            raise ModelError("Linear layer requires in_features and out_features")
        return torch.nn.Linear(int(in_features), int(out_features))
