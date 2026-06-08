"""Layer registry."""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

from fabric.core.layer import Layer
from fabric.core.layers.activation import ReLULayer
from fabric.core.layers.linear import LinearLayer
from fabric.utils.errors import ModelError

LAYER_REGISTRY: dict[str, type[Layer]] = {
    "Linear": LinearLayer,
    "ReLU": ReLULayer,
}


def build_layer(name: str, params: Any) -> Any:
    """Instantiate one layer module from a registry key and parameter mapping."""
    cls = LAYER_REGISTRY.get(str(name))
    if cls is None:
        available = ", ".join(sorted(LAYER_REGISTRY))
        raise ModelError(f"Unknown layer '{name}'. Available: {available}")
    if isinstance(params, dict):
        kwargs = params
    elif params is not None:
        kwargs = OmegaConf.to_container(params, resolve=True) or {}
    else:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise ModelError(f"Layer '{name}' params must be a mapping")
    return cls().build(**kwargs)
