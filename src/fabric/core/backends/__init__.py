"""Backend registry."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from fabric.core.backends.tensorflow import TensorflowBackend
from fabric.core.backends.torch import TorchBackend
from fabric.utils.errors import ConfigError

BACKEND_REGISTRY: dict[str, type[TorchBackend]] = {
    "TorchBackend": TorchBackend,
    "TensorflowBackend": TensorflowBackend,
}


def build_backend(config: Any) -> TorchBackend | TensorflowBackend:
    """Instantiate a backend from a single-key YAML mapping."""
    if config is None:
        raise ConfigError("Trainer config requires a 'backend' mapping")
    if not isinstance(config, (dict, DictConfig)):
        raise ConfigError("Trainer 'backend' must be a mapping")
    items = list(config.items())
    if len(items) != 1:
        raise ConfigError("Trainer 'backend' must contain exactly one key")
    name, params = items[0]
    cls = BACKEND_REGISTRY.get(str(name))
    if cls is None:
        available = ", ".join(sorted(BACKEND_REGISTRY))
        raise ConfigError(f"Unknown backend '{name}'. Available: {available}")
    if isinstance(params, dict):
        kwargs = params
    elif params is not None:
        kwargs = OmegaConf.to_container(params, resolve=True) or {}
    else:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise ConfigError(f"Backend '{name}' params must be a mapping")
    return cls(**kwargs)


__all__ = ["TensorflowBackend", "TorchBackend", "BACKEND_REGISTRY", "build_backend"]
