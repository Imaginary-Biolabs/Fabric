"""Collater registry."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from fabric.core.collaters.long import LongCollater
from fabric.core.collaters.wide import WideCollater
from fabric.utils.errors import ConfigError

COLLATER_REGISTRY: dict[str, type[WideCollater]] = {
    "WideCollater": WideCollater,
    "LongCollater": LongCollater,
}


def build_collater(config: Any) -> WideCollater | LongCollater:
    """Instantiate a collater from a single-key YAML mapping."""
    if config is None:
        raise ConfigError("Trainer config requires a 'collater' mapping")
    if not isinstance(config, (dict, DictConfig)):
        raise ConfigError("Trainer 'collater' must be a mapping")
    items = list(config.items())
    if len(items) != 1:
        raise ConfigError("Trainer 'collater' must contain exactly one key")
    name, params = items[0]
    cls = COLLATER_REGISTRY.get(str(name))
    if cls is None:
        available = ", ".join(sorted(COLLATER_REGISTRY))
        raise ConfigError(f"Unknown collater '{name}'. Available: {available}")
    if isinstance(params, dict):
        kwargs = params
    elif params is not None:
        kwargs = OmegaConf.to_container(params, resolve=True) or {}
    else:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise ConfigError(f"Collater '{name}' params must be a mapping")
    return cls(**kwargs)


__all__ = ["LongCollater", "WideCollater", "build_collater", "COLLATER_REGISTRY"]
