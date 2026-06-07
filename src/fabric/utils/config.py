"""Configuration loading with OmegaConf."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from fabric.utils.errors import ConfigurationError


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> DictConfig:
    """Load a YAML config file and apply optional overrides."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"Config file not found: {config_path.resolve()}")

    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg
