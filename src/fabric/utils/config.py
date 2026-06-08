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
    """Load a YAML config file and apply optional overrides.

    Args:
        path: Filesystem path to a YAML configuration file.
        overrides: Optional mapping merged on top of the loaded config.

    Returns:
        Resolved OmegaConf :class:`~omegaconf.DictConfig` object.

    Raises:
        ConfigurationError: If the config file does not exist.

    Example:
        >>> from fabric.utils.config import load_config
        >>> cfg = load_config("tests/fixtures/datasets/D_local.yaml")
        >>> "id" in cfg
        True
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"Config file not found: {config_path.resolve()}")

    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg
