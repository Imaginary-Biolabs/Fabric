"""External adapter registry and factory helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from fabric.core.external import External
from fabric.core.externals.local import LocalAdapter
from fabric.utils.errors import ExternalError

EXTERNAL_REGISTRY: dict[str, type[External]] = {
    "LocalAdapter": LocalAdapter,
}


def build_external(config: dict[str, Any] | DictConfig, *, config_dir: Path) -> External:
    """Instantiate an external adapter from a dataset ``external`` YAML block.

    Args:
        config: Mapping with exactly one adapter key (for example ``LocalAdapter``).
        config_dir: Directory containing the dataset config file.

    Returns:
        Configured :class:`~fabric.core.external.External` instance.

    Raises:
        ExternalError: If the config shape is invalid or the adapter is unknown.

    Example:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.create({"LocalAdapter": {"paths": ["*.pdb"]}})
        >>> adapter = build_external(cfg, config_dir=Path("."))
    """
    if len(config) != 1:
        raise ExternalError("Dataset 'external' section must contain exactly one adapter")
    name, params = next(iter(config.items()))
    cls = EXTERNAL_REGISTRY.get(str(name))
    if cls is None:
        available = ", ".join(sorted(EXTERNAL_REGISTRY))
        raise ExternalError(f"Unknown external '{name}'. Available: {available}")
    kwargs = OmegaConf.to_container(params, resolve=True) if params is not None else {}
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise ExternalError(f"External '{name}' params must be a mapping")
    return cls(config_dir=config_dir, **kwargs)
