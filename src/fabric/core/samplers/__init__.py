"""Sampler registry and YAML factory helpers."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from fabric.core.sampler import Sampler
from fabric.core.samplers.molecule import RandomMoleculeSampler
from fabric.utils.errors import SamplerError

SAMPLER_REGISTRY: dict[str, type[Sampler]] = {
    "RandomMoleculeSampler": RandomMoleculeSampler,
}


def build_sampler(config: dict[str, Any] | DictConfig | ListConfig | None) -> Sampler:
    """Instantiate a sampler from benchmark YAML.

    Args:
        config: Single-key mapping such as ``{"RandomMoleculeSampler": {}}``.

    Returns:
        Configured :class:`~fabric.core.sampler.Sampler` instance.

    Raises:
        SamplerError: If the config shape is invalid or the sampler is unknown.
    """
    if config is None:
        raise SamplerError("Benchmark config requires a 'sampler' section")
    if isinstance(config, (dict, DictConfig)):
        items = list(config.items())
    else:
        raise SamplerError(f"Invalid sampler config: {config!r}")
    if len(items) != 1:
        raise SamplerError(f"Sampler config must contain exactly one entry: {config!r}")
    name, params = items[0]
    cls = SAMPLER_REGISTRY.get(str(name))
    if cls is None:
        available = ", ".join(sorted(SAMPLER_REGISTRY))
        raise SamplerError(f"Unknown sampler '{name}'. Available: {available}")
    kwargs = OmegaConf.to_container(params, resolve=True) if params is not None else {}
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise SamplerError(f"Sampler '{name}' params must be a mapping")
    return cls(**kwargs)
