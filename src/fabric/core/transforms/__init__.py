"""Transform registry and YAML factory helpers."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from fabric.core.transform import Compose, Transform
from fabric.core.transforms.filter_sequence_length import FilterSequenceLength
from fabric.core.transforms.identity import Identity
from fabric.core.transforms.random_split import RandomSplit
from fabric.utils.errors import TransformError

TRANSFORM_REGISTRY: dict[str, type[Transform]] = {
    "Identity": Identity,
    "RandomSplit": RandomSplit,
    "FilterSequenceLength": FilterSequenceLength,
}


def build_transforms(config: list[Any] | ListConfig | None) -> Compose:
    """Instantiate a transform chain from dataset YAML.

    Args:
        config: List of single-key mappings such as
            ``[{"RandomSplit": {"n_train": 8, "n_val": 2, "n_test": 2}}]``.
            When ``None`` or empty, returns :class:`Identity` only.

    Returns:
        :class:`~fabric.core.transform.Compose` transform chain.

    Raises:
        TransformError: If an entry is malformed or references an unknown transform.

    Example:
        >>> from omegaconf import OmegaConf
        >>> cfg = OmegaConf.create([{"Identity": {}}])
        >>> chain = build_transforms(cfg)
    """
    if not config:
        return Compose([Identity()])
    transforms: list[Transform] = []
    for entry in config:
        if isinstance(entry, (dict, DictConfig)):
            items = list(entry.items())
        else:
            raise TransformError(f"Invalid transform entry: {entry!r}")
        if len(items) != 1:
            raise TransformError(f"Transform entry must have exactly one key: {entry!r}")
        name, params = items[0]
        cls = TRANSFORM_REGISTRY.get(str(name))
        if cls is None:
            available = ", ".join(sorted(TRANSFORM_REGISTRY))
            raise TransformError(f"Unknown transform '{name}'. Available: {available}")
        kwargs = OmegaConf.to_container(params, resolve=True) if params is not None else {}
        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, dict):
            raise TransformError(f"Transform '{name}' params must be a mapping")
        transforms.append(cls(**kwargs))
    return Compose(transforms)
