"""Canonical hashing for configs and transform chains."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _to_canonical_mapping(obj: Any) -> Any:
    """Recursively convert configs to sorted, JSON-serializable structures.

    Args:
        obj: Arbitrary config object, possibly containing nested DictConfig values.

    Returns:
        Plain Python dicts/lists with sorted dict keys for stable hashing.
    """
    if isinstance(obj, DictConfig):
        obj = OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, dict):
        return {key: _to_canonical_mapping(obj[key]) for key in sorted(obj)}
    if isinstance(obj, list):
        return [_to_canonical_mapping(item) for item in obj]
    return obj


def config_hash(config: Any) -> str:
    """Return a stable ``sha256:`` digest for a config object.

    Args:
        config: Mapping, list, OmegaConf object, or other JSON-serializable
            structure after canonicalization.

    Returns:
        Hash string prefixed with ``"sha256:"``.

    Example:
        >>> config_hash({"b": 2, "a": 1})
        'sha256:...'
    """
    canonical = _to_canonical_mapping(config)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
