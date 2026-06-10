"""Registry client for Imaginary platform assets.

Fetch asset metadata and versioned YAML configs from the API, with optional
local caching under :meth:`~fabric.utils.settings.Settings.registry_cache_path`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from fabric import Settings
from fabric.platform.client import PlatformClient
from fabric.utils.errors import RegistryError


def _cache_path(asset_id: str, version: str) -> Path:
    return Settings.registry_cache_path() / asset_id / version / "config.yaml"


def fetch_asset(asset_id: str) -> dict[str, Any]:
    """Fetch asset metadata from the platform registry.

    Args:
        asset_id: Registry asset identifier.

    Returns:
        Asset payload from the API (id, kind, visibility, …).

    Example:
        >>> # fetch_asset("D_mini")  # doctest: +SKIP
        ... # {'id': 'D_mini', 'kind': 'dataset', ...}
    """
    client = PlatformClient()
    return client.request("GET", f"/assets/{asset_id}")


def fetch_asset_version(asset_id: str, version: str, *, cache: bool = True) -> dict[str, Any]:
    """Fetch one asset version and optionally cache its config locally.

    Args:
        asset_id: Registry asset identifier.
        version: Version label (for example ``"1"``).
        cache: When ``True``, write ``config.yaml`` and ``meta.yaml`` under the
            registry cache directory.

    Returns:
        Version payload including ``config_yaml`` and optional ``meta``.

    Example:
        >>> # fetch_asset_version("D_mini", "1", cache=False)  # doctest: +SKIP
    """
    client = PlatformClient()
    payload = client.request("GET", f"/assets/{asset_id}/versions/{version}")
    if cache:
        path = _cache_path(asset_id, version)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload["config_yaml"])
        meta_path = path.parent / "meta.yaml"
        meta_path.write_text(yaml.safe_dump({"meta": payload.get("meta") or {}}, sort_keys=False))
    return payload


def list_assets(*, kind: str | None = None, q: str | None = None) -> list[dict[str, Any]]:
    """List public assets, optionally filtered by kind or search query.

    Args:
        kind: Restrict to one asset kind (``dataset``, ``model``, …).
        q: Free-text search string.

    Returns:
        List of asset summary dicts from the API.

    Example:
        >>> # list_assets(kind="dataset")  # doctest: +SKIP
        ... # [{'id': 'D_mini', 'kind': 'dataset', ...}]
    """
    client = PlatformClient()
    params: dict[str, Any] = {"visibility": "public"}
    if kind:
        params["kind"] = kind
    if q:
        params["q"] = q
    payload = client.request("GET", "/assets", params=params)
    return payload.get("items") or []


def cached_config_path(asset_id: str, version: str = "1") -> Path:
    """Return a local path to a cached asset config YAML.

    Downloads and caches the version on first access.

    Args:
        asset_id: Registry asset identifier.
        version: Version label. Defaults to ``"1"``.

    Returns:
        Path to ``config.yaml`` under the registry cache.

    Raises:
        RegistryError: If caching fails.

    Example:
        >>> # cached_config_path("D_mini")  # doctest: +SKIP
        ... # PosixPath('~/.imaginary/registry/D_mini/1/config.yaml')
    """
    path = _cache_path(asset_id, version)
    if not path.is_file():
        fetch_asset_version(asset_id, version, cache=True)
    if not path.is_file():
        raise RegistryError(f"Failed to cache config for {asset_id} v{version}")
    return path
