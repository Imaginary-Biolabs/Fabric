"""Registry client."""

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
    client = PlatformClient()
    return client.request("GET", f"/assets/{asset_id}")


def fetch_asset_version(asset_id: str, version: str, *, cache: bool = True) -> dict[str, Any]:
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
    client = PlatformClient()
    params: dict[str, Any] = {"visibility": "public"}
    if kind:
        params["kind"] = kind
    if q:
        params["q"] = q
    payload = client.request("GET", "/assets", params=params)
    return payload.get("items") or []


def cached_config_path(asset_id: str, version: str = "1") -> Path:
    path = _cache_path(asset_id, version)
    if not path.is_file():
        fetch_asset_version(asset_id, version, cache=True)
    if not path.is_file():
        raise RegistryError(f"Failed to cache config for {asset_id} v{version}")
    return path
