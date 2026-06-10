"""Phase 8 toll station — platform HTTP client."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from fabric.platform.client import PlatformClient, save_credentials
from fabric.platform.registry import fetch_asset_version
from fabric.utils.errors import AuthError


def test_api_key_required(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("IMAGINARY_API_KEY", raising=False)
    monkeypatch.delenv("IMAGINARY_API_BASE", raising=False)
    monkeypatch.setattr("fabric.platform.client.Path.home", lambda: tmp_path)
    with pytest.raises(AuthError):
        PlatformClient()


def test_fetch_asset_version_caches(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IMAGINARY_API_KEY", "img_dev_test")
    monkeypatch.setenv("IMAGINARY_API_BASE", "http://example.test/v1")
    payload = {
        "config_yaml": "id: B_000010\nsplit_scheme: random_8_2_2\n",
        "meta": {"metric_spec": {"primary_metric": "mae"}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Imaginary-Api-Key"] == "img_dev_test"
        return httpx.Response(200, json=payload)

    with patch("fabric.platform.client.httpx.Client") as mock_client:
        instance = mock_client.return_value
        instance.__enter__.return_value = instance
        instance.request.return_value = httpx.Response(200, json=payload)
        cache_patch = "fabric.platform.registry.Settings.registry_cache_path"
        with patch(cache_patch, return_value=tmp_path / "registry"):
            result = fetch_asset_version("B_000010", "1", cache=True)
    assert "split_scheme" in result["config_yaml"]
    cached = tmp_path / "registry" / "B_000010" / "1" / "config.yaml"
    assert cached.is_file()


def test_save_credentials(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("fabric.platform.client.Path.home", lambda: tmp_path)
    path = save_credentials(api_key_value="img_dev_x", api_base_value="http://localhost:8080/v1")
    assert path.is_file()
    assert "img_dev_x" in path.read_text()
