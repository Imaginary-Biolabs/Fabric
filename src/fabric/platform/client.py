"""Shared HTTP client for Imaginary platform API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import yaml

from fabric.utils.errors import AuthError, FabricError, RegistryError


def api_base() -> str:
    base = os.environ.get("IMAGINARY_API_BASE", "").rstrip("/")
    if not base:
        creds = load_credentials()
        base = str(creds.get("api_base", "")).rstrip("/")
    if not base:
        raise AuthError(
            "Set IMAGINARY_API_BASE or add api_base to ~/.imaginary/credentials.yaml"
        )
    return base


def api_key() -> str:
    key = os.environ.get("IMAGINARY_API_KEY", "")
    if not key:
        creds = load_credentials()
        key = str(creds.get("api_key", ""))
    if not key:
        raise AuthError(
            "Set IMAGINARY_API_KEY or add api_key to ~/.imaginary/credentials.yaml"
        )
    return key


def load_credentials(path: Path | None = None) -> dict[str, Any]:
    """Load ~/.imaginary/credentials.yaml if present."""
    cred_path = path or Path.home() / ".imaginary" / "credentials.yaml"
    if not cred_path.is_file():
        return {}
    return yaml.safe_load(cred_path.read_text()) or {}


def save_credentials(
    *,
    api_key_value: str,
    api_base_value: str = "http://localhost:8080/v1",
    path: Path | None = None,
) -> Path:
    cred_path = path or Path.home() / ".imaginary" / "credentials.yaml"
    cred_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"api_key": api_key_value, "api_base": api_base_value}
    cred_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return cred_path


class PlatformClient:
    """Thin httpx wrapper with Imaginary auth headers."""

    def __init__(self, *, base_url: str | None = None, key: str | None = None) -> None:
        self.base_url = (base_url or api_base()).rstrip("/")
        self._key = key or api_key()

    def _headers(self) -> dict[str, str]:
        headers = {"X-Imaginary-Api-Key": self._key}
        org = os.environ.get("IMAGINARY_ORG_ID")
        if org:
            headers["X-Imaginary-Org-Id"] = org
        return headers

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=120.0) as client:
            response = client.request(method, url, headers=self._headers(), **kwargs)
        if response.status_code >= 400:
            self._raise_api_error(response)
        if response.status_code == 204:
            return None
        return response.json()

    @staticmethod
    def _raise_api_error(response: httpx.Response) -> None:
        try:
            payload = response.json()
            error = payload.get("error") or {}
            code = error.get("code", "internal_error")
            message = error.get("message", response.text)
        except Exception:
            code = "internal_error"
            message = response.text
        if response.status_code in {401, 403}:
            raise AuthError(message)
        if code in {"asset_not_found", "version_not_found"}:
            raise RegistryError(message)
        raise FabricError(message)
