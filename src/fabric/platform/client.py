"""HTTP client and credential helpers for the Imaginary platform API.

Reads ``IMAGINARY_API_BASE`` and ``IMAGINARY_API_KEY`` from the environment,
falling back to ``~/.imaginary/credentials.yaml``. Use :class:`PlatformClient`
for authenticated JSON requests.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import yaml

from fabric.utils.errors import AuthError, FabricError, RegistryError


def api_base() -> str:
    """Return the platform API base URL.

    Returns:
        Base URL without a trailing slash (for example ``http://localhost:8080/v1``).

    Raises:
        AuthError: If neither ``IMAGINARY_API_BASE`` nor credentials provide a base URL.

    Example:
        >>> import os
        >>> os.environ["IMAGINARY_API_BASE"] = "http://localhost:8080/v1"
        >>> api_base()
        'http://localhost:8080/v1'
    """
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
    """Return the Imaginary API key.

    Returns:
        API key string from the environment or credentials file.

    Raises:
        AuthError: If no key is configured.

    Example:
        >>> import os
        >>> os.environ["IMAGINARY_API_KEY"] = "test-key"
        >>> api_key()
        'test-key'
    """
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
    """Load platform credentials from a YAML file.

    Args:
        path: Credentials file path. Defaults to ``~/.imaginary/credentials.yaml``.

    Returns:
        Parsed YAML mapping, or an empty dict when the file is missing.

    Example:
        >>> load_credentials(Path("/nonexistent/credentials.yaml"))
        {}
    """
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
    """Write API credentials to disk.

    Args:
        api_key_value: API key to store.
        api_base_value: API base URL. Defaults to ``http://localhost:8080/v1``.
        path: Output file path. Defaults to ``~/.imaginary/credentials.yaml``.

    Returns:
        Path to the written credentials file.

    Example:
        >>> from pathlib import Path
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     out = save_credentials(
        ...         api_key_value="secret",
        ...         path=Path(tmp) / "credentials.yaml",
        ...     )
        ...     out.name
        'credentials.yaml'
    """
    cred_path = path or Path.home() / ".imaginary" / "credentials.yaml"
    cred_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"api_key": api_key_value, "api_base": api_base_value}
    cred_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return cred_path


class PlatformClient:
    """Authenticated JSON client for the Imaginary platform API.

    Sends ``X-Imaginary-Api-Key`` on every request and optionally
    ``X-Imaginary-Org-Id`` when ``IMAGINARY_ORG_ID`` is set.

    Example:
        >>> client = PlatformClient(base_url="http://localhost:8080/v1", key="test")
        >>> client.base_url
        'http://localhost:8080/v1'
    """

    def __init__(self, *, base_url: str | None = None, key: str | None = None) -> None:
        """Create a client with optional explicit base URL and API key.

        Args:
            base_url: API base URL. Defaults to :func:`api_base`.
            key: API key. Defaults to :func:`api_key`.
        """
        self.base_url = (base_url or api_base()).rstrip("/")
        self._key = key or api_key()

    def _headers(self) -> dict[str, str]:
        headers = {"X-Imaginary-Api-Key": self._key}
        org = os.environ.get("IMAGINARY_ORG_ID")
        if org:
            headers["X-Imaginary-Org-Id"] = org
        return headers

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Send an authenticated HTTP request and parse the JSON body.

        Args:
            method: HTTP verb (``GET``, ``POST``, …).
            path: API path beginning with ``/`` (for example ``/assets/{id}``).
            **kwargs: Forwarded to :func:`httpx.Client.request` (``json``, ``params``, …).

        Returns:
            Parsed JSON response, or ``None`` for HTTP 204.

        Raises:
            AuthError: On HTTP 401/403.
            RegistryError: On asset or version not found.
            FabricError: On other API errors.
        """
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
