"""Platform HTTP client (optional extra)."""

from __future__ import annotations

import importlib

from fabric.utils.errors import PlatformExtraRequired

__all__ = ["auth", "registry", "upload", "jobs"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module 'fabric.platform' has no attribute {name!r}")
    try:
        import httpx  # noqa: F401
    except ImportError as exc:
        raise PlatformExtraRequired() from exc
    return importlib.import_module(f"fabric.platform.{name}")
