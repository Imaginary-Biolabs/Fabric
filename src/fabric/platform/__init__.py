"""Platform HTTP client (optional extra).

Submodules (:mod:`auth`, :mod:`registry`, :mod:`upload`, :mod:`jobs`) are loaded
lazily and require the ``platform`` extra (``httpx``).

Example:
    >>> import fabric.platform as platform  # doctest: +SKIP
    >>> platform.auth  # doctest: +SKIP
"""

from __future__ import annotations

import importlib

from fabric.utils.errors import PlatformExtraRequired

__all__ = ["auth", "client", "registry", "upload", "jobs"]


def __getattr__(name: str):
    """Lazily import a platform submodule after checking for ``httpx``.

    Args:
        name: Submodule name (one of :attr:`__all__`).

    Returns:
        Imported platform submodule.

    Raises:
        AttributeError: If ``name`` is not a known submodule.
        PlatformExtraRequired: If ``httpx`` is not installed.
    """
    if name not in __all__:
        raise AttributeError(f"module 'fabric.platform' has no attribute {name!r}")
    try:
        import httpx  # noqa: F401
    except ImportError as exc:
        raise PlatformExtraRequired() from exc
    return importlib.import_module(f"fabric.platform.{name}")
