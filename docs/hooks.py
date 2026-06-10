"""MkDocs hooks for Fabric documentation builds."""

from __future__ import annotations


def on_pre_build(config) -> None:  # noqa: ANN001
    """Optional pre-build steps (reserved for generated assets)."""
