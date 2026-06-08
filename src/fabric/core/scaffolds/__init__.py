"""Scaffold registry."""

from fabric.core.scaffolds.mlp import MLPScaffold

SCAFFOLD_REGISTRY: dict[str, type[MLPScaffold]] = {
    "MLPScaffold": MLPScaffold,
}

__all__ = ["MLPScaffold", "SCAFFOLD_REGISTRY"]
