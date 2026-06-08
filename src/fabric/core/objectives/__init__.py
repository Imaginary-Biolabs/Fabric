"""Objective registry."""

from fabric.core.objectives.supervised import SupervisedObjective

OBJECTIVE_REGISTRY: dict[str, type[SupervisedObjective]] = {
    "SupervisedObjective": SupervisedObjective,
}

__all__ = ["SupervisedObjective", "OBJECTIVE_REGISTRY"]
