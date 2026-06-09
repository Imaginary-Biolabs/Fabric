"""Workflow node execution context."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fabric.core.workflow import WorkflowPlan, resolve_value


@dataclass
class ExecutionContext:
    """Mutable state shared across node executors during a run."""

    plan: WorkflowPlan
    mode: str
    root: Path
    node_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    remote_submit: Callable[[str, dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None

    @property
    def inputs(self) -> dict[str, Any]:
        return self.plan.inputs

    def resolve(self, value: Any) -> Any:
        return resolve_value(value, inputs=self.plan.inputs, node_outputs=self.node_outputs)
