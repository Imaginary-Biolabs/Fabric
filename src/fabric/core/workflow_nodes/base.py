"""Workflow node executor protocol and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from fabric.core.run_record import StepRecord
from fabric.core.workflow_nodes.context import ExecutionContext
from fabric.utils.errors import WorkflowError

NODE_REGISTRY: dict[str, type[NodeExecutor]] = {}


class NodeExecutor(ABC):
    """Execute one workflow node operation."""

    op: str = "NodeExecutor"
    supported_runtimes: tuple[str, ...] = ("local", "remote", "platform")

    @abstractmethod
    def execute(self, node_id: str, config: dict[str, Any], ctx: ExecutionContext) -> StepRecord:
        raise NotImplementedError


def register_executor(cls: type[NodeExecutor]) -> type[NodeExecutor]:
    NODE_REGISTRY[cls.op] = cls
    return cls


def get_executor(op: str) -> NodeExecutor:
    cls = NODE_REGISTRY.get(op)
    if cls is None:
        available = ", ".join(sorted(NODE_REGISTRY))
        raise WorkflowError(f"Unknown workflow op '{op}'. Available: {available}")
    return cls()


def runtime_allowed(node_runtime: str, runner_mode: str) -> bool:
    """Return whether a node runtime can execute under the selected runner mode."""
    if runner_mode == "local":
        return node_runtime == "local"
    if runner_mode == "remote":
        return node_runtime in {"local", "remote", "platform"}
    if runner_mode == "hybrid":
        return True
    raise WorkflowError(f"Unknown runner mode '{runner_mode}'")
