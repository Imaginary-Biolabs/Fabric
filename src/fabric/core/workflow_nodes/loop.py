"""Workflow node executor for fixed-count iteration.

Repeats a nested body of nodes and exposes per-iteration and last-iteration outputs.
"""

from __future__ import annotations

from typing import Any

from fabric.core.run_record import StepRecord
from fabric.core.workflow_nodes.base import NodeExecutor, get_executor, register_executor
from fabric.core.workflow_nodes.context import ExecutionContext
from fabric.utils.errors import WorkflowError


@register_executor
class LoopExecutor(NodeExecutor):
    """Repeat a nested body of nodes a fixed number of times."""

    op = "loop"
    supported_runtimes = ("local", "remote", "platform")

    def execute(self, node_id: str, config: dict[str, Any], ctx: ExecutionContext) -> StepRecord:
        count = int(ctx.resolve(config.get("count", 1)))
        body = config.get("body")
        if not isinstance(body, dict) or not body:
            raise WorkflowError(f"Node '{node_id}' requires a non-empty 'body' mapping")

        iterations: list[dict[str, Any]] = []
        last_outputs: dict[str, Any] = {}
        for index in range(count):
            iteration_outputs: dict[str, Any] = {}
            for body_id, body_cfg in body.items():
                if not isinstance(body_cfg, dict):
                    raise WorkflowError(f"Loop body node '{body_id}' must be a mapping")
                scoped_id = f"{node_id}__{index}__{body_id}"
                executor = get_executor(str(body_cfg.get("op") or ""))
                step = executor.execute(scoped_id, body_cfg, ctx)
                iteration_outputs[body_id] = step.outputs
                last_outputs = step.outputs
            iterations.append({"index": index, "outputs": iteration_outputs})

        last_prefixed = {f"last_{key}": value for key, value in last_outputs.items()}
        outputs = {"iterations": iterations, **last_prefixed}
        ctx.node_outputs[node_id] = outputs
        return StepRecord(
            node_id=node_id,
            op=self.op,
            runtime=str(config.get("runtime") or "local"),
            status="succeeded",
            inputs={"count": count},
            outputs=outputs,
            iterations=iterations,
        )
