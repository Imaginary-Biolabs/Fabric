from __future__ import annotations

from typing import Any

from fabric.core.run_record import StepRecord
from fabric.core.workflow_nodes.base import NodeExecutor, register_executor
from fabric.core.workflow_nodes.context import ExecutionContext
from fabric.utils.errors import WorkflowError


@register_executor
class ValueExecutor(NodeExecutor):
    """Materialize static or referenced values as node outputs."""

    op = "value"
    supported_runtimes = ("local", "remote", "platform")

    def execute(self, node_id: str, config: dict[str, Any], ctx: ExecutionContext) -> StepRecord:
        outputs_spec = config.get("outputs", {})
        if not isinstance(outputs_spec, dict):
            raise WorkflowError(f"Node '{node_id}' requires an 'outputs' mapping")
        outputs = ctx.resolve(outputs_spec)
        ctx.node_outputs[node_id] = outputs
        return StepRecord(
            node_id=node_id,
            op=self.op,
            runtime=str(config.get("runtime") or "local"),
            status="succeeded",
            outputs=outputs,
        )
