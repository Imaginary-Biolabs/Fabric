"""Workflow node executor for remote platform operations.

Delegates proprietary or hosted nodes to the platform runtime when the runner
mode is ``hybrid`` or ``remote``.
"""

from __future__ import annotations

from typing import Any

from fabric.core.run_record import StepRecord
from fabric.core.workflow_nodes.base import NodeExecutor, register_executor
from fabric.core.workflow_nodes.context import ExecutionContext
from fabric.utils.errors import PlatformExtraRequired, WorkflowError


@register_executor
class PlatformExecutor(NodeExecutor):
    """Execute proprietary or platform-hosted workflow nodes remotely."""

    op = "platform"
    supported_runtimes = ("remote", "platform")

    def execute(self, node_id: str, config: dict[str, Any], ctx: ExecutionContext) -> StepRecord:
        node_type = str(config.get("node") or config.get("platform_node") or "")
        if not node_type:
            raise WorkflowError(f"Node '{node_id}' requires 'node' naming a platform node")

        if ctx.mode == "local":
            raise WorkflowError(
                f"Node '{node_id}' requires platform execution; run with mode='hybrid' or 'remote'"
            )

        resolved_inputs = ctx.resolve(config.get("inputs", {}))
        if ctx.remote_submit is not None:
            outputs = ctx.remote_submit(node_type, config, resolved_inputs)
            ctx.node_outputs[node_id] = outputs
            return StepRecord(
                node_id=node_id,
                op=self.op,
                runtime="platform",
                status="succeeded",
                inputs=resolved_inputs,
                outputs=outputs,
                logs={"platform_node": node_type, "transport": "remote_submit"},
            )

        raise PlatformExtraRequired()
