"""Workflow node executor for pause and resume.

Blocks a workflow until an external resume payload is supplied.
"""

from __future__ import annotations

from typing import Any

from fabric.core.run_record import StepRecord
from fabric.core.workflow_nodes.base import NodeExecutor, register_executor
from fabric.core.workflow_nodes.context import ExecutionContext


@register_executor
class WaitExecutor(NodeExecutor):
    """Pause a workflow until user or external input resumes the run."""

    op = "wait"
    supported_runtimes = ("local", "remote", "platform")

    def execute(self, node_id: str, config: dict[str, Any], ctx: ExecutionContext) -> StepRecord:
        kind = str(config.get("kind") or "user_input")
        present = ctx.resolve(config.get("present", {}))
        resume_payload = config.get("resume")
        if resume_payload is None:
            ctx.node_outputs[node_id] = {"status": "paused", "present": present}
            return StepRecord(
                node_id=node_id,
                op=self.op,
                runtime=str(config.get("runtime") or "local"),
                status="paused",
                inputs={"kind": kind, "present": present},
                outputs={"status": "paused"},
                logs={"message": "Workflow paused until resume payload is supplied"},
            )
        outputs = ctx.resolve(resume_payload)
        ctx.node_outputs[node_id] = outputs
        return StepRecord(
            node_id=node_id,
            op=self.op,
            runtime=str(config.get("runtime") or "local"),
            status="succeeded",
            inputs={"kind": kind, "present": present},
            outputs=outputs,
        )
