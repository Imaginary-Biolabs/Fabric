"""Workflow runner with local, remote, and hybrid execution modes."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fabric.core.run_record import RunRecord
from fabric.core.workflow import Workflow, resolve_output_ref
from fabric.core.workflow_nodes import ExecutionContext, get_executor
from fabric.core.workflow_nodes.base import runtime_allowed
from fabric.utils.errors import WorkflowError

_VALID_MODES = frozenset({"local", "remote", "hybrid"})


RemoteSubmit = Callable[[str, dict[str, Any], dict[str, Any]], dict[str, Any]]


class Runner:
    """Execute compiled workflow plans and persist inspectable run records.

    Args:
        mode: ``local`` runs only local nodes, ``remote`` allows platform nodes,
            ``hybrid`` runs local nodes locally and delegates platform nodes remotely.
        root: Directory where run artifacts are written.
        remote_submit: Optional callback for platform node execution during tests
            or platform integration.
    """

    def __init__(
        self,
        *,
        mode: str = "local",
        root: str | Path = "results/workflows",
        remote_submit: RemoteSubmit | None = None,
    ) -> None:
        if mode not in _VALID_MODES:
            raise WorkflowError(f"Runner mode must be one of {sorted(_VALID_MODES)}; got {mode!r}")
        self.mode = mode
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.remote_submit = remote_submit

    def run(
        self,
        workflow: Workflow,
        *,
        inputs: dict[str, Any] | None = None,
        resume: dict[str, dict[str, Any]] | None = None,
    ) -> RunRecord:
        """Compile and execute a workflow, returning a persisted run record."""
        plan = workflow.compile(run_inputs=inputs)
        record = RunRecord.new(
            workflow_id=plan.workflow_id,
            mode=self.mode,
            root=self.root,
            workflow_hash=plan.workflow_hash,
        )
        record.inputs = dict(plan.inputs)
        record.status = "running"

        ctx = ExecutionContext(
            plan=plan,
            mode=self.mode,
            root=self.root / record.run_id,
            remote_submit=self.remote_submit,
        )
        ctx.root.mkdir(parents=True, exist_ok=True)

        try:
            for node_id in plan.execution_order:
                node = plan.nodes[node_id]
                if not runtime_allowed(node.runtime, self.mode):
                    raise WorkflowError(
                        f"Node '{node_id}' requires runtime '{node.runtime}' but runner mode is "
                        f"'{self.mode}'"
                    )
                node_config = dict(node.config)
                if resume and node_id in resume:
                    node_config["resume"] = resume[node_id]
                executor = get_executor(node.op)
                step = executor.execute(node_id, node_config, ctx)
                record.steps[node_id] = step
                if step.status == "paused":
                    record.status = "paused"
                    record.finished_at = datetime.now(timezone.utc).isoformat()
                    record.save()
                    return record

            record.outputs = {
                name: resolve_output_ref(ref, node_outputs=ctx.node_outputs, inputs=plan.inputs)
                for name, ref in plan.outputs.items()
            }
            record.status = "succeeded"
        except Exception as exc:
            record.status = "failed"
            record.outputs = {"error": str(exc)}
            record.finished_at = datetime.now(timezone.utc).isoformat()
            record.save()
            raise

        record.finished_at = datetime.now(timezone.utc).isoformat()
        record.save()
        return record
