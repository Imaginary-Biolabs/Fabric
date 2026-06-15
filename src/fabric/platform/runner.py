"""Remote platform workflow execution via workflow_run jobs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fabric.core.run_record import RunRecord, StepRecord
from fabric.core.workflow import Workflow, resolve_output_ref
from fabric.core.workflow_nodes.context import ExecutionContext
from fabric.platform.jobs import (
    get_job_events,
    submit_benchmark_eval,
    submit_workflow_run,
    wait_for_job,
)
from fabric.utils.errors import WorkflowError

_REGISTRY_ASSET = re.compile(r"^[DBMLWC]_[0-9]{6}$")
_REGISTRY_WORKFLOW = re.compile(r"^W_[0-9]{6}$")


def _resolve_node_inputs(plan_inputs: dict[str, Any], node_inputs: dict[str, Any]) -> dict[str, Any]:
    ctx = ExecutionContext(
        plan=type("_Plan", (), {"inputs": plan_inputs, "nodes": {}, "execution_order": []})(),
        mode="remote",
        root=Path("."),
    )
    return ctx.resolve(node_inputs)


def plan_remote_submission(workflow: Workflow, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Map a single-node workflow to a platform job submission spec.

    Returns a dict with ``kind`` set to ``workflow_run`` or ``benchmark_eval``
    plus the keyword arguments for the matching submit helper.

    Raises:
        WorkflowError: When the workflow cannot run remotely.
    """
    plan = workflow.compile(run_inputs=inputs)
    wf_id = plan.workflow_id

    if _REGISTRY_WORKFLOW.match(wf_id):
        return {
            "kind": "workflow_asset",
            "asset_id": wf_id,
            "version": "1",
            "mode": "run",
            "inputs": dict(plan.inputs),
            "parameters": {},
        }

    if len(plan.execution_order) != 1:
        raise WorkflowError(
            "Remote mode supports registry workflow assets (W_000NNN) or single-node "
            f"predict/eval/train graphs. This workflow has {len(plan.execution_order)} nodes."
        )

    node_id = plan.execution_order[0]
    node = plan.nodes[node_id]
    config = dict(node.config)
    resolved = _resolve_node_inputs(plan.inputs, config.get("inputs") or {})

    if node.op == "predict":
        model = str(config.get("model") or "")
        if not _REGISTRY_ASSET.match(model):
            raise WorkflowError(
                f"Remote predict requires a registry model id (M_000NNN), got {model!r}"
            )
        run_inputs = dict(plan.inputs)
        if "features" in resolved:
            run_inputs["features"] = resolved["features"]
        parameters = {
            key: resolved[key]
            for key in ("batch_size",)
            if key in resolved
        }
        return {
            "kind": "workflow_run",
            "asset_id": model,
            "version": str(config.get("model_version") or "1"),
            "mode": "run",
            "inputs": run_inputs,
            "parameters": parameters,
        }

    if node.op == "eval":
        benchmark = str(config.get("benchmark") or "")
        model = str(config.get("model") or "")
        if not _REGISTRY_ASSET.match(benchmark) or not _REGISTRY_ASSET.match(model):
            raise WorkflowError(
                "Remote eval requires registry asset ids (B_000NNN, M_000NNN)"
            )
        overrides: dict[str, Any] = {}
        if "batch_size" in resolved:
            overrides["batch_size"] = resolved["batch_size"]
        if "split" in resolved:
            overrides["split"] = resolved["split"]
        return {
            "kind": "benchmark_eval",
            "benchmark_id": benchmark,
            "benchmark_version": str(config.get("benchmark_version") or "1"),
            "model_id": model,
            "model_version": str(config.get("model_version") or "1"),
            "overrides": overrides,
        }

    if node.op == "train":
        model = str(config.get("model") or "")
        if not _REGISTRY_ASSET.match(model):
            raise WorkflowError(
                f"Remote train requires a registry model id (M_000NNN), got {model!r}"
            )
        parameters = dict(config.get("parameters") or {})
        for key in ("epochs", "learning_rate", "batch_size"):
            if key in resolved:
                parameters[key] = resolved[key]
        develop_inputs = dict(plan.inputs)
        if resolved.get("dataset"):
            develop_inputs["dataset"] = resolved["dataset"]
        return {
            "kind": "workflow_run",
            "asset_id": model,
            "version": str(config.get("model_version") or "1"),
            "mode": "develop",
            "inputs": develop_inputs,
            "parameters": parameters,
            "develop_profile": str(config.get("profile") or "finetune"),
        }

    if node.op == "platform":
        raise WorkflowError(
            "Platform-only workflows cannot run remotely yet; use hybrid mode with a "
            "custom remote_submit hook or submit a model workflow_run job directly."
        )

    raise WorkflowError(
        f"Remote mode does not support op '{node.op}' on node '{node_id}'"
    )


def _job_to_run_record(
    workflow: Workflow,
    *,
    job: dict[str, Any],
    root: Path,
    workflow_inputs: dict[str, Any] | None,
) -> RunRecord:
    plan = workflow.compile(run_inputs=workflow_inputs)
    record = RunRecord.new(
        workflow_id=plan.workflow_id,
        mode="remote",
        root=root,
        workflow_hash=plan.workflow_hash,
    )
    record.inputs = dict(plan.inputs)
    record.status = job["status"]
    record.finished_at = job.get("finished_at") or ""

    result = job.get("result") or {}
    outputs_payload = result.get("outputs") or result

    for node_id in plan.execution_order:
        node = plan.nodes.get(node_id)
        step_outputs = outputs_payload if isinstance(outputs_payload, dict) else {"result": outputs_payload}
        record.steps[node_id] = StepRecord(
            node_id=node_id,
            op=node.op if node else "remote",
            runtime="remote",
            status="succeeded" if job["status"] == "succeeded" else job["status"],
            outputs=step_outputs if node_id == plan.execution_order[-1] else {},
            logs={"job_id": job.get("id"), "platform_job": True},
        )

    if isinstance(outputs_payload, dict):
        record.outputs = outputs_payload
    elif plan.execution_order:
        last_id = plan.execution_order[-1]
        record.outputs = {
            name: resolve_output_ref(ref, node_outputs={last_id: outputs_payload}, inputs=plan.inputs)
            for name, ref in plan.outputs.items()
        }
    record.save()
    return record


def run_workflow_remote(
    workflow: Workflow,
    *,
    inputs: dict[str, Any] | None = None,
    root: str | Path = "results/workflows",
    timeout_s: float = 300.0,
    poll_s: float = 1.0,
) -> RunRecord:
    """Execute a workflow on the platform via ``workflow_run`` or eval shim.

    Supports single-node predict, eval, and train workflows with registry asset ids.
    """
    spec = plan_remote_submission(workflow, inputs)
    kind = spec.pop("kind")

    if kind == "workflow_run":
        job = submit_workflow_run(**spec)
    elif kind == "workflow_asset":
        spec.pop("mode", None)
        job = submit_workflow_run(**spec)
    elif kind == "benchmark_eval":
        job = submit_benchmark_eval(**spec)
    else:
        raise WorkflowError(f"Unsupported remote submission kind: {kind}")

    job_id = job["id"]
    events: list[dict[str, Any]] = []

    def _on_poll(latest: dict[str, Any]) -> None:
        nonlocal events
        fresh = get_job_events(job_id)
        if len(fresh) > len(events):
            events = fresh

    final = wait_for_job(job_id, timeout_s=timeout_s, poll_s=poll_s, on_poll=_on_poll)
    return _job_to_run_record(
        workflow,
        job=final,
        root=Path(root),
        workflow_inputs=inputs,
    )


def make_remote_submit(*, workflow_asset_id: str | None = None):
    """Build a ``remote_submit`` callback for hybrid/remote platform node execution."""

    def _remote_submit(node_type: str, config: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
        if _REGISTRY_WORKFLOW.match(node_type):
            job = submit_workflow_run(asset_id=node_type, inputs=inputs)
            final = wait_for_job(job["id"], timeout_s=600.0)
            outputs = (final.get("result") or {}).get("outputs") or {}
            return outputs if isinstance(outputs, dict) else {"result": outputs}

        if workflow_asset_id and _REGISTRY_WORKFLOW.match(workflow_asset_id):
            job = submit_workflow_run(
                asset_id=workflow_asset_id,
                inputs={"platform_node": node_type, **inputs},
            )
            final = wait_for_job(job["id"], timeout_s=600.0)
            outputs = (final.get("result") or {}).get("outputs") or {}
            if isinstance(outputs, dict):
                for key in ("structures", "designs", "predictions"):
                    if key in outputs:
                        return {key: outputs[key]} if key != "designs" else {"structures": outputs[key]}
                return outputs
            return {"structures": [f"{node_type}_stub"]}

        raise WorkflowError(
            f"Platform node {node_type!r} requires a registry workflow asset remote runner. "
            "Publish a W_* workflow containing the node or use local hybrid mode."
        )

    return _remote_submit


def platform_runner(
    *,
    root: str | Path = "results/workflows",
    mode: str = "remote",
    workflow_asset_id: str | None = None,
):
    """Return a :class:`~fabric.core.runner.Runner` wired for platform execution.

    For single-node model workflows, prefer :func:`run_workflow_remote`. This helper
    configures ``remote_submit`` for hybrid/remote DAG execution.
    """
    from fabric.core.runner import Runner

    return Runner(mode=mode, root=root, remote_submit=make_remote_submit(workflow_asset_id=workflow_asset_id))
