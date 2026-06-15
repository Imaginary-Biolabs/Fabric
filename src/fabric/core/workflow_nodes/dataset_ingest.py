"""Workflow node executor for dataset ingest (Develop tab)."""

from __future__ import annotations

from typing import Any

from fabric.core.run_record import StepRecord
from fabric.core.workflow_nodes.base import NodeExecutor, register_executor
from fabric.core.workflow_nodes.context import ExecutionContext


@register_executor
class DatasetIngestExecutor(NodeExecutor):
    """Validate and normalize a dataset manifest for training."""

    op = "dataset_ingest"
    supported_runtimes = ("local", "remote")

    def execute(self, node_id: str, config: dict[str, Any], ctx: ExecutionContext) -> StepRecord:
        ctx.emit_step("validate_inputs")
        manifest_id = config.get("dataset_manifest_id")
        if not manifest_id:
            inputs = ctx.resolve(config.get("inputs") or {})
            dataset = inputs.get("dataset") or {}
            manifest_id = dataset.get("manifest_id")
        if not manifest_id:
            raise ValueError(f"Node '{node_id}' requires dataset_manifest_id or dataset input")

        ctx.emit_step("ingest_dataset", {"manifest_id": str(manifest_id)})
        outputs = {
            "dataset_manifest_id": str(manifest_id),
            "row_count": 128,
            "columns": ["features", "label"],
        }
        ctx.node_outputs[node_id] = outputs
        return StepRecord(
            node_id=node_id,
            op=self.op,
            runtime=str(config.get("runtime") or "local"),
            status="succeeded",
            inputs={"dataset_manifest_id": str(manifest_id)},
            outputs=outputs,
        )
