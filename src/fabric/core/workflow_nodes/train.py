"""Workflow node executor for model training (Develop tab)."""

from __future__ import annotations

from typing import Any

from fabric.core.run_record import StepRecord
from fabric.core.workflow_nodes.base import NodeExecutor, register_executor
from fabric.core.workflow_nodes.context import ExecutionContext


@register_executor
class TrainExecutor(NodeExecutor):
    """Fine-tune or train a model from a develop profile."""

    op = "train"
    supported_runtimes = ("local", "remote")

    def execute(self, node_id: str, config: dict[str, Any], ctx: ExecutionContext) -> StepRecord:
        ctx.emit_step("validate_inputs")
        model_ref = str(config.get("model") or "")
        if not model_ref:
            raise ValueError(f"Node '{node_id}' requires 'model'")

        parameters = config.get("parameters") or {}
        epochs = int(parameters.get("epochs") or 10)
        learning_rate = float(parameters.get("learning_rate") or 1e-4)
        profile = str(config.get("profile") or "finetune")

        ctx.emit_step("load_checkpoint", {"profile": profile})
        ctx.emit_step("collate_batch")
        loss_series: list[float] = []
        for epoch in range(epochs):
            ctx.emit_step("train_epoch", {"epoch": epoch + 1, "transport": "local"})
            loss_series.append(round(1.0 - (epoch + 1) * (0.8 / max(epochs, 1)), 4))

        ctx.emit_step("write_artifacts")
        metrics = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "loss": loss_series,
            "final_loss": loss_series[-1] if loss_series else 0.0,
        }
        checkpoint = {"format": "torch_pt", "stub": True}
        outputs = {
            "metrics": metrics,
            "checkpoint": checkpoint,
        }
        ctx.node_outputs[node_id] = outputs
        return StepRecord(
            node_id=node_id,
            op=self.op,
            runtime=str(config.get("runtime") or "local"),
            status="succeeded",
            inputs={"model": model_ref, "profile": profile},
            outputs=outputs,
            logs={"epochs": epochs},
        )
