"""Workflow node executor for ad-hoc model inference (Run tab)."""

from __future__ import annotations

from typing import Any

import grumpy as gr

from fabric.core.backends import build_backend
from fabric.core.collater import CollatedBatch
from fabric.core.factory import Factory
from fabric.core.models import attach_optimizer, build_model
from fabric.core.run_record import StepRecord
from fabric.core.workflow_nodes.base import NodeExecutor, register_executor
from fabric.core.workflow_nodes.context import ExecutionContext
from fabric.utils.config import load_config
from fabric.utils.errors import WorkflowError


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise WorkflowError(
            "predict op requires PyTorch: pip install 'imaginary-fabric[torch]'"
        ) from exc
    return torch


def _format_prediction_output(values: list[float]) -> dict[str, Any]:
    rows = [[f"sample_{index + 1}", float(value)] for index, value in enumerate(values)]
    return {"columns": ["sample_id", "prediction"], "rows": rows}


@register_executor
class PredictExecutor(NodeExecutor):
    """Run ad-hoc inference for a configured model."""

    op = "predict"
    supported_runtimes = ("local", "remote")

    def execute(self, node_id: str, config: dict[str, Any], ctx: ExecutionContext) -> StepRecord:
        ctx.emit_step("validate_inputs")
        resolved = ctx.resolve(config.get("inputs", {}))
        features = resolved.get("features")
        if features is None:
            raise WorkflowError(f"Node '{node_id}' requires 'features' in inputs")

        model_ref = str(config.get("model") or "")
        if not model_ref:
            raise WorkflowError(f"Node '{node_id}' requires 'model'")

        if model_ref in ctx.model_configs:
            model_cfg = ctx.model_configs[model_ref]
        else:
            model_cfg = load_config(Factory._resolve_model_path(model_ref))

        ctx.emit_step("load_checkpoint")
        torch = _require_torch()
        if not isinstance(features, list):
            raise WorkflowError("features must be a list of numeric vectors")

        tensor_rows: list[list[float]] = []
        for row in features:
            if not isinstance(row, list):
                raise WorkflowError("each features row must be a list of numbers")
            tensor_rows.append([float(v) for v in row])

        feature_tensor = torch.tensor(tensor_rows, dtype=torch.float32)
        input_dim = int(feature_tensor.shape[1])

        ctx.emit_step("collate_batch", {"batch_size": int(feature_tensor.shape[0])})
        model = build_model(model_cfg, input_dim=input_dim)
        attach_optimizer(model, model_cfg.get("optimizer"))
        trainer_defaults = model_cfg.get("trainer_defaults") or {}
        backend_cfg = config.get("backend") or trainer_defaults.get("backend")
        if backend_cfg is None:
            backend_cfg = {"TorchBackend": {"accelerator": "cpu"}}
        elif not isinstance(backend_cfg, dict):
            backend_cfg = {str(backend_cfg): {"accelerator": "cpu"}}
        backend = build_backend(backend_cfg)
        model = backend.setup(model)

        features_gr = gr.from_torch(feature_tensor.detach().cpu(), dtype=gr.float32)
        batch_size = int(feature_tensor.shape[0])
        batch = CollatedBatch(
            features=features_gr,
            y=gr.array([0.0] * batch_size, dtype=gr.float32),
            meta={"layout": "flat", "slots": {"features": features_gr}},
        )

        ctx.emit_step("forward_pass")
        with backend.no_grad():
            step = model.validation_step(batch, backend=backend)
        values = step.predictions.flatten().tolist()

        ctx.emit_step("postprocess")
        table = _format_prediction_output(values)
        outputs = {"predictions": table}
        ctx.node_outputs[node_id] = outputs
        return StepRecord(
            node_id=node_id,
            op=self.op,
            runtime=str(config.get("runtime") or "local"),
            status="succeeded",
            inputs={"feature_rows": len(tensor_rows)},
            outputs=outputs,
            logs={"model": model_ref},
        )
