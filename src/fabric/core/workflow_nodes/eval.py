from __future__ import annotations

from typing import Any

import grumpy as gr

from fabric.core.backends import build_backend
from fabric.core.collaters import build_collater
from fabric.core.factory import Factory
from fabric.core.models import attach_optimizer, build_model
from fabric.core.run_record import StepRecord
from fabric.core.workflow_nodes.base import NodeExecutor, register_executor
from fabric.core.workflow_nodes.context import ExecutionContext
from fabric.utils.config import load_config
from fabric.utils.errors import WorkflowError


@register_executor
class EvalExecutor(NodeExecutor):
    """Run benchmark evaluation for a configured model."""

    op = "eval"
    supported_runtimes = ("local", "remote")

    def execute(self, node_id: str, config: dict[str, Any], ctx: ExecutionContext) -> StepRecord:
        resolved = ctx.resolve(config.get("inputs", {}))
        batch_size = int(resolved.get("batch_size", config.get("batch_size", 8)))
        split = str(resolved.get("split", config.get("split", "test")))

        benchmark_ref = str(config.get("benchmark") or "")
        model_ref = str(config.get("model") or "")
        if not benchmark_ref or not model_ref:
            raise WorkflowError(f"Node '{node_id}' requires 'benchmark' and 'model'")

        benchmark = Factory.benchmark(benchmark_ref)
        collater = build_collater(config.get("collater"))
        model_cfg = load_config(Factory._resolve_model_path(model_ref))
        model = build_model(model_cfg, collater=collater)
        attach_optimizer(model, model_cfg.get("optimizer"))
        backend = build_backend(config.get("backend", {"TorchBackend": {"accelerator": "cpu"}}))
        model = backend.setup(model)

        benchmark.metrics.reset()
        loader = getattr(benchmark, f"{split}_loader")(batch_size=batch_size, progress=False)
        for X, y in loader:
            batch = collater.collate(X, y)
            _, predictions = backend.eval_step(model, batch)
            benchmark.update(predictions.astype(gr.float64), y)
        metrics = benchmark.metrics.compute_all()

        outputs = {"metrics": metrics}
        ctx.node_outputs[node_id] = outputs
        return StepRecord(
            node_id=node_id,
            op=self.op,
            runtime=str(config.get("runtime") or "local"),
            status="succeeded",
            inputs={"batch_size": batch_size, "split": split},
            outputs=outputs,
            logs={"benchmark": benchmark.id, "model": model_ref},
        )
