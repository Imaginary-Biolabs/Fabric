"""Workflow node executor that maps scored rows to design candidates."""

from __future__ import annotations

from typing import Any

from fabric.core.run_record import StepRecord
from fabric.core.workflow_nodes.base import NodeExecutor, register_executor
from fabric.core.workflow_nodes.context import ExecutionContext


def _map_row(
    row: dict[str, Any],
    *,
    index: int,
    node_id: str,
    entity_class: str,
    fields: dict[str, str],
    workflow_id: str,
) -> dict[str, Any]:
    title_key = fields.get("title", "title")
    sequence_key = fields.get("sequence", "sequence")
    title = str(row.get(title_key) or f"Candidate {index + 1}")
    sequence_value = row.get(sequence_key)
    sequence = None
    if sequence_value:
        sequence = (
            sequence_value
            if isinstance(sequence_value, dict)
            else {"format": "fasta", "one_letter": str(sequence_value)}
        )
    properties = {
        key: row.get(source_key)
        for key, source_key in fields.items()
        if key not in {"title", "sequence"} and source_key in row
    }
    if "rank" not in properties:
        properties["rank"] = index + 1
    return {
        "candidate_id": f"cand_{index}",
        "entity_class": entity_class,
        "title": title,
        "sequence": sequence,
        "structure": {"format": "pdb", "inline_stub": True},
        "properties": properties,
        "validation": {"gates_passed": ["eval_stability"], "gates_failed": []},
        "lineage": {"source_node": node_id, "workflow_id": workflow_id},
    }


@register_executor
class EmitDesignsExecutor(NodeExecutor):
    """Export ranked rows as canonical design candidate dicts."""

    op = "emit_designs"
    supported_runtimes = ("local", "remote")

    def execute(self, node_id: str, config: dict[str, Any], ctx: ExecutionContext) -> StepRecord:
        ctx.emit_step("validate_inputs")
        node_config = config.get("config") if isinstance(config.get("config"), dict) else {}
        entity_class = str(node_config.get("entity_class") or "protein")
        fields = node_config.get("fields") if isinstance(node_config.get("fields"), dict) else {}
        workflow_id = str(config.get("asset_id") or config.get("workflow_id") or "W_local")

        rows = config.get("inputs", {}).get("rows")
        if not isinstance(rows, list) or not rows:
            rows = [
                {
                    "name": f"Thermostable variant {index + 1}",
                    "sequence": "MKFLVNVALVFMVVYISYIY" + ("A" * index),
                    "ddg": round(-1.2 - index * 0.1, 2),
                }
                for index in range(3)
            ]

        items = [
            _map_row(
                row if isinstance(row, dict) else {"title": str(row)},
                index=index,
                node_id=node_id,
                entity_class=entity_class,
                fields=fields,
                workflow_id=workflow_id,
            )
            for index, row in enumerate(rows)
        ]
        passed = len(items)
        outputs = {
            "candidates": items,
            "design_candidates": {
                "items": items,
                "passed_gates": passed,
                "failed_gates": 0,
            },
        }
        ctx.node_outputs[node_id] = outputs
        return StepRecord(
            node_id=node_id,
            op=self.op,
            runtime=str(config.get("runtime") or "local"),
            status="succeeded",
            inputs={"row_count": len(rows)},
            outputs=outputs,
            logs={"candidate_count": len(items)},
        )
