"""YAML workflow specification, planning, and expression resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from fabric.utils.errors import WorkflowError
from fabric.utils.hashing import config_hash

_REF_PATTERN = re.compile(r"^\$(inputs|nodes)\.([A-Za-z0-9_.]+)(?:\.([A-Za-z0-9_]+))?$")
_RUNTIME_LOCAL = "local"
_RUNTIME_REMOTE = "remote"
_RUNTIME_PLATFORM = "platform"
_RUNTIME_HYBRID = "hybrid"


@dataclass(frozen=True)
class PlanNode:
    """One executable node in a compiled workflow plan."""

    node_id: str
    op: str
    runtime: str
    config: dict[str, Any]
    depends_on: tuple[str, ...] = ()


@dataclass
class WorkflowPlan:
    """Compiled DAG ready for the runner."""

    workflow_id: str
    nodes: dict[str, PlanNode] = field(default_factory=dict)
    execution_order: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    workflow_hash: str = ""


class Workflow:
    """Workflow asset loaded from YAML."""

    def __init__(self, cfg: DictConfig, *, config_path: Path) -> None:
        self.cfg = cfg
        self.config_path = Path(config_path)
        self.id = str(cfg.get("id") or self.config_path.stem)

    def compile(self, *, run_inputs: dict[str, Any] | None = None) -> WorkflowPlan:
        """Validate the workflow graph and produce an execution plan."""
        payload = OmegaConf.to_container(self.cfg, resolve=True) or {}
        if not isinstance(payload, dict):
            raise WorkflowError("Workflow config must be a mapping")

        input_spec = _coerce_mapping(payload.get("inputs", {}))
        defaults = {
            key: spec.get("default")
            for key, spec in input_spec.items()
            if isinstance(spec, dict) and "default" in spec
        }
        merged_inputs = {**defaults, **(run_inputs or {})}
        _require_inputs(input_spec, merged_inputs)

        nodes_spec = _coerce_mapping(payload.get("nodes", {}))
        if not nodes_spec:
            raise WorkflowError("Workflow requires a non-empty 'nodes' mapping")

        plan_nodes: dict[str, PlanNode] = {}
        for node_id, node_cfg in nodes_spec.items():
            _register_node(plan_nodes, str(node_id), _coerce_mapping(node_cfg))

        edges = payload.get("edges") or []
        explicit_deps = _dependencies_from_edges(edges)
        ref_deps = _dependencies_from_refs(nodes_spec)
        for node_id, node in plan_nodes.items():
            deps = set(explicit_deps.get(node_id, ()))
            deps.update(ref_deps.get(node_id, ()))
            plan_nodes[node_id] = PlanNode(
                node_id=node.node_id,
                op=node.op,
                runtime=node.runtime,
                config=node.config,
                depends_on=tuple(sorted(deps)),
            )

        order = _topological_sort(plan_nodes)
        outputs = _parse_outputs(_coerce_mapping(payload.get("outputs", {})))
        return WorkflowPlan(
            workflow_id=self.id,
            nodes=plan_nodes,
            execution_order=order,
            outputs=outputs,
            inputs=merged_inputs,
            workflow_hash=config_hash(payload),
        )


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise WorkflowError(f"Expected mapping, got {type(value).__name__}")


def _require_inputs(input_spec: dict[str, Any], values: dict[str, Any]) -> None:
    for name, spec in input_spec.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("required", False) and name not in values:
            raise WorkflowError(f"Missing required workflow input '{name}'")


def _register_node(
    plan_nodes: dict[str, PlanNode],
    node_id: str,
    node_cfg: dict[str, Any],
) -> None:
    if node_id in plan_nodes:
        raise WorkflowError(f"Duplicate workflow node id '{node_id}'")
    op = str(node_cfg.get("op") or "")
    if not op:
        raise WorkflowError(f"Node '{node_id}' requires an 'op' field")
    runtime = str(node_cfg.get("runtime") or _RUNTIME_LOCAL)
    plan_nodes[node_id] = PlanNode(
        node_id=node_id,
        op=op,
        runtime=runtime,
        config=node_cfg,
    )


def _dependencies_from_edges(edges: Any) -> dict[str, set[str]]:
    deps: dict[str, set[str]] = {}
    if not isinstance(edges, list):
        return deps
    for edge in edges:
        if not isinstance(edge, dict):
            raise WorkflowError(f"Invalid edge entry: {edge!r}")
        target = edge.get("to")
        source = edge.get("from")
        if isinstance(target, str):
            target_node = target.split(".", 1)[0]
        elif isinstance(target, list) and target:
            target_node = str(target[0])
        else:
            raise WorkflowError(f"Edge target must name a node: {edge!r}")
        if isinstance(source, str):
            source_node = source.split(".", 1)[0]
        elif isinstance(source, list) and source:
            source_node = str(source[0])
        else:
            raise WorkflowError(f"Edge source must name a node: {edge!r}")
        deps.setdefault(target_node, set()).add(source_node)
    return deps


def _iter_refs(value: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(value, str) and value.startswith("$"):
        refs.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            refs.extend(_iter_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.extend(_iter_refs(item))
    return refs


def _dependencies_from_refs(nodes_spec: dict[str, Any]) -> dict[str, set[str]]:
    deps: dict[str, set[str]] = {}
    for node_id, node_cfg in nodes_spec.items():
        refs = _iter_refs(node_cfg)
        for ref in refs:
            parsed = parse_ref(ref)
            if parsed["kind"] == "nodes":
                deps.setdefault(str(node_id), set()).add(parsed["node_id"])
    return deps


def _topological_sort(plan_nodes: dict[str, PlanNode]) -> list[str]:
    remaining = {node_id: set(node.depends_on) for node_id, node in plan_nodes.items()}
    order: list[str] = []
    while remaining:
        ready = [node_id for node_id, deps in remaining.items() if not deps]
        if not ready:
            raise WorkflowError("Workflow graph contains a cycle")
        for node_id in sorted(ready):
            order.append(node_id)
            del remaining[node_id]
        for deps in remaining.values():
            deps.difference_update(ready)
    return order


def _parse_outputs(outputs_spec: dict[str, Any]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for name, spec in outputs_spec.items():
        if isinstance(spec, dict) and "from" in spec:
            outputs[str(name)] = str(spec["from"])
        elif isinstance(spec, str):
            outputs[str(name)] = spec
        else:
            raise WorkflowError(f"Invalid workflow output '{name}': {spec!r}")
    return outputs


def parse_ref(ref: str) -> dict[str, str]:
    """Parse ``$inputs.x`` or ``$nodes.node.outputs.port`` references."""
    match = _REF_PATTERN.match(ref)
    if not match:
        raise WorkflowError(f"Invalid workflow reference '{ref}'")
    kind, path, port = match.groups()
    if kind == "inputs":
        return {"kind": "inputs", "name": path, "port": port or ""}
    parts = path.split(".", 1)
    node_id = parts[0]
    suffix = parts[1] if len(parts) > 1 else ""
    if suffix.startswith("outputs."):
        port = suffix.split(".", 1)[1]
    elif suffix and not port:
        port = suffix
    return {"kind": "nodes", "node_id": node_id, "port": port or ""}


NodeOutputs = dict[str, dict[str, Any]]


def resolve_value(
    value: Any,
    *,
    inputs: dict[str, Any],
    node_outputs: NodeOutputs,
) -> Any:
    """Resolve expressions embedded in node configuration values."""
    if isinstance(value, str) and value.startswith("$"):
        parsed = parse_ref(value)
        if parsed["kind"] == "inputs":
            if parsed["name"] not in inputs:
                raise WorkflowError(f"Unknown workflow input '{parsed['name']}'")
            resolved = inputs[parsed["name"]]
            if parsed["port"]:
                raise WorkflowError(f"Workflow inputs do not expose ports: '{value}'")
            return resolved
        node_id = parsed["node_id"]
        if node_id not in node_outputs:
            raise WorkflowError(f"Node '{node_id}' has no outputs yet (reference '{value}')")
        outputs = node_outputs[node_id]
        port = parsed["port"] or next(iter(outputs))
        if port not in outputs:
            raise WorkflowError(f"Node '{node_id}' output '{port}' is unavailable for '{value}'")
        return outputs[port]
    if isinstance(value, dict):
        return {
            key: resolve_value(item, inputs=inputs, node_outputs=node_outputs)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [resolve_value(item, inputs=inputs, node_outputs=node_outputs) for item in value]
    return value


def resolve_output_ref(
    ref: str,
    *,
    node_outputs: NodeOutputs,
    inputs: dict[str, Any],
) -> Any:
    """Resolve a workflow-level output reference."""
    if ref.startswith("$"):
        return resolve_value(ref, inputs=inputs, node_outputs=node_outputs)
    if "." in ref:
        node_id, port = ref.split(".", 1)
        if node_id in node_outputs and port in node_outputs[node_id]:
            return node_outputs[node_id][port]
    raise WorkflowError(f"Cannot resolve workflow output reference '{ref}'")
