"""Model registry and YAML factory helpers."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from fabric.core.collater import Collater
from fabric.core.model import Model, validate_collater_spec
from fabric.core.objectives import OBJECTIVE_REGISTRY
from fabric.core.scaffold import CollaterSpec
from fabric.core.scaffolds import SCAFFOLD_REGISTRY
from fabric.utils.errors import ModelError


def _single_entry(
    config: Any,
    *,
    field: str,
    registry: dict[str, type],
) -> tuple[str, dict[str, Any]]:
    if not isinstance(config, (dict, DictConfig)):
        raise ModelError(f"Model config requires a '{field}' mapping")
    items = list(config.items())
    if len(items) != 1:
        raise ModelError(f"Model '{field}' must contain exactly one key")
    name, params = items[0]
    cls = registry.get(str(name))
    if cls is None:
        available = ", ".join(sorted(registry))
        raise ModelError(f"Unknown {field} '{name}'. Available: {available}")
    if isinstance(params, dict):
        kwargs = params
    elif params is not None:
        kwargs = OmegaConf.to_container(params, resolve=True) or {}
    else:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise ModelError(f"{field} '{name}' params must be a mapping")
    return str(name), kwargs


def _input_spec(config: Any) -> CollaterSpec:
    if config is None:
        return CollaterSpec(layout="flat", slots=("features",))
    if not isinstance(config, (dict, DictConfig)):
        raise ModelError("Model 'input' must be a mapping")
    layout = str(config.get("layout", "flat"))
    slots = config.get("slots", ["features"])
    if isinstance(slots, (list, tuple)):
        slot_tuple = tuple(str(slot) for slot in slots)
    else:
        raise ModelError("Model 'input.slots' must be a list of slot names")
    return CollaterSpec(layout=layout, slots=slot_tuple)


def build_model(
    config: dict[str, Any] | DictConfig,
    *,
    collater: Collater | None = None,
    input_dim: int | None = None,
) -> Model:
    """Instantiate a Fabric model from YAML config."""
    cfg = OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else config
    if not isinstance(cfg, dict):
        raise ModelError("Model config must be a mapping")

    input_spec = _input_spec(cfg.get("input"))
    if collater is not None:
        validate_collater_spec(input_spec, collater)

    scaffold_name, scaffold_kwargs = _single_entry(
        cfg.get("scaffold"),
        field="scaffold",
        registry=SCAFFOLD_REGISTRY,
    )
    objective_name, objective_kwargs = _single_entry(
        cfg.get("objective", {"SupervisedObjective": {}}),
        field="objective",
        registry=OBJECTIVE_REGISTRY,
    )

    resolved_input_dim = input_dim
    if resolved_input_dim is None:
        resolved_input_dim = scaffold_kwargs.get("input_dim")
    if resolved_input_dim is None and collater is not None and hasattr(collater, "features"):
        resolved_input_dim = len(collater.features)
    if resolved_input_dim is None:
        raise ModelError(
            "Model input_dim is required when no collater with features is provided"
        )

    output_dim = int(scaffold_kwargs.get("output_dim", 1))
    scaffold_cls = SCAFFOLD_REGISTRY[scaffold_name]
    scaffold = scaffold_cls(**scaffold_kwargs)
    module = scaffold.build(input_dim=int(resolved_input_dim), output_dim=output_dim)
    objective = OBJECTIVE_REGISTRY[objective_name](**objective_kwargs)

    model_id = str(cfg.get("id", scaffold_name))
    model = Model(
        module=module,
        objective=objective,
        input_spec=input_spec,
        input_dim=int(resolved_input_dim),
        output_dim=output_dim,
    )
    model.name = model_id
    return model


def _build_adam(model: Model, params: dict[str, Any]) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ModelError(
            "Optimizers require PyTorch: pip install 'imaginary-fabric[torch]'"
        ) from exc
    lr = float(params.get("lr", 1e-3))
    return torch.optim.Adam(model.parameters, lr=lr)


_OPTIMIZER_BUILDERS: dict[str, Any] = {
    "Adam": _build_adam,
}


def attach_optimizer(model: Model, config: Any) -> None:
    """Attach an optimizer from a single-key YAML mapping."""
    if config is None:
        return
    if not isinstance(config, (dict, DictConfig)):
        raise ModelError("Model 'optimizer' must be a mapping")
    items = list(config.items())
    if len(items) != 1:
        raise ModelError("Model 'optimizer' must contain exactly one key")
    name, params = items[0]
    builder = _OPTIMIZER_BUILDERS.get(str(name))
    if builder is None:
        available = ", ".join(sorted(_OPTIMIZER_BUILDERS))
        raise ModelError(f"Unknown optimizer '{name}'. Available: {available}")
    if isinstance(params, dict):
        kwargs = params
    elif params is not None:
        kwargs = OmegaConf.to_container(params, resolve=True) or {}
    else:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise ModelError(f"Optimizer '{name}' params must be a mapping")
    model.optimizer = builder(model, kwargs)
