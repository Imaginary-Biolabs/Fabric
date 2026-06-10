"""Multi-layer perceptron scaffold for flat feature batches.

Builds a sequential PyTorch network on the ``features`` collater slot.
"""

from typing import Any

from fabric.core.layers import build_layer
from fabric.core.scaffold import Scaffold
from fabric.utils.errors import ModelError


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ModelError(
            "Scaffolds require PyTorch: pip install 'imaginary-fabric[torch]'"
        ) from exc
    return torch


class _SlotMLP:
    """Run a sequential network on the ``features`` slot."""

    def __init__(self, network: Any) -> None:
        self.network = network

    def __call__(self, slots: dict[str, Any]) -> dict[str, Any]:
        output = self.network(slots["features"])
        if output.ndim > 1 and output.shape[-1] == 1:
            output = output.squeeze(-1)
        return {"predictions": output}


class MLPScaffold(Scaffold):
    """Multi-layer perceptron on flat ``features`` batches.

    Args:
        hidden: Hidden layer widths when ``layers`` is omitted.
        layers: Optional explicit layer stack from the layer registry.
        output_dim: Output width for the final prediction head.

    Example:
        >>> scaffold = MLPScaffold(hidden=[32, 16])
        >>> scaffold.build(input_dim=8)  # doctest: +SKIP
    """

    name = "MLPScaffold"

    def __init__(
        self,
        *,
        hidden: list[int] | None = None,
        layers: list[dict[str, Any]] | None = None,
        output_dim: int = 1,
    ) -> None:
        self.hidden = [int(width) for width in (hidden or [])]
        self.layers = list(layers or [])
        self.output_dim = int(output_dim)
        if not self.layers and not self.hidden:
            raise ModelError("MLPScaffold requires 'hidden' or 'layers'")

    def build(self, *, input_dim: int, output_dim: int = 1) -> Any:
        torch = _require_torch()
        out_dim = self.output_dim if output_dim == 1 else int(output_dim)
        modules: list[Any] = []

        if self.layers:
            current_dim = int(input_dim)
            for entry in self.layers:
                if not isinstance(entry, dict) or len(entry) != 1:
                    raise ModelError(f"Layer entry must be a single-key mapping: {entry!r}")
                name, params = next(iter(entry.items()))
                if str(name) == "Linear":
                    params = dict(params or {})
                    params.setdefault("in_features", current_dim)
                    if "out_features" not in params:
                        raise ModelError("Linear layer requires out_features")
                    current_dim = int(params["out_features"])
                module = build_layer(str(name), params)
                modules.append(module)
            if not isinstance(modules[-1], torch.nn.Linear):
                modules.append(torch.nn.Linear(current_dim, out_dim))
        else:
            dims = [int(input_dim), *self.hidden, out_dim]
            for in_features, out_features in zip(dims, dims[1:]):
                modules.append(torch.nn.Linear(in_features, out_features))
                if out_features != out_dim:
                    modules.append(torch.nn.ReLU())

        return _SlotMLP(torch.nn.Sequential(*modules))
