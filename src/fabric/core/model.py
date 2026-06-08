"""Config-driven Fabric model with slot-based training steps."""

from __future__ import annotations

import math
from typing import Any

from fabric.core.collater import CollatedBatch, Collater
from fabric.core.objective import Objective
from fabric.core.scaffold import CollaterSpec, StepResult
from fabric.utils.errors import BackendError, ModelError


def validate_collater_spec(model_spec: CollaterSpec, collater: Collater) -> None:
    """Ensure a collater satisfies a model input contract."""
    collater_spec = collater.spec
    if model_spec.layout != collater_spec.layout:
        raise ModelError(
            f"Model expects collater layout '{model_spec.layout}', "
            f"but got '{collater_spec.layout}' from {collater.name}"
        )
    for slot in model_spec.slots:
        if slot not in collater_spec.slots:
            raise ModelError(
                f"Model requires slot '{slot}', but {collater.name} provides "
                f"{list(collater_spec.slots)}"
            )


class Model:
    """Fabric model binding a scaffold, objective, and training semantics.

    The user attaches ``model.optimizer`` before calling :meth:`Trainer.fit`.
    """

    name: str = "Model"

    def __init__(
        self,
        *,
        module: Any,
        objective: Objective,
        input_spec: CollaterSpec,
        input_dim: int,
        output_dim: int = 1,
        optimizer: Any = None,
    ) -> None:
        self.module = module
        self.objective = objective
        self.input_spec = input_spec
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.optimizer = optimizer

    @property
    def parameters(self):
        if hasattr(self.module, "parameters"):
            return self.module.parameters()
        if hasattr(self.module, "network"):
            return self.module.network.parameters()
        raise ModelError("Model module does not expose trainable parameters")

    def train(self) -> None:
        network = getattr(self.module, "network", self.module)
        if hasattr(network, "train"):
            network.train()

    def eval(self) -> None:
        network = getattr(self.module, "network", self.module)
        if hasattr(network, "eval"):
            network.eval()

    def state_dict(self) -> dict[str, Any]:
        network = getattr(self.module, "network", self.module)
        if hasattr(network, "state_dict"):
            return network.state_dict()
        raise ModelError("Model module does not support state_dict()")

    def load_state_dict(self, state: dict[str, Any]) -> None:
        network = getattr(self.module, "network", self.module)
        if hasattr(network, "load_state_dict"):
            network.load_state_dict(state)
            return
        raise ModelError("Model module does not support load_state_dict()")

    def to(self, device: Any) -> Model:
        network = getattr(self.module, "network", self.module)
        if hasattr(network, "to"):
            network.to(device)
        return self

    def forward(self, batch: CollatedBatch, *, backend: Any) -> dict[str, Any]:
        """Run the scaffold module on collated batch slots."""
        tensors = backend.batch_tensors(batch)
        if hasattr(self.module, "__call__") and not hasattr(self.module, "forward"):
            return self.module(tensors)
        return self.module.forward(tensors)

    def training_step(self, batch: CollatedBatch, *, backend: Any) -> StepResult:
        """Compute loss, backpropagate, and return predictions."""
        self.train()
        backend.zero_grad(self)
        outputs = self.forward(batch, backend=backend)
        tensors = backend.batch_tensors(batch)
        losses = self.objective.loss(outputs, batch, tensors=tensors, backend=backend)
        loss = losses["loss"]
        backend.backward(loss)
        backend.step(self)
        value = backend.loss_value(loss)
        if math.isnan(value):
            raise BackendError("Training step produced NaN loss")
        predictions = backend.to_grumpy(outputs["predictions"])
        logs = {
            key: backend.loss_value(tensor)
            for key, tensor in losses.items()
            if key != "loss"
        }
        return StepResult(loss=value, predictions=predictions, logs=logs)

    def validation_step(self, batch: CollatedBatch, *, backend: Any) -> StepResult:
        """Evaluate without parameter updates."""
        self.eval()
        outputs = self.forward(batch, backend=backend)
        tensors = backend.batch_tensors(batch)
        with backend.no_grad():
            losses = self.objective.loss(outputs, batch, tensors=tensors, backend=backend)
        value = backend.loss_value(losses["loss"])
        predictions = backend.to_grumpy(outputs["predictions"])
        return StepResult(loss=value, predictions=predictions)

    def hash_params(self) -> dict[str, Any]:
        """Return constructor parameters for provenance hashing."""
        return {
            "input_spec": {
                "layout": self.input_spec.layout,
                "slots": list(self.input_spec.slots),
            },
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "objective": self.objective.name,
        }
