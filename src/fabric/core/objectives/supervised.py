"""Supervised regression objectives for Fabric models.

Compute backend-specific losses between model predictions and batch targets.
"""

from typing import Any

from fabric.core.collater import CollatedBatch
from fabric.core.objective import Objective
from fabric.utils.errors import ModelError


class SupervisedObjective(Objective):
    """Regression loss between ``predictions`` and batch targets.

    Args:
        loss: Supported value is ``'mse'``.

    Example:
        >>> objective = SupervisedObjective(loss="mse")
        >>> objective.loss_name
        'mse'
    """

    name = "SupervisedObjective"

    def __init__(self, *, loss: str = "mse") -> None:
        self.loss_name = str(loss)

    def loss(
        self,
        outputs: dict[str, Any],
        batch: CollatedBatch,
        *,
        tensors: dict[str, Any],
        backend: Any,
    ) -> dict[str, Any]:
        if self.loss_name != "mse":
            raise ModelError(f"Unknown supervised loss '{self.loss_name}'")
        predictions = outputs["predictions"]
        targets = tensors["y"]
        return {"loss": backend.mse_loss(predictions, targets)}
