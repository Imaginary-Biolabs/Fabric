"""Backend registry."""

from fabric.core.backends.tensorflow import TensorflowBackend
from fabric.core.backends.torch import TorchBackend

__all__ = ["TensorflowBackend", "TorchBackend"]
