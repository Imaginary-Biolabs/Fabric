from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fabric.core.backend import Backend
from fabric.core.collater import CollatedBatch
from fabric.utils.errors import BackendError, BackendExtraRequired


def _require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise BackendExtraRequired("tensorflow") from exc
    return tf


def _require_model_optimizer(model: Any) -> Any:
    optimizer = getattr(model, "optimizer", None)
    if optimizer is None:
        raise BackendError(
            "TensorflowBackend requires model.optimizer; attach an optimizer to the model "
            "before calling Trainer.fit()."
        )
    return optimizer


class TensorflowBackend(Backend):
    """TensorFlow backend using ``GradientTape`` for training steps.

    Args:
        accelerator: ``cpu`` or ``gpu``.
    """

    name = "TensorflowBackend"

    def __init__(
        self,
        *,
        accelerator: str = "cpu",
    ) -> None:
        if accelerator not in {"cpu", "gpu", "cuda"}:
            raise BackendError(
                f"TensorflowBackend supports accelerator='cpu' or 'gpu'; got {accelerator!r}"
            )
        if accelerator in {"gpu", "cuda"}:
            tf = _require_tensorflow()
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                raise BackendError(
                    "TensorflowBackend requested accelerator='gpu' but no GPU is available; "
                    "use accelerator='cpu'."
                )
        self.accelerator = "gpu" if accelerator in {"gpu", "cuda"} else "cpu"
        self._tf = None

    def _init_runtime(self) -> None:
        if self._tf is not None:
            return
        self._tf = _require_tensorflow()

    def setup(self, model: Any) -> Any:
        """Validate that the model exposes a user-configured optimizer."""
        _require_tensorflow()
        self._init_runtime()
        _require_model_optimizer(model)
        return model

    def to_tensor(self, array: np.ndarray) -> Any:
        tf = _require_tensorflow()
        self._init_runtime()
        return tf.convert_to_tensor(array, dtype=tf.float32)

    def to_numpy(self, tensor: Any) -> np.ndarray:
        self._init_runtime()
        return np.asarray(tensor.numpy(), dtype=np.float32)

    def _predict(self, model: Any, features: Any) -> Any:
        tf = self._tf
        output = model(features, training=False)
        return tf.reshape(output, (-1,))

    def train_step(self, model: Any, batch: CollatedBatch) -> float:
        tf = _require_tensorflow()
        self._init_runtime()
        optimizer = _require_model_optimizer(model)
        features = self.to_tensor(batch.features)
        targets = self.to_tensor(batch.y)
        with tf.GradientTape() as tape:
            predictions = self._predict(model, features)
            loss = tf.reduce_mean(tf.square(predictions - targets))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        value = float(loss.numpy())
        if np.isnan(value):
            raise BackendError("Training step produced NaN loss")
        return value

    def eval_step(self, model: Any, batch: CollatedBatch) -> tuple[float, np.ndarray]:
        tf = _require_tensorflow()
        self._init_runtime()
        features = self.to_tensor(batch.features)
        targets = self.to_tensor(batch.y)
        predictions = self._predict(model, features)
        loss = tf.reduce_mean(tf.square(predictions - targets))
        return float(loss.numpy()), self.to_numpy(predictions)

    def save_checkpoint(
        self,
        path: str,
        *,
        model: Any,
        epoch: int,
        step: int,
    ) -> None:
        self._init_runtime()
        optimizer = _require_model_optimizer(model)
        checkpoint = Path(path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(checkpoint.with_suffix(".weights.h5"))
        np.savez(
            checkpoint.with_suffix(".meta.npz"),
            epoch=epoch,
            step=step,
            optimizer=np.asarray(optimizer.get_weights(), dtype=object),
        )

    def load_checkpoint(self, path: str, *, model: Any) -> dict[str, int]:
        self._init_runtime()
        optimizer = _require_model_optimizer(model)
        checkpoint = Path(path)
        model.load_weights(checkpoint.with_suffix(".weights.h5"))
        payload = np.load(checkpoint.with_suffix(".meta.npz"), allow_pickle=True)
        optimizer.set_weights(list(payload["optimizer"]))
        return {"epoch": int(payload["epoch"]), "step": int(payload["step"])}
