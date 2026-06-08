from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import grumpy as gr
from grumpy import GrumpyArray

from fabric.core.backend import Backend
from fabric.core.collater import CollatedBatch
from fabric.utils.errors import BackendError, BackendExtraRequired


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise BackendExtraRequired("torch") from exc
    return torch


def _require_model_optimizer(model: Any) -> Any:
    optimizer = getattr(model, "optimizer", None)
    if optimizer is None:
        raise BackendError(
            "TorchBackend requires model.optimizer; attach an optimizer to the model "
            "before calling Trainer.fit()."
        )
    return optimizer


class TorchBackend(Backend):
    """PyTorch backend with optional Lightning Fabric device management.

    Args:
        accelerator: ``cpu``, ``cuda``, or ``gpu``.
        devices: Number of devices (Fabric-managed when Lightning is installed).
        precision: Fabric precision setting.
    """

    name = "TorchBackend"

    def __init__(
        self,
        *,
        accelerator: str = "cpu",
        devices: int = 1,
        precision: str | int = 32,
    ) -> None:
        self.accelerator = accelerator
        self.devices = int(devices)
        self.precision = precision
        self._torch = None
        self._fabric = None
        self._device = None

    def _init_runtime(self) -> None:
        if self._torch is not None:
            return
        torch = _require_torch()
        accelerator = self.accelerator
        if accelerator == "gpu":
            accelerator = "cuda"
        if accelerator == "cuda" and not torch.cuda.is_available():
            raise BackendError(
                "TorchBackend requested accelerator='cuda' but CUDA is unavailable; "
                "use accelerator='cpu' or install a CUDA-enabled PyTorch build."
            )
        self._torch = torch
        try:
            from lightning.fabric import Fabric
        except ImportError:
            self._fabric = None
            self._device = torch.device("cuda" if accelerator == "cuda" else "cpu")
            return
        self._fabric = Fabric(
            accelerator=accelerator,
            devices=self.devices,
            precision=self.precision,
        )
        self._device = self._fabric.device

    @property
    def device(self):
        self._init_runtime()
        return self._device

    def setup(self, model: Any) -> Any:
        """Place the model and ``model.optimizer`` on the selected device."""
        _require_torch()
        self._init_runtime()
        optimizer = _require_model_optimizer(model)
        if self._fabric is not None:
            model, optimizer = self._fabric.setup(model, optimizer)
            model.optimizer = optimizer
            return model
        model = model.to(self.device)
        return model

    def to_tensor(self, array: GrumpyArray) -> Any:
        self._init_runtime()
        return array.to_torch().to(self.device)

    def to_grumpy(self, tensor: Any) -> GrumpyArray:
        return gr.from_torch(tensor.detach().cpu(), dtype=gr.float32)

    def _predict(self, model: Any, features: Any) -> Any:
        output = model(features)
        if output.ndim > 1:
            output = output.reshape(output.shape[0], -1)[:, 0]
        return output

    def train_step(self, model: Any, batch: CollatedBatch) -> float:
        torch = _require_torch()
        self._init_runtime()
        optimizer = _require_model_optimizer(model)
        model.train()
        features = self.to_tensor(batch.features)
        targets = self.to_tensor(batch.y)
        optimizer.zero_grad(set_to_none=True)
        predictions = self._predict(model, features)
        loss = torch.mean((predictions - targets) ** 2)
        if self._fabric is not None:
            self._fabric.backward(loss)
        else:
            loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu().item())
        if math.isnan(value):
            raise BackendError("Training step produced NaN loss")
        return value

    def eval_step(self, model: Any, batch: CollatedBatch) -> tuple[float, GrumpyArray]:
        torch = _require_torch()
        self._init_runtime()
        model.eval()
        with torch.no_grad():
            features = self.to_tensor(batch.features)
            targets = self.to_tensor(batch.y)
            predictions = self._predict(model, features)
            loss = torch.mean((predictions - targets) ** 2)
        return float(loss.detach().cpu().item()), self.to_grumpy(predictions)

    def save_checkpoint(
        self,
        path: str,
        *,
        model: Any,
        epoch: int,
        step: int,
    ) -> None:
        torch = _require_torch()
        self._init_runtime()
        optimizer = _require_model_optimizer(model)
        checkpoint = Path(path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
        }
        if self._fabric is not None:
            self._fabric.save(checkpoint.with_suffix(".pt"), payload)
            return
        torch.save(payload, checkpoint.with_suffix(".pt"))

    def load_checkpoint(self, path: str, *, model: Any) -> dict[str, int]:
        torch = _require_torch()
        self._init_runtime()
        optimizer = _require_model_optimizer(model)
        checkpoint = Path(path).with_suffix(".pt")
        if self._fabric is not None:
            payload = self._fabric.load(checkpoint)
        else:
            payload = torch.load(checkpoint, map_location=self.device)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        return {"epoch": int(payload["epoch"]), "step": int(payload["step"])}
