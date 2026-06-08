"""Fabric trainer orchestrating benchmark loaders, collaters, and backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import grumpy as gr

from fabric.core.backend import Backend
from fabric.core.benchmark import Benchmark
from fabric.core.collater import Collater
from fabric.core.logger import Logger
from fabric.core.loggers import DiskLogger
from fabric.utils.errors import TrainerError


class Trainer:
    """Run fit/validate/test loops for a benchmark and model.

    The model must expose ``model.optimizer``, configured by the user before
    training starts.

    Args:
        model: Framework model with an ``optimizer`` attribute.
        benchmark: Configured :class:`~fabric.core.benchmark.Benchmark`.
        backend: Deep learning backend implementation.
        collater: Collater applied to each loader batch.
        logger: Experiment logger.
        root: Directory for checkpoints and logs.
        epochs: Number of training epochs.
        batch_size: Loader batch size.
        validate_every: Run validation every N epochs (0 disables validation).
        checkpoint_name: Checkpoint filename stem written under ``root``.
        train_split: Dataset split name for training.
        val_split: Dataset split name for validation.
        test_split: Dataset split name for testing.
    """

    def __init__(
        self,
        *,
        model: Any,
        benchmark: Benchmark,
        backend: Backend,
        collater: Collater,
        logger: Logger | None = None,
        root: str | Path = "results",
        epochs: int = 1,
        batch_size: int = 8,
        validate_every: int = 1,
        checkpoint_name: str = "checkpoint",
        train_split: str = "train",
        val_split: str = "val",
        test_split: str = "test",
        progress: bool = False,
    ) -> None:
        self.model = model
        self.benchmark = benchmark
        self.backend = backend
        self.collater = collater
        self.logger = logger or DiskLogger(root)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.validate_every = int(validate_every)
        self.checkpoint_name = str(checkpoint_name)
        self.train_split = str(train_split)
        self.val_split = str(val_split)
        self.test_split = str(test_split)
        self.progress = bool(progress)
        self.epoch = 0
        self.step = 0

    def _require_optimizer(self) -> None:
        if getattr(self.model, "optimizer", None) is None:
            raise TrainerError(
                "Model must define an 'optimizer' attribute before calling Trainer.fit()."
            )

    def _split_loader(self, split: str):
        loader_name = f"{split}_loader"
        try:
            loader = getattr(self.benchmark, loader_name)
        except AttributeError as exc:
            raise TrainerError(
                f"Benchmark split '{split}' is not available for loader '{loader_name}'"
            ) from exc
        return loader(batch_size=self.batch_size, progress=self.progress)

    def _run_loader(self, split: str, *, train: bool) -> float | None:
        last_loss: float | None = None
        for X, y in self._split_loader(split):
            batch = self.collater.collate(X, y)
            if train:
                last_loss = self.backend.train_step(self.model, batch)
                self.step += 1
                self.logger.step = self.step
                self.logger.log_loss(last_loss, mode="train")
            else:
                loss, predictions = self.backend.eval_step(self.model, batch)
                self.benchmark.update(predictions.astype(gr.float64), y)
                last_loss = loss
        return last_loss

    def fit(self) -> None:
        """Train for :attr:`epochs` and optionally validate/checkpoint."""
        self._require_optimizer()
        self.model = self.backend.setup(self.model)
        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            self.logger.epoch = epoch
            self._run_loader(self.train_split, train=True)
            if self.validate_every and epoch % self.validate_every == 0:
                self.validate()
            self.backend.save_checkpoint(
                str(self.root / self.checkpoint_name),
                model=self.model,
                epoch=self.epoch,
                step=self.step,
            )

    def validate(self) -> dict[str, float | list[Any]]:
        """Evaluate on the validation split and return benchmark metrics."""
        self.benchmark.metrics.reset()
        self._run_loader(self.val_split, train=False)
        metrics = self.benchmark.metrics.compute_all()
        self.logger.log_metrics(metrics, mode="val")
        return metrics

    def test(self) -> dict[str, float | list[Any]]:
        """Evaluate on the test split and return benchmark metrics."""
        self.benchmark.metrics.reset()
        self._run_loader(self.test_split, train=False)
        metrics = self.benchmark.metrics.compute_all()
        self.logger.log_metrics(metrics, mode="test")
        return metrics
