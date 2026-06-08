"""Phase 4 toll station — trainer, collaters, and backends."""

from __future__ import annotations

from pathlib import Path

import grumpy as gr
import numpy as np
import pytest

from fabric import Settings
from fabric.core.backends import TensorflowBackend, TorchBackend
from fabric.core.collaters import LongCollater, WideCollater
from fabric.core.factory import Factory
from fabric.core.loggers import DiskLogger
from fabric.core.result import Result
from fabric.core.trainer import Trainer
from fabric.utils.errors import BackendError, CollateError, TrainerError

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _feature_dim() -> int:
    return 2


def _torch_model(*, lr: float = 1e-3):
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(
        torch.nn.Linear(_feature_dim(), 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1),
    )
    model.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model


def _tensorflow_model(*, lr: float = 1e-3):
    tf = pytest.importorskip("tensorflow")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    return model


@pytest.fixture
def bench(tmp_path: Path):
    with Settings(home=tmp_path):
        yield Factory.benchmark("tests/fixtures/benchmarks/B_mini.yaml")


def test_collater_unknown_feature_raises(bench) -> None:
    collater = WideCollater(features=["not_a_real_feature"])
    X, y = next(iter(bench.train_loader(batch_size=2, progress=False)))
    with pytest.raises(CollateError, match="Unknown collater feature"):
        collater.collate(X, y)


def test_wide_collater_suggests_long_for_nested_feature(bench) -> None:
    collater = WideCollater(features=["atom_pos"])
    X, y = next(iter(bench.train_loader(batch_size=2, progress=False)))
    with pytest.raises(CollateError, match="use LongCollater"):
        collater.collate(X, y)


def test_torch_backend_cuda_unavailable_raises() -> None:
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this environment")
    with pytest.raises(BackendError, match="CUDA is unavailable"):
        TorchBackend(accelerator="cuda").setup(_torch_model())


def test_trainer_requires_model_optimizer(tmp_path: Path, bench) -> None:
    pytest.importorskip("torch")
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(_feature_dim(), 1))
    trainer = Trainer(
        model=model,
        benchmark=bench,
        backend=TorchBackend(accelerator="cpu"),
        collater=WideCollater(features=["residue_count", "atom_count"]),
        root=tmp_path / "no_optimizer",
        epochs=1,
        batch_size=4,
        progress=False,
    )
    with pytest.raises(TrainerError, match="optimizer"):
        trainer.fit()


def test_torch_trainer_fit_validate_test(tmp_path: Path, bench) -> None:
    pytest.importorskip("torch")
    root = tmp_path / "torch_run"
    trainer = Trainer(
        model=_torch_model(),
        benchmark=bench,
        backend=TorchBackend(accelerator="cpu"),
        collater=WideCollater(features=["residue_count", "atom_count"]),
        logger=DiskLogger(root),
        root=root,
        epochs=1,
        batch_size=4,
        progress=False,
    )
    trainer.fit()
    assert (root / "checkpoint.pt").exists()
    assert (root / "train_loss.csv").exists()
    val_metrics = trainer.validate()
    assert "MAE" in val_metrics
    assert np.isfinite(val_metrics["MAE"])
    test_metrics = trainer.test()
    result = Result(name="torch_mlp", metrics=test_metrics)
    assert np.isfinite(result.metrics["MAE"])


def test_tensorflow_trainer_fit(tmp_path: Path, bench) -> None:
    pytest.importorskip("tensorflow")
    root = tmp_path / "tf_run"
    model = _tensorflow_model()
    model.build(input_shape=(None, _feature_dim()))
    model.optimizer.build(model.trainable_variables)
    trainer = Trainer(
        model=model,
        benchmark=bench,
        backend=TensorflowBackend(accelerator="cpu"),
        collater=LongCollater(features=["residue_count", "atom_count"]),
        logger=DiskLogger(root),
        root=root,
        epochs=1,
        batch_size=4,
        progress=False,
    )
    trainer.fit()
    assert (root / "checkpoint.weights.h5").exists()
    metrics = trainer.test()
    assert np.isfinite(metrics["MAE"])


def test_long_collater_records_scene_index(bench) -> None:
    collater = LongCollater(features=["residue_count", "atom_count"])
    X, y = next(iter(bench.train_loader(batch_size=2, progress=False)))
    batch = collater.collate(X, y)
    assert batch.features.shape(0) == 2
    assert gr.is_rectangular(batch.features)
    assert batch.features.shape(1).to_list() == [2, 2]
    assert batch.scene_index is not None
    assert batch.scene_index.to_list() == [0, 1]
