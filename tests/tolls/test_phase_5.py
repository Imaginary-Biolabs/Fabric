"""Phase 5 toll station — layers, scaffolds, models, and factory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fabric import Settings
from fabric.core.backends import TorchBackend
from fabric.core.collaters import LongCollater, WideCollater
from fabric.core.factory import Factory
from fabric.core.layers import LAYER_REGISTRY, build_layer
from fabric.core.model import Model
from fabric.core.models import build_model
from fabric.core.trainer import Trainer
from fabric.utils.errors import ModelError

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture
def bench(tmp_path: Path):
    with Settings(home=tmp_path):
        yield Factory.benchmark("tests/fixtures/benchmarks/B_mini.yaml")


def test_unknown_layer_registry_key_raises() -> None:
    with pytest.raises(ModelError, match="Unknown layer"):
        build_layer("NotALayer", {})


def test_model_collater_layout_mismatch_raises(bench) -> None:
    collater = LongCollater(features=["residue_count", "atom_count"])
    with pytest.raises(ModelError, match="layout"):
        Factory.model("tests/fixtures/models/M_mini.yaml", collater=collater)


def test_factory_model_builds_mlp(bench) -> None:
    pytest.importorskip("torch")
    collater = WideCollater(features=["residue_count", "atom_count"])
    model = Factory.model("tests/fixtures/models/M_mini.yaml", collater=collater)
    assert isinstance(model, Model)
    assert model.input_dim == 2
    assert model.optimizer is not None
    assert model.input_spec.layout == "flat"


def test_model_training_step_on_batch(bench) -> None:
    pytest.importorskip("torch")
    collater = WideCollater(features=["residue_count", "atom_count"])
    model = Factory.model("tests/fixtures/models/M_mini.yaml", collater=collater)
    backend = TorchBackend(accelerator="cpu")
    backend.setup(model)
    X, y = next(iter(bench.train_loader(batch_size=2, progress=False)))
    batch = collater.collate(X, y)
    result = model.training_step(batch, backend=backend)
    assert np.isfinite(result.loss)
    assert result.predictions is not None


def test_factory_trainer_fit_validate_checkpoint(tmp_path: Path, bench) -> None:
    pytest.importorskip("torch")
    root = tmp_path / "phase5_run"
    trainer = Factory.trainer(
        "tests/fixtures/train/train_mini.yaml",
        root=str(root),
        progress=False,
    )
    trainer.fit()
    assert (root / "checkpoint.pt").exists()
    metrics = trainer.validate()
    assert "MAE" in metrics
    assert np.isfinite(metrics["MAE"])


def test_checkpoint_round_trip(tmp_path: Path, bench) -> None:
    pytest.importorskip("torch")
    root = tmp_path / "ckpt_run"
    collater = WideCollater(features=["residue_count", "atom_count"])
    model = Factory.model("tests/fixtures/models/M_mini.yaml", collater=collater)
    backend = TorchBackend(accelerator="cpu")
    trainer = Trainer(
        model=model,
        benchmark=bench,
        backend=backend,
        collater=collater,
        root=root,
        epochs=1,
        batch_size=4,
        progress=False,
    )
    trainer.fit()
    payload = backend.load_checkpoint(str(root / "checkpoint"), model=trainer.model)
    assert payload["epoch"] == 1
    assert payload["step"] > 0


def test_layer_registry_lists_available_layers() -> None:
    assert "Linear" in LAYER_REGISTRY
    assert "ReLU" in LAYER_REGISTRY


def test_build_model_with_explicit_layer_stack() -> None:
    pytest.importorskip("torch")
    cfg = {
        "scaffold": {
            "MLPScaffold": {
                "layers": [
                    {"Linear": {"out_features": 4}},
                    {"ReLU": {}},
                    {"Linear": {"out_features": 1}},
                ]
            }
        },
        "objective": {"SupervisedObjective": {}},
    }
    model = build_model(cfg, input_dim=2)
    assert model.input_dim == 2
