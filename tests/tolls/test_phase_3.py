"""Phase 3 toll station — benchmark loaders, tasks, and metrics."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import grumpy as gr
import pytest

from fabric import Settings
from fabric.core.factory import Factory
from fabric.core.metrics import MAE
from fabric.core.result import Result
from fabric.utils.errors import BenchmarkError, MetricError, TaskError
from fabric.utils.io import loader_description


def _collect_indices(bench, loader_name: str, *, batch_size: int = 4) -> set[int]:
    """Map stability targets back to scene indices (stability[i] = 0.1 * (i + 1))."""
    loader = getattr(bench, loader_name)
    seen: set[int] = set()
    for _X, y in loader(batch_size=batch_size, progress=False):
        for value in y.to_list():
            seen.add(int(round(float(value) / 0.1)) - 1)
    return seen


def test_benchmark_train_val_test_loaders_no_leakage(tmp_path: Path) -> None:
    with Settings(home=tmp_path):
        bench = Factory.benchmark("tests/fixtures/benchmarks/B_mini.yaml")
        train_ids = _collect_indices(bench, "train_loader")
        val_ids = _collect_indices(bench, "val_loader")
        test_ids = _collect_indices(bench, "test_loader")

        assert len(train_ids) == 8
        assert len(val_ids) == 2
        assert len(test_ids) == 2
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)


@patch("fabric.core.benchmark.track")
def test_train_loader_uses_branded_progress(mock_track, tmp_path: Path) -> None:
    mock_track.side_effect = lambda iterable, **kwargs: iterable
    with Settings(home=tmp_path):
        bench = Factory.benchmark("tests/fixtures/benchmarks/B_mini.yaml")
        list(bench.train_loader(batch_size=4, progress=True))
    mock_track.assert_called_once()
    assert mock_track.call_args.kwargs["description"] == loader_description(
        "train", benchmark_id="B_mini"
    )
    assert mock_track.call_args.kwargs["total"] == 2


def test_benchmark_loader_yields_task_batches(tmp_path: Path) -> None:
    with Settings(home=tmp_path):
        bench = Factory.benchmark("tests/fixtures/benchmarks/B_mini.yaml")
        batches = list(bench.train_loader(batch_size=4, shuffle=False, progress=False))
        assert len(batches) == 2
        X, y = batches[0]
        assert len(X) == 1
        assert len(X[0].data) == 4
        assert len(y) == 4


def test_task_error_missing_target(tmp_path: Path) -> None:
    broken = tmp_path / "B_bad_task.yaml"
    broken.write_text(
        "dataset: D_local\n"
        "split_scheme: random_8_2_2\n"
        "sampler:\n  RandomMoleculeSampler: {}\n"
        "task:\n  PropertyPredictionTask:\n    target: missing_label\n"
        "metrics:\n  - MAE: {}\n"
    )
    with Settings(home=tmp_path):
        bench = Factory.benchmark(broken)
        with pytest.raises(TaskError, match="missing_label"):
            list(bench.train_loader(batch_size=2, progress=False))


def test_metric_error_compute_before_update() -> None:
    metric = MAE()
    with pytest.raises(MetricError, match="update"):
        metric.compute()


def test_unknown_split_loader_raises(tmp_path: Path) -> None:
    with Settings(home=tmp_path):
        bench = Factory.benchmark("tests/fixtures/benchmarks/B_mini.yaml")
        with pytest.raises(BenchmarkError, match="split 'holdout' not found"):
            bench.holdout_loader(batch_size=2, progress=False)


def test_benchmark_error_broken_dataset_ref(tmp_path: Path) -> None:
    broken = tmp_path / "B_bad_dataset.yaml"
    broken.write_text(
        "id: B_bad_dataset\n"
        "dataset: D_does_not_exist\n"
        "split_scheme: random_8_2_2\n"
        "sampler:\n  RandomMoleculeSampler: {}\n"
        "task:\n  PropertyPredictionTask:\n    target: stability\n"
        "metrics:\n  - MAE: {}\n"
    )
    with Settings(home=tmp_path):
        bench = Factory.benchmark(broken)
        with pytest.raises(BenchmarkError, match="B_bad_dataset → D_does_not_exist"):
            list(bench.train_loader(batch_size=2, progress=False))


def test_result_json_round_trip() -> None:
    result = Result(name="baseline", metrics={"MAE": 0.42})
    restored = Result.from_dict(json.loads(result.to_json()))
    assert restored.name == "baseline"
    assert restored.metrics["MAE"] == pytest.approx(0.42)


def dummy_predict(X) -> gr.GrumpyArray:
    """Return zeros matching the batch size."""
    frame = X[0].data
    return gr.array([0.0] * len(frame), dtype=gr.float64)


def _expected_zero_mae(bench, split: str) -> float:
    parts = bench.dataset.splits.partitions["random_8_2_2"][split].to_list()
    stability = bench.dataset.assets.data["stability"]
    values = [abs(float(stability[index])) for index in parts]
    return sum(values) / len(values)


def test_benchmark_update_and_result(tmp_path: Path) -> None:
    with Settings(home=tmp_path):
        bench = Factory.benchmark("tests/fixtures/benchmarks/B_mini.yaml")
        expected = _expected_zero_mae(bench, "test")

        bench.metrics.reset()
        for X, y in bench.test_loader(batch_size=16, progress=False):
            bench.update(dummy_predict(X), y)

        result = bench.result(name="baseline")
        assert result.name == "baseline"
        assert "MAE" in result.metrics
        assert result.metrics["MAE"] == pytest.approx(expected, rel=1e-6)
        assert "MAE" in json.loads(result.to_json())["metrics"]


def test_benchmark_update_applies_inverse_transform(tmp_path: Path) -> None:
    with Settings(home=tmp_path):
        bench = Factory.benchmark("tests/fixtures/benchmarks/B_mini.yaml")
        inverse = bench.dataset.transforms.inverse_transform
        with patch.object(
            bench.dataset.transforms, "inverse_transform", wraps=inverse
        ) as mock_inverse:
            bench.metrics.reset()
            y = gr.array([0.5, 1.0], dtype=gr.float64)
            bench.update(y, y)
            assert mock_inverse.call_count == 2
