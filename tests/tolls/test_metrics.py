"""Metric axis grouping and reduction."""

from __future__ import annotations

import grumpy as gr
import pytest

from fabric.core.metrics import MAE, build_metrics
from fabric.utils.errors import MetricError


def test_mae_flat_default() -> None:
    metric = MAE()
    metric.update(gr.array([1.0, 3.0]), gr.array([1.0, 2.0]))
    assert metric.compute() == pytest.approx(0.5)


def test_mae_per_molecule_reduction_none() -> None:
    y_pred = gr.array([[1.0, 2.0], [3.0, 4.0, 5.0]])
    y_true = gr.array([[1.0, 0.0], [3.0, 3.0, 6.0]])
    metric = MAE(on="residue", per="molecule", reduction="none")
    metric.update(y_pred, y_true)
    assert metric.compute().to_list() == pytest.approx([1.0, 2.0 / 3.0])


def test_mae_per_molecule_reduction_mean() -> None:
    y_pred = gr.array([[1.0, 2.0], [3.0, 4.0, 5.0]])
    y_true = gr.array([[1.0, 0.0], [3.0, 3.0, 6.0]])
    metric = MAE(on="residue", per="molecule", reduction="mean")
    metric.update(y_pred, y_true)
    assert metric.compute() == pytest.approx(5.0 / 6.0)


def test_mae_batches_concat_before_reduction() -> None:
    metric = MAE(reduction="sum")
    metric.update(gr.array([1.0, 2.0]), gr.array([0.0, 0.0]))
    metric.update(gr.array([3.0]), gr.array([1.0]))
    assert metric.compute() == pytest.approx(5.0)


def test_metric_invalid_schema_level() -> None:
    metric = MAE(on="residue", per="not_a_level")
    with pytest.raises(MetricError, match="Unknown schema level"):
        metric.update(gr.array([[1.0]]), gr.array([[0.0]]))


def test_metric_invalid_per_relative_to_on() -> None:
    metric = MAE(on="molecule", per="residue")
    with pytest.raises(MetricError, match="outer schema level"):
        metric.update(gr.array([[1.0, 2.0]]), gr.array([[1.0, 0.0]]))


def test_build_metrics_passes_axis_kwargs() -> None:
    metrics = build_metrics(
        [{"MAE": {"on": "residue", "per": "molecule", "reduction": "none"}}]
    )
    mae = metrics.metrics[0]
    assert mae.on == "residue"
    assert mae.per == "molecule"
    assert mae.reduction == "none"
