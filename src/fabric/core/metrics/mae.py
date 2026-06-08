from __future__ import annotations

import grumpy as gr
from grumpy import GrumpyArray

from fabric.core.metric import Metric


class MAE(Metric):
    """Mean absolute error between predictions and targets.

    Example:
        >>> metric = MAE()
        >>> metric.update(gr.array([1.0, 3.0]), gr.array([1.0, 2.0]))
        >>> metric.compute()
        0.5
    """

    name = "MAE"

    def _elementwise(self, y_pred: GrumpyArray, y_true: GrumpyArray) -> GrumpyArray:
        return gr.abs(y_pred - y_true)
