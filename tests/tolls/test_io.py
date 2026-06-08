"""Terminal I/O helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fabric.utils import constants
from fabric.utils.io import (
    ProgressBar,
    _imaginary_progress,
    loader_description,
    progress,
    track,
)


def test_loader_description_uses_brand_markup() -> None:
    label = loader_description("train", benchmark_id="B_mini")
    assert constants.BRAND_GREEN_DARK in label
    assert "B_mini" in label


def test_imaginary_progress_uses_brand_colors() -> None:
    bar = _imaginary_progress()
    bar_column = next(col for col in bar.columns if col.__class__.__name__ == "BarColumn")
    assert bar_column.style == constants.BRAND_INK_600
    assert bar_column.complete_style == constants.BRAND_GREEN_DARK
    assert bar_column.finished_style == constants.BRAND_GREEN_LIGHT


@patch("fabric.utils.io._imaginary_progress")
def test_progress_context_manager(mock_factory: MagicMock) -> None:
    mock_progress = MagicMock()
    mock_factory.return_value = mock_progress
    mock_progress.add_task.return_value = "task-1"

    with progress("loading", total=10) as bar:
        bar.advance(2)
        bar.advance()

    mock_factory.assert_called_once_with(transient=True)
    mock_progress.start.assert_called_once()
    mock_progress.add_task.assert_called_once_with("loading", total=10)
    assert mock_progress.update.call_args_list == [
        (("task-1",), {"advance": 2}),
        (("task-1",), {"advance": 1}),
    ]
    mock_progress.stop.assert_called_once()


def test_progress_advance_without_context_raises() -> None:
    bar = ProgressBar("loading", total=1)
    with pytest.raises(RuntimeError, match="not active"):
        bar.advance()


@patch("fabric.utils.io.progress")
def test_track_iterates_and_advances(mock_progress: MagicMock) -> None:
    mock_bar = MagicMock()
    mock_progress.return_value.__enter__.return_value = mock_bar

    result = list(track([1, 2, 3], description="items"))

    assert result == [1, 2, 3]
    mock_progress.assert_called_once_with("items", total=3, transient=True)
    assert mock_bar.advance.call_count == 3
