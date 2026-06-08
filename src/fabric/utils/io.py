"""Terminal I/O helpers."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from fabric.utils.constants import BRAND_GREEN_DARK, BRAND_GREEN_LIGHT, BRAND_INK_600

_console = Console(stderr=True)


def info(message: str) -> None:
    """Print an informational Fabric message to stderr.

    Args:
        message: Message text (without the ``fabric`` prefix).
    """
    _console.print(f"[bold cyan]fabric[/] {message}")


def warn(message: str) -> None:
    """Print a warning message to stderr.

    Args:
        message: Warning text.
    """
    _console.print(f"[bold yellow]fabric[/] [yellow]warning[/]: {message}", file=sys.stderr)


def error(message: str) -> None:
    """Print an error message to stderr.

    Args:
        message: Error text.
    """
    _console.print(f"[bold red]fabric[/] [red]error[/]: {message}", file=sys.stderr)


def _imaginary_progress(*, transient: bool = True) -> Progress:
    """Create a Rich progress bar styled with Imaginary brand colors.

    Args:
        transient: When ``True``, remove the bar after completion.

    Returns:
        Configured :class:`~rich.progress.Progress` instance.
    """
    return Progress(
        TextColumn("[bold]fabric[/] {task.description}"),
        BarColumn(
            bar_width=40,
            style=BRAND_INK_600,
            complete_style=BRAND_GREEN_DARK,
            finished_style=BRAND_GREEN_LIGHT,
            pulse_style=BRAND_INK_600,
        ),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=_console,
        transient=transient,
        expand=True,
    )


class ProgressBar:
    """Context-managed progress bar with Imaginary styling.

    Args:
        description: Task label shown in the progress bar.
        total: Optional total step count for percentage display.
        transient: When ``True``, remove the bar after the context exits.

    Example:
        >>> from fabric.utils.io import ProgressBar
        >>> with ProgressBar("loading", total=3) as bar:
        ...     for _ in range(3):
        ...         bar.advance()
    """

    def __init__(
        self,
        description: str,
        total: int | float | None = None,
        *,
        transient: bool = True,
    ) -> None:
        self._description = description
        self._total = total
        self._transient = transient
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None

    def __enter__(self) -> ProgressBar:
        """Start the progress bar and register the task.

        Returns:
            This progress bar instance.
        """
        self._progress = _imaginary_progress(transient=self._transient)
        self._progress.start()
        self._task_id = self._progress.add_task(self._description, total=self._total)
        return self

    def __exit__(self, *_: object) -> None:
        """Stop and optionally hide the progress bar."""
        if self._progress is not None:
            self._progress.stop()

    def advance(self, advance: float = 1) -> None:
        """Advance the progress bar by one or more steps.

        Args:
            advance: Number of steps to add to the current task.

        Raises:
            RuntimeError: If called outside an active context manager.
        """
        if self._progress is None or self._task_id is None:
            raise RuntimeError("progress bar is not active")
        self._progress.update(self._task_id, advance=advance)


def progress(
    description: str,
    total: int | float | None = None,
    *,
    transient: bool = True,
) -> ProgressBar:
    """Open a branded terminal progress bar.

    Args:
        description: Task label shown in the progress bar.
        total: Optional total step count.
        transient: When ``True``, remove the bar after completion.

    Returns:
        Context-managed :class:`ProgressBar`.

    Example:
        >>> from fabric.utils.io import progress
        >>> with progress("releasing", total=10) as bar:
        ...     bar.advance(10)
    """
    return ProgressBar(description, total, transient=transient)


def track(
    iterable: Iterable[Any],
    *,
    description: str = "working",
    total: int | None = None,
    transient: bool = True,
) -> Iterator[Any]:
    """Iterate with a branded progress bar.

    Args:
        iterable: Items to yield while updating progress.
        description: Task label shown in the progress bar.
        total: Optional total count. Inferred from ``len(iterable)`` when omitted.
        transient: When ``True``, remove the bar after iteration completes.

    Yields:
        Items from ``iterable`` in order.

    Example:
        >>> from fabric.utils.io import track
        >>> list(track(range(3), description="items"))
        [0, 1, 2]
    """
    if total is None and hasattr(iterable, "__len__"):
        total = len(iterable)  # type: ignore[arg-type]
    with progress(description, total=total, transient=transient) as bar:
        for item in iterable:
            yield item
            bar.advance()
