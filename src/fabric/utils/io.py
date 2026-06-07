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
    _console.print(f"[bold cyan]fabric[/] {message}")


def warn(message: str) -> None:
    _console.print(f"[bold yellow]fabric[/] [yellow]warning[/]: {message}", file=sys.stderr)


def error(message: str) -> None:
    _console.print(f"[bold red]fabric[/] [red]error[/]: {message}", file=sys.stderr)


def _imaginary_progress(*, transient: bool = True) -> Progress:
    """Rich progress bar styled with Imaginary brand colors."""
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
    """Context-managed progress bar with Imaginary styling."""

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
        self._progress = _imaginary_progress(transient=self._transient)
        self._progress.start()
        self._task_id = self._progress.add_task(self._description, total=self._total)
        return self

    def __exit__(self, *_: object) -> None:
        if self._progress is not None:
            self._progress.stop()

    def advance(self, advance: float = 1) -> None:
        if self._progress is None or self._task_id is None:
            raise RuntimeError("progress bar is not active")
        self._progress.update(self._task_id, advance=advance)


def progress(
    description: str,
    total: int | float | None = None,
    *,
    transient: bool = True,
) -> ProgressBar:
    """Open a branded terminal progress bar."""
    return ProgressBar(description, total, transient=transient)


def track(
    iterable: Iterable[Any],
    *,
    description: str = "working",
    total: int | None = None,
    transient: bool = True,
) -> Iterator[Any]:
    """Iterate with a branded progress bar."""
    if total is None and hasattr(iterable, "__len__"):
        total = len(iterable)  # type: ignore[arg-type]
    with progress(description, total=total, transient=transient) as bar:
        for item in iterable:
            yield item
            bar.advance()
