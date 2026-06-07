"""Terminal I/O helpers."""

from __future__ import annotations

import sys

from rich.console import Console

_console = Console(stderr=True)


def info(message: str) -> None:
    _console.print(f"[bold cyan]fabric[/] {message}")


def warn(message: str) -> None:
    _console.print(f"[bold yellow]fabric[/] [yellow]warning[/]: {message}", file=sys.stderr)


def error(message: str) -> None:
    _console.print(f"[bold red]fabric[/] [red]error[/]: {message}", file=sys.stderr)
