"""Branded Rich console helpers for the Fabric CLI."""

from __future__ import annotations

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from fabric.utils.constants import (
    BRAND_GREEN_DARK,
    BRAND_GREEN_LIGHT,
    BRAND_INK_400,
    BRAND_INK_600,
)

console = Console(stderr=True)


def print_banner(*, compact: bool = False) -> None:
    """Print the Imaginary Fabric welcome banner."""
    if compact:
        console.print(
            f"[bold {BRAND_GREEN_DARK}]Imaginary Fabric[/] "
            f"[{BRAND_INK_400}]· biology is becoming programmable[/]"
        )
        return
    title = Text("Imaginary Fabric", style=f"bold {BRAND_GREEN_DARK}", justify="center")
    tagline = Text(
        "Biology is becoming programmable.",
        style=f"italic {BRAND_INK_400}",
        justify="center",
    )
    body = Text.assemble(title, "\n", tagline)
    console.print(
        Panel(
            Align.center(body),
            border_style=BRAND_GREEN_DARK,
            padding=(1, 4),
            title="[bold]imaginary.bio[/]",
            title_align="left",
        )
    )


def success_panel(title: str, rows: list[tuple[str, str]]) -> None:
    """Render a green-bordered summary panel for successful commands."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style=f"bold {BRAND_INK_600}")
    table.add_column(style=BRAND_GREEN_LIGHT)
    for label, value in rows:
        table.add_row(label, value)
    console.print(
        Panel(
            table,
            title=f"[bold {BRAND_GREEN_DARK}]{title}[/]",
            border_style=BRAND_GREEN_DARK,
            padding=(0, 1),
        )
    )


def info_panel(message: str, *, title: str = "fabric") -> None:
    """Render an informational panel."""
    console.print(
        Panel(
            message,
            title=f"[bold {BRAND_GREEN_DARK}]{title}[/]",
            border_style=BRAND_INK_400,
            padding=(0, 1),
        )
    )


def print_error(message: str) -> None:
    """Print a branded error line."""
    console.print(f"[bold red]fabric[/] [red]error[/]: {message}")


def branded_result_table(name: str, metrics: dict[str, float | list]) -> Table:
    """Build a Rich metrics table with Imaginary styling."""
    table = Table(
        title=f"[bold {BRAND_GREEN_DARK}]eval · {name}[/]",
        header_style=f"bold {BRAND_INK_600}",
        border_style=BRAND_GREEN_DARK,
        title_style=f"bold {BRAND_GREEN_DARK}",
    )
    table.add_column("metric", style=BRAND_INK_600)
    table.add_column("value", justify="right", style=BRAND_GREEN_LIGHT)
    for key in sorted(metrics):
        table.add_row(key, f"{metrics[key]:.6g}")
    return table
