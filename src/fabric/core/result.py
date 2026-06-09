"""Benchmark evaluation result formatting."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table

from fabric.utils.constants import BRAND_GREEN_DARK, BRAND_GREEN_LIGHT, BRAND_INK_600

_console = Console(stderr=True)


@dataclass
class Result:
    """Named benchmark evaluation result with multiple output formats.

    Args:
        name: Model or run label.
        metrics: Mapping of metric name to scalar value.

    Example:
        >>> result = Result(name="baseline", metrics={"MAE": 0.42})
        >>> result.to_json()
    """

    name: str
    metrics: dict[str, float | list[Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""
        return {"name": self.name, "metrics": dict(self.metrics)}

    def to_json(self) -> str:
        """Serialize the result as a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Result:
        """Restore a result from :meth:`to_dict` output."""
        metrics = payload.get("metrics") or {}
        parsed: dict[str, float | list[Any]] = {}
        for key, value in metrics.items():
            if isinstance(value, list):
                parsed[str(key)] = value
            else:
                parsed[str(key)] = float(value)
        return cls(name=str(payload.get("name", "")), metrics=parsed)

    def to_rich(self) -> None:
        """Print a Rich table to stderr."""
        table = Table(
            title=f"[bold {BRAND_GREEN_DARK}]eval · {self.name}[/]",
            header_style=f"bold {BRAND_INK_600}",
            border_style=BRAND_GREEN_DARK,
        )
        table.add_column("metric", style=BRAND_INK_600)
        table.add_column("value", justify="right", style=BRAND_GREEN_LIGHT)
        for key in sorted(self.metrics):
            table.add_row(key, f"{self.metrics[key]:.6g}")
        _console.print(table)

    def to_latex(self) -> str:
        """Return a LaTeX tabular block."""
        rows = " \\\\\n".join(
            f"{key} & {self.metrics[key]:.6g}" for key in sorted(self.metrics)
        )
        return (
            "\\begin{tabular}{lr}\n"
            "\\textbf{metric} & \\textbf{value} \\\\\n"
            f"{rows}\n"
            "\\end{tabular}\n"
        )
