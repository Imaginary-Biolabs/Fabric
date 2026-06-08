"""Benchmark evaluation result formatting."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table

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
        table = Table(title=f"fabric result: {self.name}")
        table.add_column("metric")
        table.add_column("value", justify="right")
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
