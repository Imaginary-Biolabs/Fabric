"""Disk-backed training logger.

Appends per-step losses and evaluation metrics to files under a run directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fabric.core.logger import Logger


class DiskLogger(Logger):
    """Append losses and metrics to CSV/JSON files under a run directory.

    Args:
        root: Directory for run artifacts.

    Example:
        >>> from pathlib import Path
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     logger = DiskLogger(root=Path(tmp))
        ...     logger.log_loss(0.42)
        ...     (Path(tmp) / "train_loss.csv").exists()
        True
    """

    name = "DiskLogger"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.epoch = 0
        self.step = 0

    def log_loss(self, loss: float, *, mode: str = "train") -> None:
        """Append one loss row to ``{mode}_loss.csv``."""
        path = self.root / f"{mode}_loss.csv"
        row = f"{self.epoch},{self.step},{loss}\n"
        if not path.exists():
            path.write_text("epoch,step,loss\n")
        with path.open("a") as handle:
            handle.write(row)

    def log_metrics(self, metrics: dict[str, Any], *, mode: str = "eval") -> None:
        """Write metrics as JSON under the run directory."""
        payload = {"epoch": self.epoch, "step": self.step, "metrics": metrics}
        path = self.root / f"{mode}_metrics.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
