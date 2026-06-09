"""Workflow run records persisted as YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from omegaconf import OmegaConf


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StepRecord:
    """Execution record for one workflow node."""

    node_id: str
    op: str
    runtime: str
    status: str
    started_at: str = ""
    finished_at: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    logs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    iterations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunRecord:
    """Inspectable workflow run with per-step artifacts."""

    workflow_id: str
    run_id: str
    mode: str
    status: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    steps: dict[str, StepRecord] = field(default_factory=dict)
    root: str = ""
    started_at: str = field(default_factory=_utc_now)
    finished_at: str = ""
    workflow_hash: str = ""

    @classmethod
    def new(cls, *, workflow_id: str, mode: str, root: Path, workflow_hash: str = "") -> RunRecord:
        return cls(
            workflow_id=workflow_id,
            run_id=uuid4().hex[:12],
            mode=mode,
            status="pending",
            root=str(root),
            workflow_hash=workflow_hash,
        )

    def save(self) -> Path:
        """Write ``run.yaml`` and ``steps/*.yaml`` under the run directory."""
        run_dir = Path(self.root) / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        steps_dir = run_dir / "steps"
        steps_dir.mkdir(exist_ok=True)

        payload = asdict(self)
        steps = payload.pop("steps")
        OmegaConf.save(OmegaConf.create(payload), run_dir / "run.yaml")
        for step_id, step in steps.items():
            OmegaConf.save(OmegaConf.create(step), steps_dir / f"{step_id}.yaml")
        return run_dir

    @classmethod
    def load(cls, path: str | Path) -> RunRecord:
        """Load a run record from a run directory or ``run.yaml`` file."""
        run_path = Path(path)
        if run_path.is_file():
            run_dir = run_path.parent
        else:
            run_dir = run_path
        data = OmegaConf.to_container(OmegaConf.load(run_dir / "run.yaml"), resolve=True) or {}
        steps: dict[str, StepRecord] = {}
        steps_dir = run_dir / "steps"
        if steps_dir.is_dir():
            for step_file in sorted(steps_dir.glob("*.yaml")):
                step_id = step_file.stem
                step_data = OmegaConf.to_container(OmegaConf.load(step_file), resolve=True) or {}
                steps[step_id] = StepRecord(**step_data)
        data["steps"] = steps
        return cls(**data)
