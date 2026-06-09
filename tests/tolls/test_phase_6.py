"""Phase 6 toll station — YAML workflows and runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fabric import Settings
from fabric.core.factory import Factory
from fabric.core.run_record import RunRecord
from fabric.core.runner import Runner
from fabric.core.workflow import Workflow
from fabric.utils.errors import PlatformExtraRequired, WorkflowError

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture
def workflow_home(tmp_path: Path):
    with Settings(home=tmp_path):
        yield tmp_path


def test_workflow_cycle_raises() -> None:
    workflow = Factory.workflow("tests/fixtures/workflows/W_cycle.yaml")
    with pytest.raises(WorkflowError, match="cycle"):
        workflow.compile()


def test_platform_node_local_mode_raises(workflow_home: Path) -> None:
    workflow = Factory.workflow("tests/fixtures/workflows/W_platform.yaml")
    runner = Runner(mode="local", root=workflow_home / "runs")
    with pytest.raises(WorkflowError, match="requires runtime 'platform'"):
        runner.run(workflow)


def test_platform_node_hybrid_uses_remote_submit(workflow_home: Path) -> None:
    workflow = Factory.workflow("tests/fixtures/workflows/W_platform.yaml")

    def _remote_submit(node_type: str, config: dict, inputs: dict) -> dict:
        assert node_type == "M_rfdiffusion"
        return {"structures": ["design_1", "design_2"]}

    runner = Runner(mode="hybrid", root=workflow_home / "runs", remote_submit=_remote_submit)
    run = runner.run(workflow)
    assert run.status == "succeeded"
    assert run.outputs["designs"] == ["design_1", "design_2"]


def test_platform_node_remote_without_hook_raises(workflow_home: Path) -> None:
    workflow = Factory.workflow("tests/fixtures/workflows/W_platform.yaml")
    runner = Runner(mode="remote", root=workflow_home / "runs")
    with pytest.raises(PlatformExtraRequired):
        runner.run(workflow)


def test_runner_eval_workflow(workflow_home: Path) -> None:
    pytest.importorskip("torch")
    workflow = Factory.workflow("tests/fixtures/workflows/W_mini.yaml")
    runner = Runner(mode="local", root=workflow_home / "runs")
    run = runner.run(workflow, inputs={"batch_size": 4})
    assert run.status == "succeeded"
    assert "metrics" in run.outputs
    assert "MAE" in run.outputs["metrics"]
    assert np.isfinite(run.outputs["metrics"]["MAE"])
    assert (Path(run.root) / run.run_id / "run.yaml").exists()
    assert (Path(run.root) / run.run_id / "steps" / "eval_baseline.yaml").exists()


def test_run_record_reload(workflow_home: Path) -> None:
    pytest.importorskip("torch")
    workflow = Factory.workflow("tests/fixtures/workflows/W_mini.yaml")
    runner = Runner(mode="local", root=workflow_home / "runs")
    run = runner.run(workflow)
    loaded = RunRecord.load(Path(run.root) / run.run_id)
    assert loaded.run_id == run.run_id
    assert loaded.steps["eval_baseline"].op == "eval"


def test_loop_workflow_runs_twice(workflow_home: Path) -> None:
    pytest.importorskip("torch")
    workflow = Factory.workflow("tests/fixtures/workflows/W_loop.yaml")
    runner = Runner(mode="local", root=workflow_home / "runs")
    run = runner.run(workflow, inputs={"batch_size": 2})
    assert run.status == "succeeded"
    assert len(run.steps["repeat_eval"].iterations) == 2
    assert "MAE" in run.outputs["metrics"]


def test_wait_node_pauses_run(workflow_home: Path) -> None:
    from omegaconf import OmegaConf

    cfg = {
        "id": "W_wait",
        "nodes": {
            "approval": {
                "op": "wait",
                "runtime": "local",
                "kind": "user_select",
                "present": {"candidates": [1, 2, 3]},
            }
        },
        "outputs": {"approved": {"from": "approval.status"}},
    }
    workflow = Workflow(OmegaConf.create(cfg), config_path=Path("W_wait.yaml"))
    runner = Runner(mode="local", root=workflow_home / "runs")
    run = runner.run(workflow)
    assert run.status == "paused"
    assert run.steps["approval"].status == "paused"
