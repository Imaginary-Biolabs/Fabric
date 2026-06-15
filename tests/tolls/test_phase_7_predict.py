"""Phase 7 predict executor tests."""

from __future__ import annotations

from pathlib import Path

import pytest

FABRIC_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_predict_v1_runs_locally(tmp_path: Path) -> None:
    from omegaconf import OmegaConf

    from fabric.core.runner import Runner
    from fabric.core.workflow import Workflow
    from fabric.utils.config import load_config

    model_cfg = load_config(FABRIC_ROOT / "tests/fixtures/models/M_mini.yaml")
    config_path = tmp_path / "predict_test.yaml"
    workflow = Workflow(
        OmegaConf.create(
            {
                "id": "predict_test",
                "nodes": {
                    "predict": {
                        "op": "predict",
                        "runtime": "local",
                        "model": "M_mini",
                        "inputs": {"features": [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]},
                    }
                },
                "outputs": {"predictions": {"from": "predict.predictions"}},
            }
        ),
        config_path=config_path,
    )
    record = Runner(mode="local", root=tmp_path).run(
        workflow,
        platform_context={"model_configs": {"M_mini": model_cfg}},
    )
    assert record.status == "succeeded"
    predictions = record.outputs["predictions"]
    assert predictions["columns"] == ["sample_id", "prediction"]
    assert len(predictions["rows"]) == 2
