"""Phase 8 toll station — platform HTTP client, jobs, runner, publish."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import yaml

from fabric.core.factory import Factory
from fabric.platform.client import PlatformClient, save_credentials
from fabric.platform.jobs import (
    get_job_events,
    submit_benchmark_eval,
    submit_workflow_run,
)
from fabric.platform.registry import fetch_asset_version
from fabric.platform.runner import plan_remote_submission, run_workflow_remote
from fabric.utils.errors import AuthError


def test_api_key_required(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("IMAGINARY_API_KEY", raising=False)
    monkeypatch.delenv("IMAGINARY_API_BASE", raising=False)
    monkeypatch.setattr("fabric.platform.client.Path.home", lambda: tmp_path)
    with pytest.raises(AuthError):
        PlatformClient()


def test_fetch_asset_version_caches(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IMAGINARY_API_KEY", "img_dev_test")
    monkeypatch.setenv("IMAGINARY_API_BASE", "http://example.test/v1")
    payload = {
        "config_yaml": "id: B_000010\nsplit_scheme: random_8_2_2\n",
        "meta": {"metric_spec": {"primary_metric": "mae"}},
    }

    with patch("fabric.platform.client.httpx.Client") as mock_client:
        instance = mock_client.return_value
        instance.__enter__.return_value = instance
        instance.request.return_value = httpx.Response(200, json=payload)
        cache_patch = "fabric.platform.registry.Settings.registry_cache_path"
        with patch(cache_patch, return_value=tmp_path / "registry"):
            result = fetch_asset_version("B_000010", "1", cache=True)
    assert "split_scheme" in result["config_yaml"]
    cached = tmp_path / "registry" / "B_000010" / "1" / "config.yaml"
    assert cached.is_file()


def test_save_credentials(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("fabric.platform.client.Path.home", lambda: tmp_path)
    path = save_credentials(api_key_value="img_dev_x", api_base_value="http://localhost:8080/v1")
    assert path.is_file()
    assert "img_dev_x" in path.read_text()


def test_submit_workflow_run_payload(monkeypatch) -> None:
    monkeypatch.setenv("IMAGINARY_API_KEY", "img_dev_test")
    monkeypatch.setenv("IMAGINARY_API_BASE", "http://example.test/v1")
    captured: dict = {}

    def handler(*_args, **kwargs) -> httpx.Response:
        captured["json"] = kwargs.get("json")
        return httpx.Response(
            201,
            json={"job": {"id": "job-1", "type": "workflow_run", "status": "queued"}},
        )

    with patch("fabric.platform.client.httpx.Client") as mock_client:
        instance = mock_client.return_value
        instance.__enter__.return_value = instance
        instance.request.side_effect = handler
        job = submit_workflow_run(
            asset_id="M_000003",
            mode="run",
            inputs={"features": [[1.0, 2.0]]},
            parameters={"batch_size": 2},
        )
    assert job["id"] == "job-1"
    assert captured["json"]["type"] == "workflow_run"
    assert captured["json"]["asset_id"] == "M_000003"
    assert captured["json"]["inputs"]["features"] == [[1.0, 2.0]]


def test_submit_benchmark_eval_emits_deprecation(monkeypatch) -> None:
    monkeypatch.setenv("IMAGINARY_API_KEY", "img_dev_test")
    monkeypatch.setenv("IMAGINARY_API_BASE", "http://example.test/v1")

    with patch("fabric.platform.client.httpx.Client") as mock_client:
        instance = mock_client.return_value
        instance.__enter__.return_value = instance
        instance.request.return_value = httpx.Response(
            201,
            json={"job": {"id": "job-2", "type": "workflow_run", "status": "queued"}},
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            submit_benchmark_eval(
                benchmark_id="B_000010",
                benchmark_version="1",
                model_id="M_000003",
                model_version="1",
            )
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)


def test_get_job_events(monkeypatch) -> None:
    monkeypatch.setenv("IMAGINARY_API_KEY", "img_dev_test")
    monkeypatch.setenv("IMAGINARY_API_BASE", "http://example.test/v1")

    with patch("fabric.platform.client.httpx.Client") as mock_client:
        instance = mock_client.return_value
        instance.__enter__.return_value = instance
        instance.request.return_value = httpx.Response(
            200,
            json={"items": [{"message": "validate_inputs", "level": "info", "created_at": "t"}]},
        )
        events = get_job_events("job-1")
    assert events[0]["message"] == "validate_inputs"


def test_plan_remote_submission_predict(tmp_path: Path) -> None:
    workflow_yaml = tmp_path / "W_predict.yaml"
    workflow_yaml.write_text(
        yaml.safe_dump(
            {
                "id": "W_predict",
                "schema_version": 1,
                "nodes": {
                    "infer": {
                        "op": "predict",
                        "runtime": "remote",
                        "model": "M_000003",
                        "inputs": {"features": "$inputs.features"},
                    }
                },
                "outputs": {"predictions": {"from": "infer.predictions"}},
            }
        )
    )
    workflow = Factory.workflow(str(workflow_yaml))
    spec = plan_remote_submission(workflow, {"features": [[1.0, 2.0]]})
    assert spec["kind"] == "workflow_run"
    assert spec["asset_id"] == "M_000003"
    assert spec["mode"] == "run"


def test_run_workflow_remote_builds_run_record(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("IMAGINARY_API_KEY", "img_dev_test")
    monkeypatch.setenv("IMAGINARY_API_BASE", "http://example.test/v1")
    workflow_yaml = tmp_path / "W_predict.yaml"
    workflow_yaml.write_text(
        yaml.safe_dump(
            {
                "id": "W_predict",
                "schema_version": 1,
                "nodes": {
                    "infer": {
                        "op": "predict",
                        "runtime": "remote",
                        "model": "M_000003",
                        "inputs": {"features": "$inputs.features"},
                    }
                },
                "outputs": {"predictions": {"from": "infer.predictions"}},
            }
        )
    )
    workflow = Factory.workflow(str(workflow_yaml))

    def handler(*args, **kwargs) -> httpx.Response:
        url = str(args[1] if len(args) > 1 else kwargs.get("url", ""))
        if "/events" in url:
            return httpx.Response(200, json={"items": []})
        if kwargs.get("method") == "POST" or (len(args) > 0 and args[0] == "POST"):
            return httpx.Response(
                201,
                json={"job": {"id": "job-3", "type": "workflow_run", "status": "queued"}},
            )
        return httpx.Response(
            200,
            json={
                "job": {
                    "id": "job-3",
                    "type": "workflow_run",
                    "status": "succeeded",
                    "finished_at": "2026-01-01T00:00:00+00:00",
                    "result": {
                        "outputs": {
                            "predictions": {
                                "columns": ["sample_id", "prediction"],
                                "rows": [["s1", 0.5]],
                            }
                        }
                    },
                }
            },
        )

    with patch("fabric.platform.client.httpx.Client") as mock_client:
        instance = mock_client.return_value
        instance.__enter__.return_value = instance
        instance.request.side_effect = handler
        with patch("fabric.platform.jobs.time.sleep", return_value=None):
            run = run_workflow_remote(
                workflow,
                inputs={"features": [[1.0, 2.0]]},
                root=tmp_path / "runs",
                poll_s=0.0,
            )
    assert run.status == "succeeded"
    assert run.mode == "remote"
    assert "predictions" in run.outputs
