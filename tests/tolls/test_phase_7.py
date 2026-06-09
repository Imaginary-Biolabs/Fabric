"""Phase 7 toll station — CLI smoke tests."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

FABRIC_ROOT = Path(__file__).resolve().parents[2]
CLI = [sys.executable, "-m", "fabric.cli.main"]


def _run(*args: str, cwd: Path = FABRIC_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*CLI, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def test_cli_version() -> None:
    result = _run("--version")
    assert result.returncode == 0
    assert re.search(r"\d+\.\d+\.\d+", result.stdout)


def test_cli_release_smoke(tmp_path: Path) -> None:
    result = _run(
        "release",
        "--config",
        "tests/fixtures/datasets/D_local.yaml",
        "--home",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert "release complete" in result.stderr
    assert "transform_hash" in result.stderr
    assert "D_local" in result.stderr


def test_cli_release_missing_config_shows_example() -> None:
    result = _run("release")
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "config" in combined.lower() or "required" in combined.lower()


def test_cli_train_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    root = tmp_path / "train_out"
    result = _run(
        "train",
        "--config",
        "tests/fixtures/train/train_mini.yaml",
        "--epochs",
        "1",
        "--root",
        str(root),
        "--home",
        str(tmp_path),
        "--no-progress",
    )
    assert result.returncode == 0, result.stderr
    assert "training complete" in result.stderr
    assert (root / "checkpoint.pt").exists()


def test_cli_train_epochs_override(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    root = tmp_path / "epochs_out"
    result = _run(
        "train",
        "--config",
        "tests/fixtures/train/train_mini.yaml",
        "--epochs",
        "2",
        "--root",
        str(root),
        "--home",
        str(tmp_path),
        "--no-progress",
    )
    assert result.returncode == 0, result.stderr
    assert "epochs" in result.stderr
    assert "2" in result.stderr


def test_cli_eval_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    root = tmp_path / "eval_run"
    train = _run(
        "train",
        "--config",
        "tests/fixtures/train/train_mini.yaml",
        "--epochs",
        "1",
        "--root",
        str(root),
        "--home",
        str(tmp_path),
        "--no-progress",
    )
    assert train.returncode == 0, train.stderr

    result = _run(
        "eval",
        "--benchmark",
        "B_mini",
        "--model",
        "M_mini",
        "--checkpoint",
        str(root / "checkpoint"),
        "--train-config",
        "tests/fixtures/train/train_mini.yaml",
        "--home",
        str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert "MAE" in result.stderr
    assert "eval" in result.stderr


def test_cli_workflow_run_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    result = _run(
        "workflow",
        "run",
        "--config",
        "tests/fixtures/workflows/W_mini.yaml",
        "--home",
        str(tmp_path),
        "--root",
        str(tmp_path / "wf_runs"),
        "--set",
        "batch_size=4",
    )
    assert result.returncode == 0, result.stderr
    assert "workflow complete" in result.stderr
    assert "succeeded" in result.stderr


def test_cli_platform_upload_stub() -> None:
    result = _run("platform", "upload")
    assert result.returncode != 0
    assert "platform extra" in (result.stdout + result.stderr).lower()
