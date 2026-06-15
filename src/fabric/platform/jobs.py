"""Job submission and polling for the Imaginary platform.

Submit workflow_run jobs, poll status and events, and use the legacy
benchmark_eval shim when needed.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from typing import Any
from uuid import UUID

from fabric.platform.client import PlatformClient
from fabric.utils.errors import JobError


def submit_workflow_run(
    *,
    asset_id: str,
    version: str = "1",
    mode: str = "run",
    inputs: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
    develop_profile: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Submit a canonical workflow_run job for model Run or Develop.

    Args:
        asset_id: Model asset id (for example ``M_000003``).
        version: Model version label.
        mode: ``run`` for inference or ``develop`` for training.
        inputs: Normalized run/develop inputs.
        parameters: Optional batch size and other execution parameters.
        develop_profile: Develop profile id when ``mode='develop'``.
        meta: Optional job metadata.

    Returns:
        Job record dict (``id``, ``status``, …).

    Example:
        >>> # submit_workflow_run(
        ... #     asset_id="M_000003",
        ... #     mode="run",
        ... #     inputs={"features": [[1.0, 2.0]]},
        ... # )  # doctest: +SKIP
    """
    client = PlatformClient()
    payload: dict[str, Any] = {
        "type": "workflow_run",
        "asset_id": asset_id,
        "asset_version": str(version),
        "mode": mode,
        "inputs": inputs or {},
        "parameters": parameters or {},
        "meta": meta or {},
    }
    if develop_profile:
        payload["develop_profile"] = develop_profile
    response = client.request("POST", "/jobs", json=payload)
    return response["job"]


def submit_benchmark_eval(
    *,
    benchmark_id: str,
    benchmark_version: str,
    model_id: str,
    model_version: str,
    overrides: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Submit a benchmark evaluation job (legacy shim → ``eval_v1``).

    .. deprecated::
        Prefer :func:`submit_workflow_run` for model hub runs. Keep this for
        benchmark leaderboard eval until the shim is removed server-side.

    Args:
        benchmark_id: Benchmark asset id.
        benchmark_version: Benchmark version label.
        model_id: Model asset id.
        model_version: Model version label.
        overrides: Optional benchmark or model config overrides.
        meta: Optional job metadata.

    Returns:
        Job record dict (``id``, ``status``, …).

    Example:
        >>> # submit_benchmark_eval(
        ... #     benchmark_id="B_mini",
        ... #     benchmark_version="1",
        ... #     model_id="M_mlp",
        ... #     model_version="1",
        ... # )  # doctest: +SKIP
    """
    warnings.warn(
        "submit_benchmark_eval is deprecated; prefer submit_workflow_run for model "
        "hub runs. This shim remains for benchmark leaderboard eval.",
        DeprecationWarning,
        stacklevel=2,
    )
    client = PlatformClient()
    payload = client.request(
        "POST",
        "/jobs",
        json={
            "type": "benchmark_eval",
            "benchmark_id": benchmark_id,
            "benchmark_version": benchmark_version,
            "model_id": model_id,
            "model_version": model_version,
            "overrides": overrides or {},
            "meta": meta or {},
        },
    )
    return payload["job"]


def get_job(job_id: str | UUID) -> dict[str, Any]:
    """Fetch the current state of one platform job.

    Args:
        job_id: Job UUID or string id.

    Returns:
        Job record dict including ``status`` and optional ``error_message``.

    Example:
        >>> # get_job("550e8400-e29b-41d4-a716-446655440000")  # doctest: +SKIP
    """
    client = PlatformClient()
    payload = client.request("GET", f"/jobs/{job_id}")
    return payload["job"]


def get_job_events(job_id: str | UUID) -> list[dict[str, Any]]:
    """Fetch canonical step/event messages for a platform job.

    Args:
        job_id: Job UUID or string id.

    Returns:
        List of event dicts with ``message``, ``level``, and ``created_at``.

    Example:
        >>> # get_job_events(job_id)  # doctest: +SKIP
    """
    client = PlatformClient()
    payload = client.request("GET", f"/jobs/{job_id}/events")
    return list(payload.get("items") or [])


def wait_for_job(
    job_id: str | UUID,
    *,
    timeout_s: float = 300.0,
    poll_s: float = 1.0,
    on_poll: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Poll a job until it reaches a terminal status.

    Args:
        job_id: Job UUID or string id.
        timeout_s: Maximum wait time in seconds.
        poll_s: Delay between status polls.
        on_poll: Optional callback invoked with the latest job record on each poll.

    Returns:
        Final job record when ``status`` is ``succeeded``.

    Raises:
        JobError: On failure, cancellation, or timeout.

    Example:
        >>> # wait_for_job(job_id, timeout_s=60.0)  # doctest: +SKIP
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = get_job(job_id)
        if on_poll is not None:
            on_poll(job)
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            if job["status"] != "succeeded":
                message = job.get("error_message") or f"Job {job_id} ended with {job['status']}"
                raise JobError(message)
            return job
        time.sleep(poll_s)
    raise JobError(f"Timed out waiting for job {job_id}")
