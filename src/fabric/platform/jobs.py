"""Job submission and polling for the Imaginary platform.

Submit benchmark evaluation jobs and block until they finish.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from fabric.platform.client import PlatformClient
from fabric.utils.errors import JobError


def submit_benchmark_eval(
    *,
    benchmark_id: str,
    benchmark_version: str,
    model_id: str,
    model_version: str,
    overrides: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Submit a benchmark evaluation job.

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


def wait_for_job(
    job_id: str | UUID,
    *,
    timeout_s: float = 300.0,
    poll_s: float = 1.0,
) -> dict[str, Any]:
    """Poll a job until it reaches a terminal status.

    Args:
        job_id: Job UUID or string id.
        timeout_s: Maximum wait time in seconds.
        poll_s: Delay between status polls.

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
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            if job["status"] != "succeeded":
                message = job.get("error_message") or f"Job {job_id} ended with {job['status']}"
                raise JobError(message)
            return job
        time.sleep(poll_s)
    raise JobError(f"Timed out waiting for job {job_id}")
