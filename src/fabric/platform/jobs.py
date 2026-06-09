"""Job submission client."""

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
    client = PlatformClient()
    payload = client.request("GET", f"/jobs/{job_id}")
    return payload["job"]


def wait_for_job(
    job_id: str | UUID,
    *,
    timeout_s: float = 300.0,
    poll_s: float = 1.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = get_job(job_id)
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            if job["status"] != "succeeded":
                raise JobError(job.get("error_message") or f"Job {job_id} ended with {job['status']}")
            return job
        time.sleep(poll_s)
    raise JobError(f"Timed out waiting for job {job_id}")
