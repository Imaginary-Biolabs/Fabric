"""Publish models and versions to the Imaginary platform registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from fabric.platform.client import PlatformClient
from fabric.platform.upload import upload_checkpoint
from fabric.utils.errors import FabricError


def publish_model(
    *,
    config_path: str | Path,
    title: str | None = None,
    visibility: str = "private",
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a new model asset from a local YAML config.

    Args:
        config_path: Path to model config YAML (must include ``id``).
        title: Optional display title. Defaults to the config ``id``.
        visibility: ``private`` or ``public``.
        meta: Optional asset meta (for example ``run_interface``).

    Returns:
        Created asset version payload from the API.

    Example:
        >>> # publish_model(config_path="models/M_mini.yaml")  # doctest: +SKIP
    """
    path = Path(config_path)
    config_text = path.read_text()
    parsed = yaml.safe_load(config_text) or {}
    asset_id = parsed.get("id")
    if not asset_id:
        raise FabricError("Model config must include an 'id' field")

    client = PlatformClient()
    payload = client.request(
        "POST",
        "/assets",
        json={
            "id": str(asset_id),
            "kind": "model",
            "title": title or str(asset_id),
            "visibility": visibility,
            "config_yaml": config_text,
            "meta": meta or {},
        },
    )
    return payload


def publish_model_version(
    *,
    asset_id: str,
    config_path: str | Path | None = None,
    checkpoint_path: str | Path,
    version: str | None = None,
    visibility: str = "private",
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bump a model version and upload its checkpoint manifest.

    Args:
        asset_id: Existing model asset id (for example ``M_000003``).
        config_path: Version config YAML. When omitted, reuses the latest version config.
        checkpoint_path: Local ``checkpoint.pt`` file or directory containing it.
        version: Explicit version label. Auto-increments when omitted.
        visibility: Stored as version ``meta.status`` (``private`` or ``public``).
        meta: Optional extra version meta.

    Returns:
        Dict with ``version`` label and upload ``manifest``.

    Example:
        >>> # publish_model_version(
        ... #     asset_id="M_000003",
        ... #     checkpoint_path="results/mini/checkpoint.pt",
        ... # )  # doctest: +SKIP
    """
    client = PlatformClient()
    if config_path is None:
        latest = client.request("GET", f"/assets/{asset_id}/versions/{_latest_version(client, asset_id)}")
        config_text = latest["config_yaml"]
    else:
        config_text = Path(config_path).read_text()

    version_meta = {"status": visibility, **(meta or {})}
    body: dict[str, Any] = {
        "config_yaml": config_text,
        "meta": version_meta,
    }
    if version:
        body["version"] = str(version)

    created = client.request("POST", f"/assets/{asset_id}/versions", json=body)
    version_label = str(created.get("version") or version or "1")

    manifest = upload_checkpoint(
        asset_id=asset_id,
        version=version_label,
        path=checkpoint_path,
    )
    return {"asset_id": asset_id, "version": version_label, "manifest": manifest}


def fork_model(*, asset_id: str, source_version: str = "1") -> dict[str, Any]:
    """Fork a public model into a private derivative asset.

    Args:
        asset_id: Source model asset id.
        source_version: Source version label.

    Returns:
        Forked model version payload from the API.

    Example:
        >>> # fork_model(asset_id="M_000003")  # doctest: +SKIP
    """
    client = PlatformClient()
    return client.request(
        "POST",
        f"/assets/{asset_id}/fork",
        json={"source_version": str(source_version)},
    )


def promote_from_job(*, asset_id: str, version: str, job_id: str) -> dict[str, Any]:
    """Promote develop train output from a completed job (CLI retry path).

    Args:
        asset_id: Target model asset id.
        version: Context version label (usually the develop base version).
        job_id: Completed develop ``workflow_run`` job id.

    Returns:
        Promotion payload from the API.

    Example:
        >>> # promote_from_job(asset_id="M_fork", version="1", job_id=job_id)  # doctest: +SKIP
    """
    client = PlatformClient()
    return client.request(
        "POST",
        f"/assets/{asset_id}/versions/{version}/promote-from-job",
        json={"job_id": job_id},
    )


def _latest_version(client: PlatformClient, asset_id: str) -> str:
    asset = client.request("GET", f"/assets/{asset_id}")
    latest = asset.get("latest_version") or asset.get("item", {}).get("latest_version")
    if latest:
        return str(latest)
    raise FabricError(f"Could not determine latest version for asset {asset_id}")
