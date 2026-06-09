"""Blob upload client."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

import httpx

from fabric.platform.client import PlatformClient
from fabric.utils.errors import BlobError


def _walk_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file()]


def upload_tree(
    *,
    asset_id: str,
    version: str,
    kind: str,
    root: Path,
) -> dict[str, Any]:
    """Upload a directory tree via blob session API."""
    root = root.resolve()
    if not root.is_dir():
        raise BlobError(f"Upload path is not a directory: {root}")
    client = PlatformClient()
    session = client.request(
        "POST",
        "/blobs/sessions",
        json={"asset_id": asset_id, "version": version, "kind": kind},
    )
    session_id = session["session_id"]
    files = _walk_files(root)
    rel_paths = [str(path.relative_to(root)) for path in files]
    complete_objects: list[dict[str, Any]] = []
    batch_size = 100
    for offset in range(0, len(rel_paths), batch_size):
        batch = rel_paths[offset : offset + batch_size]
        presign = client.request("POST", f"/blobs/sessions/{session_id}/presign", json={"paths": batch})
        uploads = {item["path"]: item["url"] for item in presign["uploads"]}
        for rel in batch:
            file_path = root / rel
            data = file_path.read_bytes()
            url = uploads[rel]
            content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            with httpx.Client(timeout=120.0) as http:
                put = http.put(url, content=data, headers={"Content-Type": content_type})
            if put.status_code >= 400:
                raise BlobError(f"Upload failed for {rel}: {put.status_code}")
            etag = put.headers.get("etag", "").strip('"')
            complete_objects.append({"path": rel, "etag": etag or "unknown", "size_bytes": len(data)})
    manifest = client.request(
        "POST",
        f"/blobs/sessions/{session_id}/complete",
        json={"objects": complete_objects},
    )
    return manifest


def upload_release(*, asset_id: str, version: str, path: str | Path) -> dict[str, Any]:
    return upload_tree(asset_id=asset_id, version=str(version), kind="dataset_release", root=Path(path))


def upload_checkpoint(*, asset_id: str, version: str, path: str | Path) -> dict[str, Any]:
    root = Path(path)
    if root.is_file():
        if root.name != "checkpoint.pt":
            raise BlobError("Checkpoint upload expects checkpoint.pt or a directory containing it")
        return upload_tree(asset_id=asset_id, version=str(version), kind="model_checkpoint", root=root.parent)
    if (root / "checkpoint.pt").is_file():
        return upload_tree(asset_id=asset_id, version=str(version), kind="model_checkpoint", root=root)
    raise BlobError(f"No checkpoint.pt found under {root}")
