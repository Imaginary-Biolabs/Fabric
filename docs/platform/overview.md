# Platform overview

The **platform** extra adds HTTP clients for the Imaginary API: registry reads, blob uploads, and benchmark jobs.

## Install

```bash
pip install "imaginary-fabric[platform]"
```

## When to use

| Offline core | Platform |
|--------------|----------|
| Local YAML + Zarr releases | Shared registry assets by id |
| Manual checkpoint paths | Uploaded weight manifests |
| Local eval | Remote `benchmark_eval` jobs + leaderboards |

Core Fabric works without the platform. Platform modules lazy-import and raise install hints if `httpx` is missing.

## Architecture

```
imaginary CLI / Fabric platform/*
        │
        ▼
  PlatformClient  (httpx)
        │
        ▼
  Imaginary API  /v1/...
        │
        ├── Postgres registry
        ├── R2 / MinIO blobs
        └── ARQ worker (eval jobs)
```

## Quick check

```bash
export IMAGINARY_API_BASE=https://api.imaginary.bio/v1
export IMAGINARY_API_KEY=img_...
imaginary platform status
```

Local backend default: `http://localhost:8080/v1`.

---

**Next:** [Authentication](authentication.md)
