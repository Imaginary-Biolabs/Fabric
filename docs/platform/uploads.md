# Uploads

Upload Grumpy Zarr **releases** and model **checkpoints** via presigned blob sessions.

## Dataset release

After local `imaginary dataset release`:

```bash
imaginary platform upload release \
  --asset D_000001 \
  --version 1 \
  --path /path/to/zarr/release
```

Requires `grumpy.json` and `release.json` in the manifest.

## Model checkpoint

```bash
imaginary platform upload checkpoint \
  --asset M_000003 \
  --version 1 \
  --path /path/to/checkpoint.pt
```

Or a directory containing `checkpoint.pt`.

## Python API

```python
from fabric.platform.upload import upload_release, upload_checkpoint

manifest = upload_release(asset_id="D_000001", version="1", path="/path/to/zarr")
print(manifest.get("id"), manifest.get("object_count"))
```

Uploads batch files (100 paths per presign request), PUT to presigned URLs, then complete the session.

---

**Next:** [Jobs](jobs.md)
