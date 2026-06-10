# Release a dataset

## Prerequisites

- Grumpy installed
- Fabric with `[core]` extra
- PDB fixtures or your own structure files

## Steps

1. **Write or reuse YAML** — see `tests/fixtures/datasets/D_local.yaml`

2. **Release**

```bash
imaginary dataset release --config tests/fixtures/datasets/D_local.yaml
```

3. **Inspect output**

The transformed path is printed. Confirm sidecar files:

```bash
ls $(python -c "
from fabric.core.factory import Factory
ds = Factory.dataset('tests/fixtures/datasets/D_local.yaml')
print(ds.release())
")
# expect grumpy.json, release.json, zarr arrays
```

4. **Upload to platform (optional)**

```bash
imaginary platform upload release --asset D_000001 --version 1 --path /path/to/release
```

Requires API credentials and a matching platform asset id.

---

**Next:** [Train a model](train-a-model.md)
