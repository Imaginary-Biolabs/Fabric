# Datasets

A **Dataset** is a config-driven container: external ingest or parent dataset, transform chain, and release artifacts on disk.

## Lifecycle

1. **Resolve config** — YAML path or id via `Factory.dataset`
2. **Ingest** — external adapter streams structures into Grumpy layout
3. **Transform** — `Compose` runs fit/apply over batches, assets, splits
4. **Release** — write Zarr tree + `release.json` sidecar
5. **Cache** — subsequent loads skip work when `transform_hash` matches

```python
from fabric.core.factory import Factory

ds = Factory.dataset("tests/fixtures/datasets/D_local.yaml")
path = ds.release()
ds2 = Factory.dataset("tests/fixtures/datasets/D_local.yaml")  # cache hit
assert ds2.transform_hash == ds.transform_hash
```

## Apply vs live transforms

| Method | When | Use for |
|--------|------|---------|
| `.apply(transform)` | Writes new release to disk | Expensive preprocessing |
| `.live(transform)` | Per-batch at read time | Augmentation, masking |

## Scratch staging

When `Settings.scratch_path()` is set, training can stage releases to fast local storage:

```python
ds.stage_to_scratch()  # copy transformed release to scratch tier
```

## CLI

```bash
imaginary dataset release --config tests/fixtures/datasets/D_local.yaml
imaginary dataset info --config tests/fixtures/datasets/D_local.yaml
```

---

**Next:** [Transforms](transforms.md) — Compose, fit, and transform_hash.
