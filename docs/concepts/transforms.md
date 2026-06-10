# Transforms

Transforms modify dataset **assets**, **splits**, or **batches**. They chain via `Compose`.

## Transform methods

Implement only what you need:

| Method | Scope |
|--------|-------|
| `transform_assets(assets)` | Whole-ingest metadata |
| `transform_splits(splits, *, n_scenes)` | Partition indices |
| `transform_batches(batches)` | Streamed structural batches |
| `inverse_transform(values)` | Map predictions back (e.g. unscaling) |

## Compose pipeline

`Compose.apply(batches, assets, splits, n_scenes=...)` runs:

1. `fit` on training data where implemented
2. `transform_assets`
3. Interleaved batch + split transforms

Final splits appear on `compose.splits` after the batch iterator completes.

## Built-in transforms

| Transform | Purpose |
|-----------|---------|
| `Identity` | No-op |
| `RandomSplit` | Train/val/test partition counts |
| `FilterSequenceLength` | Drop molecules over `max_len` |

## Example

```yaml
transforms:
  - RandomSplit: { n_train: 8, n_val: 2, n_test: 2 }
  - FilterSequenceLength: { max_len: 512 }
```

Custom transforms register by class name in YAML (see [Extend Fabric](../guides/extend-fabric.md)).

---

**Next:** [Benchmarks](benchmarks.md) — samplers, tasks, metrics, loaders.
