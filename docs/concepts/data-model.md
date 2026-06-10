# Data model

Fabric wraps **Grumpy dataframes** with a fixed structural schema for molecular ML.

## Schema hierarchy

```
scene → frame → molecule → chain → residue/group → atom
```

- **Scene** — top-level container (e.g. one structure file or simulation frame set)
- **Molecule** — covalent unit; common sampling level for property prediction
- **Atom** — coordinates, elements, B-factors, etc.

Access nested fields with Grumpy dot notation:

```python
ds = Factory.dataset("tests/fixtures/datasets/D_local.yaml")
ds.release()
print(ds.molecule.count())
```

## Core types

| Type | Role |
|------|------|
| `Data` | Validated Grumpy dataframe wrapper |
| `Batch` | Chunk from streaming ingest or loader |
| `Assets` | Ingest metadata (paths, external ids) |
| `Split` | Train/val/test index partitions |

## Streaming ingest

Externals yield `(batches, assets, splits)` where `batches` is a generator of `Batch` objects sized to the Grumpy chunk size. Transforms consume and emit the same shape.

---

**Next:** [Configs](configs.md) — YAML structure and hashing.
