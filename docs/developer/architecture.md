# Architecture

Fabric splits into four install surfaces:

```
src/fabric/
  core/       # datasets, benchmarks, training, workflows (offline-capable)
  utils/      # settings, config, hashing, errors
  platform/   # optional Imaginary API client ([platform] extra)
  cli/        # imaginary Typer entry point
```

## Design principles

1. **Config-first** — YAML recipes with canonical content hashes
2. **Grumpy underneath** — numerics and Zarr I/O stay in Grumpy
3. **Registry pattern** — transforms, externals, metrics, layers resolve by class name from YAML
4. **Platform optional** — `core` works fully offline; `platform/` adds HTTP integration
5. **One Factory** — `Factory.dataset`, `.benchmark`, `.model`, `.train`, `.eval`, `.workflow`

## Data flow

```
YAML config → Factory → Dataset.release() → Grumpy Zarr
                      → Benchmark loaders → Collater → Backend → Trainer
                      → Workflow → Runner → node executors
```

## Stack position

```
Grumpy (arrays, Zarr) → Fabric (orchestration) → Imaginary platform (registry, jobs)
                     ↘ imaginary CLI / website / Darwin
```

## Related docs

- Internal product spec: Imaginary workspace `.ai/project/fabric.md`
- Platform contracts: `.ai/project/backend/contracts.md`
- Grumpy: [grumpy docs](https://imaginary-biolabs.github.io/Grumpy/)

---

**Next:** [Testing](testing.md)
