# Project layout

Fabric organizes local storage into **tiers** under a home directory (default `~/.imaginary`).

## Directory tiers

| Tier | Setting | Default | Purpose |
|------|---------|---------|---------|
| **Home** | `Settings(home=...)` | `~/.imaginary` | Root for config and caches |
| **Raw** | `raw=` override | `{home}/raw` | External source files (PDB, mmCIF) |
| **Processed** | `processed=` | `{home}/processed` | Ingested releases before transforms |
| **Transformed** | `transformed=` | `{home}/transformed` | Final dataset releases after transform chain |
| **Scratch** | `scratch=` | `None` | Optional fast local copy for training |
| **Registry cache** | — | `{home}/registry` | Cached platform YAML configs |

```python
from fabric import Settings

with Settings(
    home="/data/imaginary",
    raw="/mnt/shared/raw",
    processed="/mnt/shared/processed",
    scratch="/local/ssd/scratch",
):
    print(Settings.processed_path())
    print(Settings.registry_cache_path())
```

## Config file IDs

Platform and local assets use typed prefixes:

| Prefix | Kind | Example |
|--------|------|---------|
| `D_` | Dataset | `D_000001` |
| `B_` | Benchmark | `B_000010` |
| `M_` | Model | `M_000003` |
| `W_` | Workflow | `W_mini` |

`Factory.dataset("D_local")` resolves YAML under `tests/fixtures/datasets/`, the registry cache, or an explicit path.

## Release cache key

Dataset releases are cached by:

```
{id}/{version}/{transform_hash}/
```

`transform_hash` is a canonical SHA-256 of the config (see [Configs](concepts/configs.md)). Re-running `release()` hits the cache when the recipe is unchanged.

## Repository layout

```
fabric/
  src/fabric/
    core/       # datasets, benchmarks, training, workflows
    utils/      # settings, config loading, hashing
    platform/   # optional Imaginary API client
    cli/        # imaginary command
  tests/
    fixtures/   # YAML configs and mini structures
    tolls/      # phase integration tests
  docs/
```

---

**Next:** [Data model](concepts/data-model.md) — schema levels and batches.
