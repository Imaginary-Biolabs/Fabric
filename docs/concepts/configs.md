# Configs

Fabric configs are **YAML files** loaded via OmegaConf. Canonical hashes (`sha256:...`) identify reproducible releases.

## Dataset config

```yaml
# D_local.yaml
id: D_local
external:
  LocalAdapter:
    paths: ["tests/fixtures/structures/*.pdb"]
transforms:
  - RandomSplit: { n_train: 8, n_val: 2, n_test: 2 }
  - FilterSequenceLength: { max_len: 512 }
```

Either `external` **or** `parent` is set — not both. `parent` derives from another dataset version.

## Benchmark config

```yaml
id: B_mini
dataset: D_local
split_scheme: random_8_2_2
sampler:
  RandomMoleculeSampler: {}
task:
  PropertyPredictionTask:
    target: stability
metrics:
  - MAE: {}
```

## Model config

Models use **slot-based** composition (layers, scaffold, objective). See fixtures under `tests/fixtures/models/`.

## Content hashing

```python
from fabric.utils.hashing import config_hash
from fabric.utils.config import load_config

cfg = load_config("tests/fixtures/datasets/D_local.yaml")
print(config_hash(cfg))
```

Hashes exclude volatile fields and sort keys deterministically. Platform assets store the hash on each `asset_version`.

## Platform IDs

When using the Imaginary registry, configs are fetched by id and version:

```python
from fabric.platform.registry import fetch_asset_version

payload = fetch_asset_version("B_000010", "1")
print(payload["config_yaml"])
```

Cached configs land in `Settings.registry_cache_path() / {id} / {version} / config.yaml`.

---

**Next:** [Datasets](datasets.md) — release lifecycle and caching.
