# Config schemas

Fabric YAML configs align with Imaginary platform JSON schemas (versioned).

## Schema versions

| Kind | Schema id | Fabric module |
|------|-----------|---------------|
| Dataset | `dataset/0.2` | `core/dataset.py` |
| Benchmark | `benchmark/0.2` | `core/benchmark.py` |
| Model | `model/0.2` | `core/models/` |
| Workflow | `workflow/0.1` | `core/workflow.py` |

Platform copies live in the backend repo and `.ai/project/backend/schemas/` in the Imaginary workspace.

## Dataset (`D_*`)

Required concepts:

- `id` — asset identifier
- `external` **xor** `parent`
- `transforms` — ordered list of transform configs

Example: `tests/fixtures/datasets/D_local.yaml`

## Benchmark (`B_*`)

Required concepts:

- `dataset` — dataset id or path
- `split_scheme` — key into dataset splits
- `sampler`, `task`, `metrics`

Platform benchmarks may store `metric_spec` in asset metadata rather than inline YAML.

Example: `tests/fixtures/benchmarks/B_mini.yaml`

## Model (`M_*`)

Slot-based:

- `scaffold` — architecture template
- `layers` — ordered layer configs
- `objective` — loss / training objective

Example: `tests/fixtures/models/M_mini.yaml`

## Workflow (`W_*`)

- `inputs` — workflow input schema
- `nodes` — list of `{id, op, inputs}`

Example: `tests/fixtures/workflows/W_mini.yaml`

## Hashing

Configs are hashed after normalizing to JSON-safe structures:

```python
from fabric.utils.hashing import config_hash
```

Platform stores `config_hash` on each `asset_version` row.

---

**Next:** [API reference](api.md)
