# Workflows

**Workflows** are YAML-defined directed graphs of ops executed by `Runner`.

## Workflow config

```yaml
id: W_mini
inputs:
  benchmark: B_mini
  model: M_mini
nodes:
  - id: eval_test
    op: eval
    inputs:
      benchmark: ${inputs.benchmark}
      model: ${inputs.model}
      split: test
```

## Runner modes

| Mode | Behavior |
|------|----------|
| `local` | Execute all nodes in-process |
| `hybrid` | Local orchestration; platform nodes delegate remotely |
| `remote` | Platform-backed execution (future) |

```python
from fabric.core.factory import Factory
from fabric.core.runner import Runner

workflow = Factory.workflow("tests/fixtures/workflows/W_mini.yaml")
run = Runner(mode="local", root="results/workflows").run(workflow)
print(run.status, run.outputs)
```

## Built-in node ops

| Op | Purpose |
|----|---------|
| `value` | Inject constant |
| `eval` | Run benchmark eval |
| `loop` | Iterate sub-graph |
| `wait` | Delay / sync |
| `platform` | Platform job stub / delegation |

## CLI

```bash
imaginary workflow run --config tests/fixtures/workflows/W_mini.yaml --mode local
```

Run artifacts land under `results/workflows/{run_id}/`.

---

**Next:** [Release a dataset](../guides/release-a-dataset.md) — step-by-step recipe.
