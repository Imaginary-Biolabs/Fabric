# Models and training

Fabric models are **YAML-configured** slot graphs: layers, scaffold, and objective. Training goes through `Trainer` and a pluggable **backend** (PyTorch first).

## Model config

Models reference slots by registry name:

```yaml
id: M_mini
scaffold:
  MLPScaffold: { hidden: 32 }
layers:
  - LinearLayer: { in_features: 64, out_features: 32 }
  - ActivationLayer: { name: relu }
objective:
  SupervisedObjective: {}
```

```python
model = Factory.model("tests/fixtures/models/M_mini.yaml")
trainer = Factory.trainer("tests/fixtures/train/train_mini.yaml")
```

## Trainer

`Trainer` accepts benchmark + model configs, backend, collater, logger, and loop settings (epochs, validation frequency, precision, etc.).

```python
Factory.train(
    "tests/fixtures/train/train_mini.yaml",
    epochs=5,
    root="results/run1",
)
```

## Collaters

Collaters reshape ragged `(X, y)` into framework tensors:

| Collater | Strategy |
|----------|----------|
| `LongCollater` | PyG-style concatenation with index vectors |
| `WideCollater` | Padding to rectangular batches |

## Backends

| Backend | Framework |
|---------|-----------|
| `TorchBackend` | PyTorch + Lightning |
| `TensorflowBackend` | TensorFlow (optional extra) |

Install torch support: `pip install "imaginary-fabric[torch]"`.

## Results

`Factory.eval` returns a `Result` with metrics, formatting helpers (Rich table, JSON, LaTeX), and optional platform leaderboard merge.

---

**Next:** [Workflows](workflows.md) — YAML graphs and Runner.
