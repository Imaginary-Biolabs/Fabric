# Benchmarks

A **Benchmark** binds a released dataset to a **sampler**, **task**, and **metrics**, producing `(X, y)` batches for train/eval loops.

## Config structure

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

## Loaders

```python
from fabric.core.factory import Factory

bench = Factory.benchmark("tests/fixtures/benchmarks/B_mini.yaml")

for X, y in bench.train_loader(batch_size=8):
    ...

for X, y in bench.test_loader(batch_size=8):
    ...
```

Split names follow the dataset's split scheme (`train`, `val`, `test`, or custom keys).

## Evaluation

```python
result = Factory.eval("B_mini", "M_mini", split="test", batch_size=8)
print(result.metrics)  # e.g. {"MAE": 0.42}
```

`Factory.eval` runs the benchmark metrics against model predictions via the configured backend and collater.

## Platform metric specs

Platform benchmarks may include `metric_spec` in asset metadata (primary key, sort direction, fabric metric names). The worker normalizes Fabric metric keys to platform leaderboard storage.

---

**Next:** [Models and training](models-and-training.md) — slots, Trainer, backends.
