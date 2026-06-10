# Run a benchmark locally

Use this guide to iterate on sampler/task/metric configs without the platform.

## 1. Ensure dataset is released

```bash
imaginary dataset release --config tests/fixtures/datasets/D_local.yaml
```

## 2. Load benchmark

```python
from fabric.core.factory import Factory

bench = Factory.benchmark("tests/fixtures/benchmarks/B_mini.yaml")

# Smoke-test loaders
train_batch = next(bench.train_loader(batch_size=2))
test_batch = next(bench.test_loader(batch_size=2))
print("train batch ok", "test batch ok")
```

## 3. Full eval with a trained checkpoint

```python
result = Factory.eval(
    "B_mini",
    "M_mini",
    split="test",
    batch_size=8,
    checkpoint="results/my-run/.../checkpoint.pt",  # optional
)
print(result.metrics)
```

## 4. CLI

```bash
imaginary eval --benchmark B_mini --model M_mini --split test --batch-size 8
```

---

**Next:** [Extend Fabric](extend-fabric.md)
