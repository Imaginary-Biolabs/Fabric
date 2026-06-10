# Quickstart

This walkthrough uses fixtures shipped in the repository. Adjust paths if you installed from PyPI only (clone the repo for fixtures, or use your own YAML configs).

## 1. Configure Fabric home

```python
from fabric import Settings

# Optional: scope all paths for this session
with Settings(home="~/.imaginary"):
    ...
```

Or pass `--home` to CLI commands.

## 2. Release a dataset

**Python:**

```python
from fabric.core.factory import Factory

ds = Factory.dataset("tests/fixtures/datasets/D_local.yaml")
release_path = ds.release()
print(f"Released to {release_path}")
```

**CLI:**

```bash
imaginary dataset release --config tests/fixtures/datasets/D_local.yaml
```

The release directory contains Grumpy Zarr arrays plus `release.json` with `transform_hash` and split metadata.

## 3. Run a benchmark loader

```python
bench = Factory.benchmark("tests/fixtures/benchmarks/B_mini.yaml")
for batch in bench.train_loader(batch_size=2):
    X, y = batch
    break
print(type(X), y.shape if hasattr(y, "shape") else y)
```

## 4. Train a model

```python
Factory.train(
    "tests/fixtures/train/train_mini.yaml",
    epochs=2,
    root="results/quickstart",
)
```

**CLI:**

```bash
imaginary train --config tests/fixtures/train/train_mini.yaml --epochs 2 --root results/quickstart
```

## 5. Evaluate

```python
result = Factory.eval("B_mini", "M_mini", split="test", batch_size=4)
print(result.metrics)
```

**CLI:**

```bash
imaginary eval --benchmark B_mini --model M_mini --split test
```

## 6. Platform (optional)

```bash
pip install "imaginary-fabric[platform]"
export IMAGINARY_API_BASE=http://localhost:8080/v1
export IMAGINARY_API_KEY=img_dev_...

imaginary platform status
python -c "from fabric.platform.registry import fetch_asset_version; print(fetch_asset_version('B_000010','1')['config_yaml'][:80])"
```

---

**Next:** [Project layout](project-layout.md) — cache tiers and config file conventions.
