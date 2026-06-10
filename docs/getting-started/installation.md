# Installation

Fabric ships as **`imaginary-fabric`** on PyPI. The console entry point is **`imaginary`**.

## Requirements

- Python ≥ 3.10
- [Grumpy](https://imaginary-biolabs.github.io/Grumpy/) for dataset I/O and Zarr releases
- Optional: PyTorch or TensorFlow for training backends

## PyPI install

```bash
pip install imaginary-fabric
```

### Extras

```bash
# Core orchestration (datasets, benchmarks, CLI)
pip install "imaginary-fabric[core]"

# PyTorch training
pip install "imaginary-fabric[torch]"

# Imaginary platform HTTP client
pip install "imaginary-fabric[platform]"

# Local development
pip install "imaginary-fabric[dev]"
```

## From source

```bash
git clone https://github.com/Imaginary-Biolabs/fabric.git
cd fabric

# Grumpy: PyPI wheel or editable sibling checkout
pip install grumpy
# pip install -e ../grumpy

pip install -e ".[dev,platform,torch]"
```

## Verify

```python
import fabric
from fabric.core.factory import Factory

print(fabric.__version__)
ds = Factory.dataset("tests/fixtures/datasets/D_local.yaml")
print(ds.id)
```

## CLI

After install:

```bash
imaginary --help
imaginary dataset --help
```

---

**Next:** [Quickstart](quickstart.md) — end-to-end release, train, and eval.
