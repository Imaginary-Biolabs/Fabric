# Home

Fabric is the **Python orchestration layer** for molecular machine learning on [Grumpy](https://imaginary-biolabs.github.io/Grumpy/). It turns YAML configs into reproducible pipelines: ingest structures, release versioned Zarr datasets, define benchmarks, train models, run evaluation, and compose multi-step workflows. Optional **platform** integration talks to the Imaginary API for registry assets, blob uploads, and remote jobs.

This guide walks from installation through datasets, benchmarks, training, workflows, and the `imaginary` CLI.

## Installation

Fabric requires **Python ≥ 3.10** and **Grumpy** for data I/O.

### PyPI

```bash
pip install imaginary-fabric
```

Extras:

| Extra | Purpose |
|-------|---------|
| `core` | Datasets, benchmarks, metrics (default install surface) |
| `torch` | PyTorch + Lightning backend |
| `platform` | Imaginary API client (`httpx`) |
| `dev` | pytest, ruff, torch for local development |

```bash
pip install "imaginary-fabric[core,torch,platform]"
```

### From source

```bash
git clone https://github.com/Imaginary-Biolabs/fabric.git
cd fabric
pip install -e ../grumpy          # Grumpy from sibling clone or PyPI
pip install -e ".[dev,platform]"
```

Verify:

```python
import fabric
print(fabric.__version__)
```

## A tour of the main features

### Settings and cache paths

Fabric stores releases under a configurable home directory (default `~/.imaginary`):

```python
from fabric import Settings

with Settings(home="/tmp/imaginary"):
    print(Settings.transformed_path())
```

### Dataset release

Load a YAML recipe, ingest structures, apply transforms, write a Grumpy Zarr release:

```python
from fabric.core.factory import Factory

ds = Factory.dataset("tests/fixtures/datasets/D_local.yaml")
path = ds.release()
print(path)  # Zarr directory with grumpy.json + release.json
```

### Benchmark loaders

Benchmarks bind a dataset to sampler, task, and metrics:

```python
bench = Factory.benchmark("tests/fixtures/benchmarks/B_mini.yaml")
batch = next(bench.train_loader(batch_size=4))
```

### Training and evaluation

```python
Factory.train("tests/fixtures/train/train_mini.yaml", epochs=2, root="results/train")
result = Factory.eval("B_mini", "M_mini", split="test", batch_size=8)
print(result.metrics)
```

### CLI

The `imaginary` command wraps the same operations:

```bash
imaginary dataset release --config tests/fixtures/datasets/D_local.yaml
imaginary train --config tests/fixtures/train/train_mini.yaml --epochs 2
imaginary eval --benchmark B_mini --model M_mini
```

### Platform (optional)

With the platform extra and API credentials:

```bash
export IMAGINARY_API_BASE=https://api.imaginary.bio/v1
export IMAGINARY_API_KEY=img_...
imaginary platform status
```

---

**Next:** [Installation](getting-started/installation.md) — extras, Grumpy dependency, and development setup.
