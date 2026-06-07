# imaginary-fabric

Python orchestration layer for molecular ML on [Grumpy](https://github.com/Imaginary-Biolabs/grumpy).

## Install

```bash
# Grumpy is required for data features (install from sibling repo in the monorepo):
pip install -e ../grumpy

pip install -e ".[core]"
```

Platform client (Phase 8+):

```bash
pip install -e ".[platform]"
```

Development:

```bash
pip install -e ".[dev]"
```

## Usage

```python
from fabric import Settings

assert Settings.home == Settings.dataset_path  # ~/.imaginary

with Settings(home="/tmp/my-imaginary"):
    assert Settings.dataset_path == Path("/tmp/my-imaginary")
```

## Development

```bash
make install
make lint
make test
```

Phased build plan: `.ai/project/fabric-impl-plan.md` (in the Imaginary monorepo).
