<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/fabric_logo_horizontal_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/fabric_logo_horizontal_light.svg">
    <img src="docs/assets/fabric_logo_horizontal_light.svg" alt="Fabric" width="280">
  </picture>
</p>

<p align="center">
  <strong>Python orchestration for molecular ML on Grumpy</strong>
</p>

<p align="center">
  <a href="https://github.com/Imaginary-Biolabs/fabric/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Imaginary-Biolabs/fabric/ci.yml?branch=main&style=for-the-badge&label=build&color=484240&logo=githubactions&logoColor=E3E1DE" alt="build status" /></a>
  <a href="https://codecov.io/gh/Imaginary-Biolabs/fabric"><img src="https://img.shields.io/codecov/c/github/Imaginary-Biolabs/fabric/main?style=for-the-badge&color=777067&logo=codecov&logoColor=E3E1DE" alt="codecov coverage" /></a>
  <a href="https://github.com/Imaginary-Biolabs/fabric/releases"><img src="https://img.shields.io/badge/version-0.1.0-C8C4BF?style=for-the-badge&logo=python&logoColor=2A2725" alt="version 0.1.0" /></a>
</p>

<p align="center">
  <a href="https://imaginary-biolabs.github.io/fabric/">Documentation</a> ·
  <a href="https://imaginary-biolabs.github.io/Grumpy/">Grumpy</a> ·
  <a href="https://github.com/Imaginary-Biolabs/fabric/issues">Issues</a>
</p>

**Fabric** is the Python orchestration layer for molecular machine learning on [Grumpy](https://github.com/Imaginary-Biolabs/Grumpy): config-driven datasets, benchmarks, training, workflows, and optional Imaginary platform integration for structural biology ML.

```bash
pip install imaginary-fabric
```

For local development (Grumpy required for data features):

```bash
pip install grumpy
pip install -e ".[dev,platform,torch]"
```

```python
from fabric import Settings
from fabric.core.factory import Factory

with Settings(home="~/.imaginary"):
    ds = Factory.dataset("tests/fixtures/datasets/D_local.yaml")
    path = ds.release()
    print(path)
```

```bash
imaginary dataset release --config tests/fixtures/datasets/D_local.yaml
imaginary train --config tests/fixtures/train/train_mini.yaml --epochs 2
imaginary eval --benchmark B_mini --model M_mini
```

## License

Proprietary — Copyright © Imaginary Biolabs GmbH. See [documentation](https://imaginary-biolabs.github.io/fabric/developer/contributing/) for contact details.
