<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/fabric_logo_horizontal_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/fabric_logo_horizontal_light.svg">
    <img src="docs/assets/fabric_logo_horizontal_light.svg" alt="Fabric" width="280">
  </picture>
</p>

<p align="center">
  <strong>Python orchestration for molecular ML pipelines</strong>
</p>

<p align="center">
  <a href="https://github.com/Imaginary-Biolabs/fabric/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Imaginary-Biolabs/fabric/ci.yml?branch=main&style=for-the-badge&label=build&color=484240&logo=githubactions&logoColor=E3E1DE" alt="build status" /></a>
  <a href="https://github.com/Imaginary-Biolabs/fabric/releases"><img src="https://img.shields.io/badge/version-0.1.0-C8C4BF?style=for-the-badge&logo=python&logoColor=2A2725" alt="version 0.1.0" /></a>
</p>

<p align="center">
  <a href="https://github.com/Imaginary-Biolabs/Grumpy">Grumpy</a> ·
  <a href="docs/">Documentation</a> ·
  <a href="https://github.com/Imaginary-Biolabs/fabric/issues">Issues</a>
</p>

**Fabric** is the Python orchestration layer for molecular machine learning on [Grumpy](https://github.com/Imaginary-Biolabs/Grumpy): config-driven datasets, benchmarks, training, and platform registry integration for structural biology ML.

```bash
pip install imaginary-fabric
```

For local development (Grumpy required for data features):

```bash
pip install -e ../grumpy
pip install -e ".[core]"
```

```python
from fabric import Settings
from fabric.utils.config import load_config

with Settings(home="~/.imaginary"):
    cfg = load_config("datasets/D_local.yaml")
```

## License

Proprietary — Copyright © Imaginary Biolabs GmbH.
