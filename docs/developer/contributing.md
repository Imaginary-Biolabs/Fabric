# Contributing

## Development setup

```bash
git clone https://github.com/Imaginary-Biolabs/fabric.git
cd fabric
pip install -e ../grumpy   # or pip install grumpy
pip install -e ".[dev,platform,torch]"
```

## Before opening a PR

```bash
make lint
make test
mkdocs build --strict    # if you changed docs or docstrings
```

## Docs

- Narrative pages: `docs/`
- API reference: generated from NumPy-style docstrings via mkdocstrings
- Include **Examples** sections in public classes and functions

## Commit style

Follow existing history: imperative subject, phase or area prefix when relevant.

## License

Fabric is **proprietary** — Copyright © Imaginary Biolabs GmbH. Contact [hello@imaginary.bio](mailto:hello@imaginary.bio) for licensing questions.

---

**Next:** [Home](../index.md)
