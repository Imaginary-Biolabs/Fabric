# grumpy

Minimal Python package ready to publish to PyPI.

## Install (after publishing)

```bash
pip install grumpy
```

## Usage

```python
import grumpy
print(grumpy.__version__)
```

## Development

Recommended: use a virtual environment (`python -m venv .venv && source .venv/bin/activate`).

### Build the distribution

```bash
python -m pip install --upgrade build
python -m build
```

This creates files in `dist/` (a wheel and an sdist).

### Upload to TestPyPI first

```bash
python -m pip install --upgrade twine
twine upload --repository testpypi dist/*
```

Verify install from TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple grumpy
```

### Upload to PyPI

```bash
twine upload dist/*
```

### Makefile shortcuts

```bash
make build
make publish-test
make publish
```


