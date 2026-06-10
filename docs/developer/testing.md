# Testing

Fabric uses **phase toll tests** — integration tests that gate each implementation phase.

## Run tests

```bash
# All phase tolls (Makefile)
make test

# Single phase
pytest tests/tolls/test_phase_2.py -x --tb=short -q

# With coverage (CI)
pytest tests/tolls/ --cov=fabric --cov-report=term-missing --cov-report=xml
```

## Phase map

| Phase | Test file | Covers |
|-------|-----------|--------|
| 1 | `test_phase_1.py` | Settings, config, hashing |
| 2 | `test_phase_2.py` | Dataset release, transforms |
| 3 | `test_phase_3.py` | Benchmarks, samplers, tasks |
| 4 | `test_phase_4.py` | Trainer, backends |
| 5 | `test_phase_5.py` | Models, slots |
| 6 | `test_phase_6.py` | Workflows, Runner |
| 7 | `test_phase_7.py` | CLI |
| 8 | `test_phase_8.py` | Platform client |

## Fixtures

`tests/fixtures/` holds minimal YAML configs, structures, and workflows. Prefer extending fixtures over inline strings in tests.

## Lint

```bash
make lint
# ruff check src tests
```

## CI

GitHub Actions runs ruff, phase tolls, coverage upload to Codecov, and MkDocs build on `main`.

---

**Next:** [Contributing](contributing.md)
