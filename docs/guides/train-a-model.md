# Train a model

## Prerequisites

- Released dataset referenced by benchmark config
- `imaginary-fabric[torch]` for PyTorch backend

## Steps

1. **Check train config** — `tests/fixtures/train/train_mini.yaml` references benchmark + model YAMLs

2. **Train**

```bash
imaginary train \
  --config tests/fixtures/train/train_mini.yaml \
  --epochs 5 \
  --root results/my-run \
  --no-progress
```

3. **Artifacts**

Checkpoints and logs appear under `{root}/{run_id}/` via `DiskLogger`.

4. **Evaluate**

```bash
imaginary eval --benchmark B_mini --model M_mini --split test
```

---

**Next:** [Run a benchmark locally](run-a-benchmark-locally.md)
