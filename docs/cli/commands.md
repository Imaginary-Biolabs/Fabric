# CLI reference

Entry point: **`imaginary`** (Typer + Rich).

Global options on most commands:

| Flag | Purpose |
|------|---------|
| `--home` | Fabric home directory (overrides `~/.imaginary`) |
| `--help` | Command help |

## `imaginary dataset`

```bash
imaginary dataset release --config PATH [--home HOME]
imaginary dataset info --config PATH
```

Release runs ingest + transforms + Zarr write. Info prints id, version, molecule count, cache path.

## `imaginary train`

```bash
imaginary train --config PATH [--epochs N] [--root DIR] [--home HOME] [--no-progress]
```

Training config references benchmark and model YAML paths.

## `imaginary eval`

```bash
imaginary eval --benchmark ID --model ID [--split SPLIT] [--batch-size N] [--checkpoint PATH]
```

Runs metrics on the chosen split; prints Rich summary.

## `imaginary workflow`

```bash
imaginary workflow run --config PATH [--mode local|hybrid|remote] [--root DIR] [--set key=value]
```

## `imaginary platform`

Requires `[platform]` extra.

```bash
imaginary platform status

imaginary platform upload release --asset D_xxx --version 1 --path /path/to/zarr
imaginary platform upload checkpoint --asset M_xxx --version 1 --path checkpoint.pt

imaginary platform job submit --benchmark B_xxx --model M_xxx
imaginary platform job status JOB_ID [--wait]
```

## Equivalent Python

| CLI | Python |
|-----|--------|
| `dataset release` | `Factory.dataset(path).release()` |
| `train` | `Factory.train(config, epochs=...)` |
| `eval` | `Factory.eval(benchmark, model, ...)` |
| `workflow run` | `Runner(...).run(Factory.workflow(...))` |

See [API reference](../reference/api.md) for full signatures.

---

**Next:** [API reference](../reference/api.md)
