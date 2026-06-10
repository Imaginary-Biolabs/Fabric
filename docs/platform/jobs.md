# Jobs

Submit **benchmark_eval** jobs to the platform worker queue.

## Submit

```bash
imaginary platform job submit \
  --benchmark B_000010 \
  --model M_000003 \
  --batch-size 8
```

Prerequisites: dataset release and model checkpoint uploaded for the referenced asset versions.

## Poll status

```bash
imaginary platform job status <job-uuid>
imaginary platform job status <job-uuid> --wait
```

## Python API

```python
from fabric.platform.jobs import submit_benchmark_eval, wait_for_job

job = submit_benchmark_eval(
    benchmark_id="B_000010",
    benchmark_version="1",
    model_id="M_000003",
    model_version="1",
    overrides={"batch_size": 8, "split": "test"},
)
finished = wait_for_job(job["id"], timeout_s=600)
print(finished["result"])
```

## Leaderboard

After success, query the API:

```bash
curl "$IMAGINARY_API_BASE/benchmarks/B_000010/leaderboard"
```

---

**Next:** [CLI reference](../cli/commands.md)
