# Extend Fabric

Fabric extension points are **registry-based**: implement a class, reference it by name in YAML.

## Custom transform

```python
from fabric.core.transform import Transform

class MyTransform(Transform):
    """Example transform — document and register for YAML use."""

    def transform_batches(self, batches):
        for batch in batches:
            yield batch  # modify in place or return new batches
```

Register in your project's import path so `build_transform` can resolve `MyTransform` from config.

## Custom external adapter

Implement `External.load()` yielding `(batches, assets, splits)`. See `core/externals/local.py`.

## Custom metric

Subclass metric base in `core/metric.py`; reference in benchmark YAML:

```yaml
metrics:
  - MyMetric: { reduction: mean }
```

## Custom workflow node

```python
from fabric.core.workflow_nodes.base import register_executor

@register_executor("my_op")
def run_my_op(ctx, node, inputs):
    return {"output": inputs["value"]}
```

## Platform client

Platform modules require `pip install "imaginary-fabric[platform]"`. See [Platform overview](../platform/overview.md).

---

**Next:** [Platform overview](../platform/overview.md)
