# Registry

Fetch dataset, benchmark, model, and workflow configs from the Imaginary API.

## Fetch asset version

```python
from fabric.platform.registry import fetch_asset_version

payload = fetch_asset_version("B_000010", "1", cache=True)
print(payload["config_yaml"])
print(payload.get("meta", {}))
```

With `cache=True`, YAML is written to:

```
~/.imaginary/registry/B_000010/1/config.yaml
```

`Factory` resolves platform ids from this cache when local fixtures are absent.

## List public assets

```python
from fabric.platform.registry import list_assets

for item in list_assets(kind="benchmark"):
    print(item["id"], item.get("title"))
```

## Cached path helper

```python
from fabric.core.factory import Factory
from fabric.platform.registry import cached_config_path

path = cached_config_path("B_000010", "1")
bench = Factory.benchmark(path)
```

---

**Next:** [Uploads](uploads.md)
