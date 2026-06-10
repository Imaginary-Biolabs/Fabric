# Authentication

Platform clients authenticate with an **API key** header.

## Environment variables

```bash
export IMAGINARY_API_KEY=img_dev_...
export IMAGINARY_API_BASE=http://localhost:8080/v1
```

Optional:

```bash
export IMAGINARY_ORG_ID=...   # org-scoped operations
```

## Credentials file

Persist credentials under `~/.imaginary/credentials.yaml`:

```python
from fabric.platform.client import save_credentials

save_credentials(
    api_key_value="img_dev_...",
    api_base_value="http://localhost:8080/v1",
)
```

Load order: explicit constructor args → environment → credentials file.

## Scopes

API keys carry scopes such as:

| Scope | Allows |
|-------|--------|
| `registry:read` | Fetch public/private assets |
| `registry:write` | Create assets and versions |
| `blobs:write` | Upload sessions and manifests |
| `jobs:write` | Submit benchmark jobs |

Dev backend keys from `make dev-key` include all scopes.

## Website auth (future)

The Imaginary website will use **Supabase JWT** for interactive sessions. The CLI continues to use API keys. Both hit the same FastAPI authorization layer.

---

**Next:** [Registry](registry.md)
