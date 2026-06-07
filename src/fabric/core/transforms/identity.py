from __future__ import annotations

from fabric.core.data import Assets, Batch, Split
from fabric.core.transform import Transform


class Identity(Transform):
    """No-op transform."""

    name = "Identity"

    def transform(self, batch: Batch, assets: Assets, split: Split) -> tuple[Batch, Assets, Split]:
        return batch, assets, split
