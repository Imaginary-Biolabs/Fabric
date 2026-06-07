"""Transform protocol and composition."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from fabric.core.data import Assets, Batch, Split
from fabric.utils.errors import TransformError


class Transform(ABC):
    """Single step in a dataset transform pipeline."""

    name: str = "Transform"

    @abstractmethod
    def transform(self, batch: Batch, assets: Assets, split: Split) -> tuple[Batch, Assets, Split]:
        raise NotImplementedError

    @property
    def transform_hash(self) -> str:
        payload = f"{type(self).__module__}.{type(self).__name__}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Compose(Transform):
    """Ordered transform chain."""

    name = "Compose"

    def __init__(self, transforms: list[Transform]) -> None:
        if not transforms:
            raise TransformError("Compose requires at least one transform")
        self.transforms = list(transforms)

    def transform(self, batch: Batch, assets: Assets, split: Split) -> tuple[Batch, Assets, Split]:
        current = (batch, assets, split)
        for step in self.transforms:
            batch, assets, split = current
            current = step.transform(batch, assets, split)
        return current

    @property
    def transform_hash(self) -> str:
        if len(self.transforms) == 1:
            return self.transforms[0].transform_hash
        payload = "|".join(step.transform_hash for step in self.transforms)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    def __repr__(self) -> str:
        names = " -> ".join(repr(step) for step in self.transforms)
        return f"Compose([{names}])"
