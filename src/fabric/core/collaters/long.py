from __future__ import annotations

import numpy as np
from grumpy import GrumpyArray

from fabric.core.collater import CollatedBatch, Collater
from fabric.core.collaters._features import extract_feature_matrix
from fabric.core.data import Data
from fabric.utils.errors import CollateError


class LongCollater(Collater):
    """Long-layout collater with per-row scene indices.

    For scene-level scalar features this matches :class:`WideCollater` but also
    records ``scene_index`` metadata used by graph-style backends.

    Args:
        features: Builtin counts or scalar dataframe columns.
    """

    name = "LongCollater"

    def __init__(self, features: list[str]) -> None:
        self.features = [str(name) for name in features]

    def collate(self, X: tuple[Data, ...], y: GrumpyArray | None) -> CollatedBatch:
        """Build a long-layout batch with scene indices."""
        if not X:
            raise CollateError("Collater received empty inputs")
        data = X[0]
        features = extract_feature_matrix(data, self.features)
        batch_size = features.shape[0]
        scene_index = np.arange(batch_size, dtype=np.int64)
        if y is None:
            targets = np.zeros((batch_size,), dtype=np.float32)
        else:
            targets = np.asarray(y.to_list(), dtype=np.float32).reshape(-1)
        if targets.shape[0] != batch_size:
            raise CollateError(
                f"Feature batch size {batch_size} does not match target size {targets.shape[0]}"
            )
        return CollatedBatch(
            features=features,
            y=targets,
            scene_index=scene_index,
            meta={"layout": "long"},
        )
