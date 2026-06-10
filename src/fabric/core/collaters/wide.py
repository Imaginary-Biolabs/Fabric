"""Wide-layout collater for pad-free rectangular batches.

Stacks per-scene scalar features into a dense feature matrix without padding.
"""

from __future__ import annotations

import grumpy as gr
from grumpy import GrumpyArray

from fabric.core.collater import CollatedBatch, Collater
from fabric.core.collaters._features import extract_feature_matrix
from fabric.core.data import Data
from fabric.core.scaffold import CollaterSpec
from fabric.utils.errors import CollateError


class WideCollater(Collater):
    """Pad-free rectangular collater for per-scene scalar features.

    Args:
        features: Builtin counts such as ``residue_count`` or scalar dataframe
            columns available in the batch.

    Example:
        >>> collater = WideCollater(features=["residue_count", "atom_count"])
        >>> batch = collater.collate(X, y)
    """

    name = "WideCollater"

    def __init__(self, features: list[str]) -> None:
        self.features = [str(name) for name in features]

    @property
    def spec(self) -> CollaterSpec:
        return CollaterSpec(layout="flat", slots=("features",))

    def collate(self, X: tuple[Data, ...], y: GrumpyArray | None) -> CollatedBatch:
        """Build a dense feature matrix for one loader batch."""
        if not X:
            raise CollateError("Collater received empty inputs")
        data = X[0]
        features = extract_feature_matrix(data, self.features)
        batch_size = int(features.shape(0))
        if y is None:
            targets = gr.array([0.0] * batch_size, dtype=gr.float32)
        else:
            targets = y.flatten().astype(gr.float32, casting="unsafe")
        if int(targets.shape(0)) != batch_size:
            raise CollateError(
                f"Feature batch size {batch_size} does not match target size "
                f"{int(targets.shape(0))}"
            )
        return CollatedBatch(
            features=features,
            y=targets,
            meta={"layout": "flat", "slots": {"features": features}},
        )
