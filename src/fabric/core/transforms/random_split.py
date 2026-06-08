from __future__ import annotations

import random
from typing import Any

import grumpy as gr

from fabric.core.data import Assets, Splits
from fabric.core.transform import Transform
from fabric.utils.errors import TransformError


class RandomSplit(Transform):
    """Assign train/val/test indices under a named split scheme.

    Indices are scene-level positions in the released Grumpy dataframe and are
    stored under ``splits.partitions[scheme]``.

    Args:
        n_train: Number of training indices.
        n_val: Number of validation indices.
        n_test: Number of test indices.
        seed: Random seed for reproducible shuffling.
        scheme: Split scheme name. Defaults to ``"random_{n_train}_{n_val}_{n_test}"``.

    Raises:
        TransformError: If the dataset has fewer scenes than requested partitions.

    Example:
        >>> step = RandomSplit(8, 2, 2, seed=0)
        >>> splits = step.transform_splits(Splits(), n_scenes=12)
        >>> "train" in splits.partitions[step.scheme]
        True
    """

    name = "RandomSplit"

    def __init__(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
        *,
        seed: int = 0,
        scheme: str | None = None,
    ) -> None:
        """Configure partition sizes and optional scheme name."""
        self.n_train = int(n_train)
        self.n_val = int(n_val)
        self.n_test = int(n_test)
        self.seed = int(seed)
        self.scheme = scheme or f"random_{self.n_train}_{self.n_val}_{self.n_test}"
        self._indices: dict[str, list[int]] = {}

    def hash_params(self) -> dict[str, Any]:
        """Return constructor parameters for hashing."""
        return {
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "seed": self.seed,
            "scheme": self.scheme,
        }

    def fit(self, *, n_scenes: int, assets: Assets, splits: Splits) -> None:
        """Shuffle scene indices and store train/val/test partitions.

        Args:
            n_scenes: Total scenes available for partitioning.
            assets: Unused.
            splits: Unused during ``fit``.

        Raises:
            TransformError: If insufficient scenes exist for the requested split sizes.
        """
        total = self.n_train + self.n_val + self.n_test
        if n_scenes < total:
            raise TransformError(
                f"RandomSplit needs {total} molecules but found {n_scenes}; "
                "add more source structures or lower n_train/n_val/n_test."
            )
        indices = list(range(n_scenes))
        rng = random.Random(self.seed)
        rng.shuffle(indices)
        self._indices = {
            "train": indices[: self.n_train],
            "val": indices[self.n_train : self.n_train + self.n_val],
            "test": indices[self.n_train + self.n_val : total],
        }

    def transform_splits(self, splits: Splits, *, n_scenes: int) -> Splits:
        """Write shuffled indices into ``splits.partitions``.

        Args:
            splits: Splits to update.
            n_scenes: Total scene count (used for ``fit`` when indices are missing).

        Returns:
            Splits with populated partition scheme.
        """
        if not self._indices:
            self.fit(n_scenes=n_scenes, assets=Assets(), splits=splits)
        splits.partitions[self.scheme] = {
            name: gr.array(values) for name, values in self._indices.items()
        }
        return splits

    def __repr__(self) -> str:
        """Return a concise constructor-style representation."""
        return (
            f"RandomSplit(n_train={self.n_train}, n_val={self.n_val}, "
            f"n_test={self.n_test}, seed={self.seed}, scheme={self.scheme!r})"
        )
