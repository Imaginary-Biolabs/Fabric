"""Sampler protocol for split-aware batch indexing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from fabric.core.data import Splits


class Sampler(ABC):
    """Select scene indices from a dataset split for benchmark loaders.

    Example:
        >>> from fabric.core.samplers import RandomMoleculeSampler
        >>> sampler = RandomMoleculeSampler()
        >>> list(sampler.batch_indices(splits, "train", scheme="demo", batch_size=2))
    """

    name: str = "Sampler"

    @abstractmethod
    def batch_indices(
        self,
        splits: Splits,
        split_name: str,
        *,
        scheme: str,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 0,
    ) -> Iterator[list[int]]:
        """Yield scene-index batches for one partition.

        Args:
            splits: Dataset partition indices.
            split_name: Partition name such as ``"train"``, ``"val"``, or ``"test"``.
            scheme: Named split scheme in ``splits.partitions``.
            batch_size: Number of scene indices per batch.
            shuffle: When ``True``, shuffle indices before batching.
            drop_last: When ``True``, omit the final partial batch.
            seed: Random seed used when ``shuffle`` is ``True``.

        Yields:
            Lists of scene-level dataframe indices.

        Raises:
            SamplerError: If the scheme or partition is missing or empty.
        """
        raise NotImplementedError
        yield  # pragma: no cover

    def hash_params(self) -> dict[str, Any]:
        """Return constructor parameters for provenance hashing."""
        return {}
