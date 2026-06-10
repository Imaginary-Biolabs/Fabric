"""Molecule-level samplers for dataset split partitions.

Yield batches of scene indices for Grumpy schema indexing.
"""

import random
from collections.abc import Iterator

from fabric.core.data import Splits
from fabric.core.sampler import Sampler
from fabric.utils.errors import SamplerError


class RandomMoleculeSampler(Sampler):
    """Sample scene indices at random within a split partition.

    Each yielded batch is a list of scene-level dataframe indices suitable for
    Grumpy schema indexing.

    Example:
        >>> sampler = RandomMoleculeSampler()
        >>> batches = list(
        ...     sampler.batch_indices(splits, "train", scheme="random_8_2_2", batch_size=2)
        ... )
    """

    name = "RandomMoleculeSampler"

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
        """Yield shuffled or ordered scene batches from one partition."""
        if batch_size < 1:
            raise SamplerError(f"batch_size must be >= 1, got {batch_size}")
        if scheme not in splits.partitions:
            available = ", ".join(sorted(splits.partitions))
            raise SamplerError(
                f"Split scheme '{scheme}' not found; available schemes: {available or '(none)'}"
            )
        parts = splits.partitions[scheme]
        if split_name not in parts:
            available = ", ".join(sorted(parts))
            raise SamplerError(
                f"Split partition '{split_name}' not found in scheme '{scheme}'; "
                f"available partitions: {available or '(none)'}"
            )
        indices = [int(i) for i in parts[split_name].to_list()]
        if not indices:
            raise SamplerError(f"Split partition '{split_name}' in scheme '{scheme}' is empty")

        ordered = list(indices)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(ordered)

        for start in range(0, len(ordered), batch_size):
            batch = ordered[start : start + batch_size]
            if len(batch) < batch_size and drop_last:
                break
            yield batch
