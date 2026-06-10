"""Sequence-length filter transform for dataset releases.

Drops scenes whose longest chain exceeds a residue-count threshold and remaps
split indices to retained scenes.
"""

from collections.abc import Iterator
from typing import Any

import grumpy as gr

from fabric.core.data import Batch, Data, Splits
from fabric.core.transform import Transform


class FilterSequenceLength(Transform):
    """Drop scenes whose longest chain exceeds ``max_len`` residues.

    Split indices in all schemes are remapped to the retained scenes.

    Args:
        max_len: Maximum allowed residues per chain.

    Example:
        >>> step = FilterSequenceLength(max_len=512)
        >>> batches = step.transform_batches(iter([batch]))
    """

    name = "FilterSequenceLength"

    @staticmethod
    def _max_chain_residue_count(scene) -> int:
        """Longest chain length for one scene via Grumpy shape metadata."""
        counts = scene.shape(dim="residue")
        if isinstance(counts, int):
            return counts
        flat = counts.flatten().to_list()
        return max(flat) if flat else 0

    def __init__(self, max_len: int) -> None:
        """Store the maximum allowed residues per chain."""
        self.max_len = int(max_len)
        self._kept_global: list[int] = []

    def hash_params(self) -> dict[str, Any]:
        """Return constructor parameters for hashing."""
        return {"max_len": self.max_len}

    def transform_batches(self, batches: Iterator[Batch]) -> Iterator[Batch]:
        """Filter scenes in each streamed batch.

        Args:
            batches: Input batch stream.

        Yields:
            Batches containing only scenes that pass the length filter.
        """
        self._kept_global = []
        global_offset = 0
        for batch in batches:
            kept_local: list[int] = []
            df = batch.data.data
            for local_idx in range(len(df)):
                global_idx = global_offset + local_idx
                if self._max_chain_residue_count(df[local_idx]) <= self.max_len:
                    kept_local.append(local_idx)
                    self._kept_global.append(global_idx)
            global_offset += len(df)
            if kept_local:
                yield Batch(data=Data(df[kept_local]), meta=dict(batch.meta))

    def transform_splits(self, splits: Splits, *, n_scenes: int) -> Splits:
        """Remap partition indices to retained global scene positions.

        Args:
            splits: Splits to remap after :meth:`transform_batches` has run.
            n_scenes: Unused; present for a uniform transform API.

        Returns:
            Splits with indices restricted to retained scenes.
        """
        if not splits.partitions:
            return splits
        old_to_new = {old: new for new, old in enumerate(self._kept_global)}
        splits.partitions = {
            scheme: {
                name: gr.array([old_to_new[i] for i in array.to_list() if i in old_to_new])
                for name, array in parts.items()
            }
            for scheme, parts in splits.partitions.items()
        }
        return splits

    def __repr__(self) -> str:
        """Return a concise constructor-style representation."""
        return f"FilterSequenceLength(max_len={self.max_len})"
