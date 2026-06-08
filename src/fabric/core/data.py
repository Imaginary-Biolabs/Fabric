"""Fabric data model on Grumpy dataframes.

The canonical biological layout is defined by :data:`fabric.utils.constants.SCHEMA`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import grumpy as gr
from grumpy import GrumpyArray, GrumpyDataFrame

from fabric.utils import constants
from fabric.utils.constants import SCHEMA


@dataclass
class Assets:
    """JSON-style sidecar metadata carried through transform pipelines.

    Assets hold provenance, labels, and other non-tabular data that transforms
    may read or update alongside the main :class:`Data` payload.

    Attributes:
        data: Mutable mapping of asset keys to JSON-serializable values.

    Example:
        >>> from fabric.core.data import Assets
        >>> assets = Assets({"source": "mini.pdb"})
        >>> assets.to_dict()
        {'source': 'mini.pdb'}
    """

    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize assets for ``release.json``.

        Returns:
            A shallow copy of the underlying asset mapping.
        """
        return dict(self.data)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> Assets:
        """Restore assets from a release sidecar.

        Args:
            payload: Asset mapping loaded from ``release.json``, or ``None``.

        Returns:
            Assets wrapper around the mapping (empty when ``payload`` is ``None``).
        """
        return cls(data=dict(payload or {}))


@dataclass
class Splits:
    """Named partitioning schemes for dataset indexing.

    Each scheme (for example ``"random_8_2_2"``) maps partition names such as
    ``"train"``, ``"val"``, and ``"test"`` to :class:`~grumpy.GrumpyArray` index
    arrays suitable for Grumpy schema indexing.

    Attributes:
        partitions: Nested mapping ``scheme -> partition -> GrumpyArray``.

    Example:
        >>> import grumpy as gr
        >>> from fabric.core.data import Splits
        >>> splits = Splits(partitions={"demo": {"train": gr.array([0, 1, 2])}})
        >>> splits.to_dict()
        {'demo': {'train': [0, 1, 2]}}
    """

    partitions: dict[str, dict[str, GrumpyArray]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, dict[str, list[Any]]]:
        """Serialize partition indices for ``release.json``.

        Returns:
            Nested plain-Python lists converted from Grumpy arrays.
        """
        return {
            scheme: {name: array.to_list() for name, array in parts.items()}
            for scheme, parts in self.partitions.items()
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Mapping[str, Any]] | None) -> Splits:
        """Restore splits from a release sidecar.

        Args:
            payload: Split mapping loaded from ``release.json``, or ``None``.

        Returns:
            Splits with Grumpy arrays rebuilt from stored lists.
        """
        partitions: dict[str, dict[str, GrumpyArray]] = {}
        for scheme, parts in (payload or {}).items():
            partitions[str(scheme)] = {
                str(name): value if isinstance(value, GrumpyArray) else gr.array(value)
                for name, value in parts.items()
            }
        return cls(partitions=partitions)


class Data:
    """Thin wrapper around a Grumpy dataframe using Fabric :data:`~fabric.utils.constants.SCHEMA`.

    Accepts either a column mapping (constructed with :func:`grumpy.dataframe`) or an
    existing :class:`~grumpy.GrumpyDataFrame`.

    Attributes:
        data: Underlying Grumpy dataframe.

    Example:
        >>> from pathlib import Path
        >>> from fabric.core.parsers.pdb import PdbParser
        >>> data = PdbParser().parse(Path("structure.pdb"))
        >>> len(data.data)
        1
    """

    data: GrumpyDataFrame

    def __init__(self, mapping_or_frame: Mapping[str, Any] | GrumpyDataFrame) -> None:
        """Wrap a mapping or existing Grumpy dataframe.

        Args:
            mapping_or_frame: Column mapping for :func:`grumpy.dataframe`, or a
                dataframe already stored on disk or in memory.
        """
        if isinstance(mapping_or_frame, GrumpyDataFrame):
            self.data = mapping_or_frame
        else:
            self.data = gr.dataframe(dict(mapping_or_frame), schema=SCHEMA)


@dataclass
class Batch:
    """One streamed chunk passed through a transform pipeline.

    Attributes:
        data: Structural payload for the batch.
        meta: Optional per-batch metadata not persisted to Zarr.

    Example:
        >>> from fabric.core.data import Batch, Data
        >>> batch = Batch(data=data)
        >>> batch.meta["shard"] = 0
    """

    data: Data
    meta: dict[str, Any] = field(default_factory=dict)


def iter_batches(data: Data, *, batch_size: int | None = None) -> Iterator[Batch]:
    """Yield scene batches from an in-memory dataframe.

    Args:
        data: Combined structural payload.
        batch_size: Scenes per batch. Defaults to :data:`~fabric.utils.constants.GRUMPY_CHUNK_SIZE`.

    Yields:
        Batches covering consecutive scene ranges in ``data``.
    """
    size = batch_size or constants.GRUMPY_CHUNK_SIZE
    df = data.data
    n = len(df)
    if n == 0:
        return
    if n <= size:
        yield Batch(data=data)
        return
    for start in range(0, n, size):
        stop = min(start + size, n)
        yield Batch(data=Data(df[start:stop]))
