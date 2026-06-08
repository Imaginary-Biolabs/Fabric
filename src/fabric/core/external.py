"""External data source protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from fabric.core.data import Assets, Batch, Splits

ExternalLoadResult = tuple[Iterator[Batch], Assets, Splits]
"""Return type of :meth:`External.load`."""


class External(ABC):
    """Load canonical Fabric data from an upstream source.

    Externals return a generator of :class:`~fabric.core.data.Batch` payloads
    sized for Grumpy Zarr chunk writes
    (:data:`~fabric.utils.constants.GRUMPY_CHUNK_SIZE` scenes per batch), plus
    ingest-time :class:`~fabric.core.data.Assets` and
    :class:`~fabric.core.data.Splits` returned once for the full source.

    Example:
        >>> from fabric.core.externals.local import LocalAdapter
        >>> from pathlib import Path
        >>> adapter = LocalAdapter(
        ...     ["structures/*.pdb"],
        ...     config_dir=Path("datasets"),
        ... )
        >>> batches, assets, splits = adapter.load()
        >>> next(batches)
    """

    @abstractmethod
    def load(self) -> ExternalLoadResult:
        """Stream ingest batches plus sidecar metadata for the full source.

        Parsing may run in parallel according to
        :meth:`~fabric.utils.settings.Settings.workers`.

        Returns:
            ``(batches, assets, splits)`` where ``batches`` yields up to
            :data:`~fabric.utils.constants.GRUMPY_CHUNK_SIZE` scenes per item.

        Raises:
            ExternalError: If the source is empty or misconfigured.
        """
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> dict[str, object]:
        """Return a JSON-serializable descriptor for provenance.

        Returns:
            Mapping stored under the ``external`` key in ``release.json``.
        """
        raise NotImplementedError
