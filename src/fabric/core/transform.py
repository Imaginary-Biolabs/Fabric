"""Transform protocol and composition."""

from __future__ import annotations

from abc import ABC
from collections.abc import Iterator
from typing import Any

from grumpy import GrumpyArray

from fabric.core.data import Assets, Batch, Splits
from fabric.utils.errors import TransformError
from fabric.utils.hashing import config_hash


class Transform(ABC):
    """Single step in a dataset transform pipeline.

    Subclasses override only the sub-methods they need:

    * :meth:`transform_assets` — whole ingest metadata
    * :meth:`transform_splits` — whole partition indices
    * :meth:`transform_batches` — streamed structural batches
    * :meth:`inverse_transform` — inverse map for model predictions

    Example:
        >>> from fabric.core.transforms import Identity
        >>> list(Identity().transform_batches(iter([batch])))
    """

    name: str = "Transform"

    def fit(self, *, n_scenes: int, assets: Assets, splits: Splits) -> None:
        """Optional pre-pass before transforms (for example split planning).

        Args:
            n_scenes: Total scene count across the full dataset.
            assets: Ingest-time assets.
            splits: Ingest-time splits.
        """
        return None

    def transform_assets(self, assets: Assets) -> Assets:
        """Transform ingest assets as a whole.

        Args:
            assets: Sidecar metadata for the full source.

        Returns:
            Updated assets (defaults to unchanged input).
        """
        return assets

    def transform_splits(self, splits: Splits, *, n_scenes: int) -> Splits:
        """Transform partition indices as a whole.

        Args:
            splits: Partition schemes to update.
            n_scenes: Total scene count before this step's batch pass.

        Returns:
            Updated splits (defaults to unchanged input).
        """
        return splits

    def transform_batches(self, batches: Iterator[Batch]) -> Iterator[Batch]:
        """Transform structural batches in a streaming fashion.

        Args:
            batches: Stream of ingest-sized batches.

        Yields:
            Transformed batches in order.
        """
        yield from batches

    def inverse_transform(self, values: GrumpyArray) -> GrumpyArray:
        """Inverse-transform model predictions (for example unscaling).

        Args:
            values: Forward-transformed prediction array.

        Returns:
            Values mapped back to the original target space.

        Raises:
            NotImplementedError: If this transform does not define an inverse.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement inverse_transform"
        )

    def hash_params(self) -> dict[str, Any]:
        """Parameters included in :attr:`hash`.

        Returns:
            JSON-serializable mapping of constructor parameters.
        """
        return {}

    @property
    def hash(self) -> str:
        """Stable ``sha256:`` digest for this transform configuration.

        Returns:
            Canonical hash string used in release cache paths.
        """
        payload = {
            "type": f"{type(self).__module__}.{type(self).__name__}",
            "params": self.hash_params(),
        }
        return config_hash(payload)

    def __repr__(self) -> str:
        """Return the transform class name."""
        return f"{type(self).__name__}()"


class Compose(Transform):
    """Ordered transform chain applied during dataset release.

    Args:
        transforms: Non-empty list of transforms executed in order.

    Raises:
        TransformError: If ``transforms`` is empty.

    Example:
        >>> from fabric.core.transforms import RandomSplit, FilterSequenceLength
        >>> chain = Compose([RandomSplit(8, 2, 2), FilterSequenceLength(512)])
    """

    name = "Compose"

    def __init__(self, transforms: list[Transform]) -> None:
        """Store an ordered list of transforms to run in sequence."""
        if not transforms:
            raise TransformError("Compose requires at least one transform")
        self.transforms = list(transforms)
        self._splits: Splits | None = None

    @property
    def splits(self) -> Splits:
        """Splits after :meth:`apply` has been fully consumed."""
        if self._splits is None:
            raise TransformError("Compose splits are unavailable until apply() finishes")
        return self._splits

    def apply(
        self,
        batches: Iterator[Batch],
        assets: Assets,
        splits: Splits,
        *,
        n_scenes: int,
    ) -> tuple[Iterator[Batch], Assets, Splits]:
        """Run the transform chain on a streamed batch ingest.

        Args:
            batches: Stream of structural batches from an external or parent ingest.
            assets: Ingest-time assets for the full source.
            splits: Ingest-time splits for the full source.
            n_scenes: Total scene count across all batches.

        Returns:
            ``(batches, assets, splits)`` where ``batches`` is a lazy pipeline.
            ``splits`` on the returned tuple is the initial value; the final splits
            are available on :attr:`splits` after the batch iterator is exhausted.
        """
        for step in self.transforms:
            step.fit(n_scenes=n_scenes, assets=assets, splits=splits)

        for step in self.transforms:
            assets = step.transform_assets(assets)

        self._splits = splits

        def pipeline() -> Iterator[Batch]:
            assert self._splits is not None
            stream: Iterator[Batch] = batches
            for step in self.transforms:
                buffered: list[Batch] = []
                for batch in step.transform_batches(stream):
                    buffered.append(batch)
                self._splits = step.transform_splits(self._splits, n_scenes=n_scenes)
                stream = iter(buffered)
            yield from stream

        return pipeline(), assets, splits

    def inverse_transform(self, values: GrumpyArray) -> GrumpyArray:
        """Apply child inverses in reverse order, skipping undefined steps."""
        for step in reversed(self.transforms):
            try:
                values = step.inverse_transform(values)
            except NotImplementedError:
                continue
        return values

    @property
    def hash(self) -> str:
        """Hash of the full transform chain.

        Returns:
            Single-step hash when the chain has one element; otherwise a combined hash.
        """
        if len(self.transforms) == 1:
            return self.transforms[0].hash
        payload = [step.hash for step in self.transforms]
        return config_hash(payload)

    def __repr__(self) -> str:
        """Return a chain of child transform representations."""
        names = " -> ".join(repr(step) for step in self.transforms)
        return f"Compose([{names}])"
