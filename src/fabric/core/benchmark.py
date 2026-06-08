"""Benchmark binds dataset, sampler, task, and metrics into split loaders."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path

from grumpy import GrumpyArray
from omegaconf import DictConfig

from fabric.core.data import Splits
from fabric.core.dataset import Dataset
from fabric.core.metric import MultiMetric
from fabric.core.metrics import build_metrics
from fabric.core.result import Result
from fabric.core.sampler import Sampler
from fabric.core.samplers import build_sampler
from fabric.core.task import Task, TaskResult
from fabric.core.tasks import build_task
from fabric.utils.errors import BenchmarkError, ConfigError, DatasetError
from fabric.utils.io import loader_description, track

LoaderResult = Iterator[TaskResult]
"""Type alias for ``(X, y)`` batches yielded by split loaders."""

_LOADER_SUFFIX = "_loader"


def _split_batch_count(
    splits: Splits,
    split_name: str,
    *,
    scheme: str,
    batch_size: int,
    drop_last: bool,
) -> int:
    """Return the number of batches a split partition will yield."""
    parts = splits.partitions[scheme][split_name]
    n = len(parts.to_list())
    if drop_last:
        return n // batch_size
    if n == 0:
        return 0
    return (n + batch_size - 1) // batch_size


class Benchmark:
    """Config-driven evaluation harness without a training backend.

    Loaders are named ``{split}_loader`` for each partition in the configured
    split scheme (for example ``train_loader``, ``val_loader``) and yield
    ``(X, y)`` tuples from the configured task.

    Args:
        cfg: Benchmark YAML loaded as OmegaConf.
        config_path: Path to the benchmark YAML file.

    Example:
        >>> from fabric.core.factory import Factory
        >>> bench = Factory.benchmark("tests/fixtures/benchmarks/B_mini.yaml")
        >>> for X, y in bench.train_loader(batch_size=2):
        ...     break
    """

    def __init__(self, cfg: DictConfig, *, config_path: Path) -> None:
        self.cfg = cfg
        self.config_path = Path(config_path)
        self.id = str(cfg.get("id") or self.config_path.stem)
        self.split_scheme = str(cfg.get("split_scheme") or "")
        self.sampler: Sampler = build_sampler(cfg.get("sampler"))
        self.task: Task = build_task(cfg.get("task"))
        self.metrics: MultiMetric = build_metrics(cfg.get("metrics"))
        self._dataset: Dataset | None = None
        self._split_names: list[str] | None = None

    @property
    def dataset(self) -> Dataset:
        """Released dataset referenced by the benchmark config."""
        self._ensure_dataset()
        assert self._dataset is not None
        return self._dataset

    def _ensure_dataset(self) -> None:
        """Load and release the configured dataset once."""
        if self._dataset is not None:
            return
        dataset_ref = self.cfg.get("dataset")
        if not dataset_ref:
            raise BenchmarkError(f"{self.id} → (missing dataset)")
        try:
            from fabric.core.factory import Factory

            dataset = Factory.dataset(str(dataset_ref))
        except (ConfigError, DatasetError) as exc:
            raise BenchmarkError(f"{self.id} → {dataset_ref} → {exc}") from exc
        except Exception as exc:
            raise BenchmarkError(f"{self.id} → {dataset_ref} → {exc}") from exc
        try:
            dataset.release()
        except DatasetError as exc:
            raise BenchmarkError(f"{self.id} → {dataset_ref} → {exc}") from exc
        self._dataset = dataset
        if not self.split_scheme:
            schemes = sorted(dataset.splits.partitions)
            if len(schemes) != 1:
                raise BenchmarkError(
                    f"{self.id} → {dataset_ref} → split_scheme is required when the "
                    f"dataset exposes multiple schemes: {', '.join(schemes)}"
                )
            self.split_scheme = schemes[0]

    def split_names(self) -> list[str]:
        """Return partition names for the configured split scheme."""
        self._ensure_dataset()
        assert self._dataset is not None
        if self._split_names is None:
            parts = self._dataset.splits.partitions.get(self.split_scheme)
            if parts is None:
                available = ", ".join(sorted(self._dataset.splits.partitions))
                raise BenchmarkError(
                    f"{self.id} → {self._dataset.id} → split scheme "
                    f"'{self.split_scheme}' not found; available: {available or '(none)'}"
                )
            self._split_names = sorted(parts)
        return list(self._split_names)

    def _verify_split(self, split_name: str) -> None:
        """Ensure a partition exists before exposing a split loader.

        Raises:
            BenchmarkError: If the split is missing from the configured scheme.
        """
        self._ensure_dataset()
        assert self._dataset is not None
        parts = self._dataset.splits.partitions.get(self.split_scheme)
        if parts is None:
            available = ", ".join(sorted(self._dataset.splits.partitions))
            raise BenchmarkError(
                f"{self.id} → {self._dataset.id} → split scheme "
                f"'{self.split_scheme}' not found; available: {available or '(none)'}"
            )
        if split_name not in parts:
            available = ", ".join(sorted(parts))
            raise BenchmarkError(
                f"{self.id} → {self._dataset.id} → split '{split_name}' not found in "
                f"scheme '{self.split_scheme}'; available: {available or '(none)'}"
            )

    def __getattr__(self, name: str) -> Callable[..., LoaderResult]:
        """Resolve ``{split}_loader`` callables for dataset partition names."""
        if not name.endswith(_LOADER_SUFFIX):
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        split_name = name[: -len(_LOADER_SUFFIX)]
        if not split_name:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        self._verify_split(split_name)
        return partial(self._loader, split_name)

    def __dir__(self) -> list[str]:
        """Include dynamic ``{split}_loader`` names when the dataset is available."""
        names = set(super().__dir__())
        try:
            for split in self.split_names():
                names.add(f"{split}{_LOADER_SUFFIX}")
        except BenchmarkError:
            pass
        return sorted(names)

    def _loader(
        self,
        split_name: str,
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 0,
        progress: bool = True,
    ) -> LoaderResult:
        """Yield task batches for one split partition."""
        self._ensure_dataset()
        assert self._dataset is not None
        index_batches = self.sampler.batch_indices(
            self._dataset.splits,
            split_name,
            scheme=self.split_scheme,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )
        if progress:
            total = _split_batch_count(
                self._dataset.splits,
                split_name,
                scheme=self.split_scheme,
                batch_size=batch_size,
                drop_last=drop_last,
            )
            index_batches = track(
                index_batches,
                description=loader_description(split_name, benchmark_id=self.id),
                total=total,
            )
        for indices in index_batches:
            yield self.task.extract(self._dataset, indices)

    def update(self, y_pred: GrumpyArray, y_true: GrumpyArray) -> None:
        """Inverse-transform targets and accumulate one metric batch.

        Predictions and ground truth are mapped back through the dataset transform
        chain before being passed to :attr:`metrics`.

        Args:
            y_pred: Model predictions in transformed target space.
            y_true: Ground-truth targets in transformed target space.
        """
        transforms = self.dataset.transforms
        self.metrics.update(
            transforms.inverse_transform(y_pred),
            transforms.inverse_transform(y_true),
        )

    def result(self, name: str | None = None) -> Result:
        """Compute accumulated metrics and return a formatted result.

        Args:
            name: Run label for the result. Defaults to the benchmark id.

        Returns:
            :class:`~fabric.core.result.Result` with computed metric values.
        """
        return Result(
            name=name or self.id,
            metrics=self.metrics.compute_all(),
        )
