"""Factory for constructing Fabric objects from YAML configs."""

from __future__ import annotations

from pathlib import Path

from fabric.core.benchmark import Benchmark
from fabric.core.dataset import Dataset
from fabric.utils.config import load_config
from fabric.utils.errors import ConfigError


class Factory:
    """Construct Fabric objects from local YAML configs.

    Example:
        >>> from fabric.core.factory import Factory
        >>> ds = Factory.dataset("tests/fixtures/datasets/D_local.yaml")
        >>> ds.id
        'D_local'
    """

    @staticmethod
    def _resolve_dataset_path(path_or_id: str | Path) -> Path:
        """Resolve a dataset YAML path from an explicit path or known id.

        Args:
            path_or_id: Filesystem path to a YAML file, or a dataset id such as
                ``"D_local"`` (resolved under common fixture directories).

        Returns:
            Existing config file path.

        Raises:
            ConfigError: If no matching YAML file is found.
        """
        path = Path(path_or_id)
        if path.is_file():
            return path
        candidates = [
            Path.cwd() / "tests/fixtures/datasets" / f"{path_or_id}.yaml",
            Path.cwd() / "tests/fixtures/configs" / f"{path_or_id}.yaml",
            Path.cwd() / f"{path_or_id}.yaml",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise ConfigError(
            f"Dataset config not found for '{path_or_id}'; "
            "pass a YAML path or a known dataset id."
        )

    @staticmethod
    def dataset(path_or_id: str | Path) -> Dataset:
        """Load a dataset from a local YAML config.

        Args:
            path_or_id: Path or id passed to :meth:`_resolve_dataset_path`.

        Returns:
            Dataset ready for :meth:`~fabric.core.dataset.Dataset.release`.

        Example:
            >>> ds = Factory.dataset("D_local")
            >>> path = ds.release()
        """
        config_path = Factory._resolve_dataset_path(path_or_id)
        cfg = load_config(config_path)
        return Dataset(cfg, config_path=config_path)

    @staticmethod
    def _resolve_benchmark_path(path_or_id: str | Path) -> Path:
        """Resolve a benchmark YAML path from an explicit path or known id.

        Args:
            path_or_id: Filesystem path to a YAML file, or a benchmark id such as
                ``"B_mini"`` (resolved under common fixture directories).

        Returns:
            Existing config file path.

        Raises:
            ConfigError: If no matching YAML file is found.
        """
        path = Path(path_or_id)
        if path.is_file():
            return path
        candidates = [
            Path.cwd() / "tests/fixtures/benchmarks" / f"{path_or_id}.yaml",
            Path.cwd() / "tests/fixtures/configs" / f"{path_or_id}.yaml",
            Path.cwd() / f"{path_or_id}.yaml",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise ConfigError(
            f"Benchmark config not found for '{path_or_id}'; "
            "pass a YAML path or a known benchmark id."
        )

    @staticmethod
    def benchmark(path_or_id: str | Path) -> Benchmark:
        """Load a benchmark from a local YAML config.

        Args:
            path_or_id: Path or id passed to :meth:`_resolve_benchmark_path`.

        Returns:
            Benchmark with split loaders bound to the referenced dataset.

        Example:
            >>> bench = Factory.benchmark("B_mini")
            >>> list(bench.train_loader(batch_size=2))
        """
        config_path = Factory._resolve_benchmark_path(path_or_id)
        cfg = load_config(config_path)
        return Benchmark(cfg, config_path=config_path)
