"""Factory for constructing Fabric objects from YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from fabric.core.backends import build_backend
from fabric.core.benchmark import Benchmark
from fabric.core.collater import Collater
from fabric.core.collaters import build_collater
from fabric.core.dataset import Dataset
from fabric.core.loggers import DiskLogger
from fabric.core.models import attach_optimizer, build_model
from fabric.core.trainer import Trainer
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

    @staticmethod
    def _resolve_config_path(
        path_or_id: str | Path,
        *,
        kind: str,
        subdir: str,
    ) -> Path:
        path = Path(path_or_id)
        if path.is_file():
            return path
        candidates = [
            Path.cwd() / "tests/fixtures" / subdir / f"{path_or_id}.yaml",
            Path.cwd() / "tests/fixtures/configs" / f"{path_or_id}.yaml",
            Path.cwd() / f"{path_or_id}.yaml",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise ConfigError(
            f"{kind} config not found for '{path_or_id}'; "
            "pass a YAML path or a known id."
        )

    @staticmethod
    def _resolve_model_path(path_or_id: str | Path) -> Path:
        return Factory._resolve_config_path(path_or_id, kind="Model", subdir="models")

    @staticmethod
    def model(
        path_or_id: str | Path,
        *,
        collater: Collater | None = None,
        input_dim: int | None = None,
    ):
        """Load a Fabric model from a local YAML config.

        Args:
            path_or_id: Path or id such as ``"M_mini"``.
            collater: Optional collater used to validate layout and infer input width.
            input_dim: Optional feature width override.

        Returns:
            Configured :class:`~fabric.core.model.Model`.
        """
        config_path = Factory._resolve_model_path(path_or_id)
        cfg = load_config(config_path)
        model = build_model(cfg, collater=collater, input_dim=input_dim)
        attach_optimizer(model, cfg.get("optimizer"))
        return model

    @staticmethod
    def _resolve_trainer_path(path_or_id: str | Path) -> Path:
        return Factory._resolve_config_path(path_or_id, kind="Trainer", subdir="train")

    @staticmethod
    def trainer(path_or_id: str | Path, **overrides: Any) -> Trainer:
        """Build a trainer from a local YAML config."""
        config_path = Factory._resolve_trainer_path(path_or_id)
        cfg = load_config(config_path)
        if isinstance(cfg, DictConfig):
            payload = OmegaConf.to_container(cfg, resolve=True) or {}
        else:
            payload = dict(cfg)
        payload.update(overrides)

        benchmark = Factory.benchmark(payload["benchmark"])
        collater = build_collater(payload["collater"])
        model_ref = payload["model"]
        if isinstance(model_ref, (dict, DictConfig)):
            model_cfg = model_ref
        else:
            model_cfg = load_config(Factory._resolve_model_path(model_ref))
        model = build_model(model_cfg, collater=collater)
        attach_optimizer(model, model_cfg.get("optimizer"))

        backend_cfg = payload.get("backend", {"TorchBackend": {"accelerator": "cpu"}})
        backend = build_backend(backend_cfg)

        logger_root = payload.get("root", "results")
        logger = DiskLogger(logger_root) if payload.get("logger", "DiskLogger") else None

        return Trainer(
            model=model,
            benchmark=benchmark,
            backend=backend,
            collater=collater,
            logger=logger,
            root=logger_root,
            epochs=int(payload.get("epochs", 1)),
            batch_size=int(payload.get("batch_size", 8)),
            progress=bool(payload.get("progress", False)),
        )
