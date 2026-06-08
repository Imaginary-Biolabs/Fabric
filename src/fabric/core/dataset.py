"""Config-driven dataset release to Grumpy Zarr with provenance sidecars."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import grumpy as gr
from omegaconf import DictConfig, OmegaConf

import fabric
from fabric.core.data import Assets, Batch, Data, Splits, iter_batches
from fabric.core.externals import build_external
from fabric.core.transforms import build_transforms
from fabric.utils.config import load_config
from fabric.utils.constants import GRUMPY_CHUNK_SIZE
from fabric.utils.errors import ConfigError, DatasetError
from fabric.utils.hashing import config_hash
from fabric.utils.io import info
from fabric.utils.settings import Settings, copy_tree


def validate_dataset_config(cfg: DictConfig) -> None:
    """Validate that a dataset config declares exactly one data source.

    Args:
        cfg: Dataset YAML loaded as an OmegaConf object.

    Raises:
        ConfigError: If both or neither of ``external`` and ``parent`` are set.
    """
    has_external = cfg.get("external") is not None
    has_parent = cfg.get("parent") is not None
    if has_external and has_parent:
        raise ConfigError("Dataset config cannot set both 'external' and 'parent'")
    if not has_external and not has_parent:
        raise ConfigError("Dataset config requires either 'external' or 'parent'")


def ingest_hash(cfg: DictConfig) -> str:
    """Hash the dataset source definition without transforms.

    Args:
        cfg: Dataset configuration.

    Returns:
        Stable digest for the external or parent ingest recipe.
    """
    external = cfg.get("external")
    payload = {
        "id": cfg.get("id"),
        "version": cfg.get("version"),
        "external": (
            OmegaConf.to_container(external, resolve=True) if external is not None else None
        ),
        "parent": cfg.get("parent"),
    }
    return config_hash(payload)


def _resolve_parent_config(parent_id: str, *, config_dir: Path) -> Path:
    """Locate a parent dataset YAML on disk.

    Args:
        parent_id: Dataset identifier referenced by ``parent:`` in the child config.
        config_dir: Directory containing the child dataset config.

    Returns:
        Resolved path to the parent dataset YAML file.

    Raises:
        DatasetError: If no candidate config file exists.
    """
    for candidate in (
        config_dir / f"{parent_id}.yaml",
        config_dir / f"{parent_id}.yml",
        Path.cwd() / "tests/fixtures/datasets" / f"{parent_id}.yaml",
        Path.cwd() / f"{parent_id}.yaml",
    ):
        if candidate.is_file():
            return candidate
    raise DatasetError(
        f"Parent dataset config '{parent_id}' not found; "
        "expected a YAML file next to the child dataset config."
    )


def _read_release_meta(path: Path) -> dict[str, Any]:
    """Load ``release.json`` when present.

    Args:
        path: Released dataset directory containing Grumpy Zarr files.

    Returns:
        Parsed metadata dictionary, or an empty dict when the sidecar is missing.
    """
    release_json = path / "release.json"
    if not release_json.is_file():
        return {}
    return json.loads(release_json.read_text())


def _write_grumpy_release(
    release_path: Path,
    batches: Iterator[Batch],
    *,
    meta: dict[str, Any],
) -> int:
    """Stream batches to a Grumpy Zarr directory and write ``release.json``.

    Returns:
        Total scene count written.
    """
    release_path.mkdir(parents=True, exist_ok=True)
    molecules = 0

    def frames():
        nonlocal molecules
        for batch in batches:
            molecules += len(batch.data.data)
            yield batch.data.data

    gr.save(frames(), str(release_path), chunk_size=GRUMPY_CHUNK_SIZE)
    release_json = json.dumps(meta, indent=2, sort_keys=True) + "\n"
    (release_path / "release.json").write_text(release_json)
    return molecules


class Dataset:
    """Config-driven dataset with tiered Grumpy Zarr storage.

    External datasets write two cache tiers:

    * **Processed** — ingested structures under
      :meth:`~fabric.utils.settings.Settings.processed_path`
    * **Transformed** — transform-chain output under
      :meth:`~fabric.utils.settings.Settings.transformed_path`

    Parent datasets read a released parent from the transformed tier and write only
    their own transformed release.

    Example:
        >>> from fabric import Settings
        >>> from fabric.core.factory import Factory
        >>> with Settings(home="/tmp/imaginary"):
        ...     ds = Factory.dataset("datasets/D_local.yaml")
        ...     path = ds.release()
        ...     len(ds.data.data) > 0
        True
    """

    def __init__(self, cfg: DictConfig, *, config_path: Path) -> None:
        """Create a dataset from a loaded YAML config.

        Args:
            cfg: Dataset configuration (``external`` or ``parent``, plus ``transforms``).
            config_path: Path to the YAML file ``cfg`` was loaded from.

        Raises:
            ConfigError: If the config is missing a data source or sets both sources.
        """
        validate_dataset_config(cfg)
        self.cfg = cfg
        self.config_path = Path(config_path)
        self.config_dir = self.config_path.parent
        self.id = str(cfg.get("id") or self.config_path.stem)
        self.version = int(cfg.get("version", 1))
        self.transforms = build_transforms(cfg.get("transforms"))
        self._release_path: Path | None = None
        self._data: Data | None = None
        self._assets: Assets | None = None
        self._splits: Splits | None = None
        self._release_meta: dict[str, Any] | None = None

    @property
    def config_hash(self) -> str:
        """Stable ``sha256:`` digest of the full dataset YAML."""
        return config_hash(self.cfg)

    @property
    def ingest_hash(self) -> str:
        """Stable ``sha256:`` digest of the ingest recipe (external or parent)."""
        return ingest_hash(self.cfg)

    @property
    def hash(self) -> str:
        """Stable ``sha256:`` digest of the transform chain."""
        return self.transforms.hash

    @property
    def processed_cache_path(self) -> Path:
        """Directory for the ingested (pre-transform) release.

        Returns:
            ``{Settings.processed_path()}/{id}/{version}/{ingest_hash}/``.
        """
        return Settings.processed_path() / self.id / str(self.version) / self.ingest_hash

    @property
    def cache_path(self) -> Path:
        """Directory for the transformed release.

        Returns:
            ``{Settings.transformed_path()}/{id}/{version}/{hash}/``.
        """
        return Settings.transformed_path() / self.id / str(self.version) / self.hash

    @property
    def scratch_path(self) -> Path | None:
        """Staged copy path for this release when scratch is configured.

        Returns:
            ``{Settings.scratch_path()}/{id}/{version}/{hash}/``, or ``None``.
        """
        root = Settings.scratch_path()
        if root is None:
            return None
        return root / self.id / str(self.version) / self.hash

    @property
    def data(self) -> Data:
        """Released structural payload."""
        self._load_if_needed()
        assert self._data is not None
        return self._data

    @property
    def assets(self) -> Assets:
        """Sidecar assets persisted in ``release.json``."""
        self._load_if_needed()
        assert self._assets is not None
        return self._assets

    @property
    def splits(self) -> Splits:
        """Partition indices persisted in ``release.json``."""
        self._load_if_needed()
        assert self._splits is not None
        return self._splits

    def _load_if_needed(self) -> None:
        """Load Zarr data and sidecars from the release directory if not cached."""
        if self._data is not None:
            return
        path = self._release_path or self.cache_path
        if not (path / "grumpy.json").is_file():
            raise DatasetError(
                f"Dataset '{self.id}' is not released at {path}; call release() first."
            )
        meta = _read_release_meta(path)
        self._data = Data(gr.load(str(path)))
        self._assets = Assets.from_dict(meta.get("assets"))
        self._splits = Splits.from_dict(meta.get("splits"))
        self._release_meta = meta or None

    def _ingest_parent(self) -> tuple[Data, Assets, Splits]:
        """Load input data from a released parent dataset."""
        parent_id = str(self.cfg.parent)
        parent_path = _resolve_parent_config(parent_id, config_dir=self.config_dir)
        parent = Dataset(load_config(parent_path), config_path=parent_path)
        if not (parent.cache_path / "grumpy.json").is_file():
            raise DatasetError(
                f"Parent dataset '{parent_id}' is not released; "
                f"run Factory.dataset('{parent_path}').release() before releasing '{self.id}'."
            )
        parent._load_if_needed()
        if parent._data is None or parent._assets is None or parent._splits is None:
            raise DatasetError(f"Parent dataset '{parent_id}' failed to load.")
        return parent._data, parent._assets, parent._splits

    def _load_or_release_processed(
        self, *, force: bool
    ) -> tuple[Iterator[Batch], Assets, Splits, int]:
        """Return ingest batches from the processed cache, building it when needed."""
        processed_path = self.processed_cache_path
        if not force:
            meta = _read_release_meta(processed_path)
            cache_valid = (
                (processed_path / "grumpy.json").is_file()
                and meta.get("ingest_hash") == self.ingest_hash
                and meta.get("stage") == "processed"
            )
            if cache_valid:
                info(
                    f"Processed cache hit {self.id} v{self.version} → {processed_path} "
                    f"(ingest_hash={meta['ingest_hash']}, molecules={meta['molecules']})"
                )
                data = Data(gr.load(str(processed_path)))
                assets = Assets.from_dict(meta.get("assets"))
                splits = Splits.from_dict(meta.get("splits"))
                return iter_batches(data), assets, splits, int(meta["molecules"])

        external = build_external(self.cfg.external, config_dir=self.config_dir)
        batches, assets, splits = external.load()
        n_scenes = int(assets.data.get("n_scenes", 0))
        if n_scenes <= 0:
            raise DatasetError(
                f"External for dataset '{self.id}' did not provide a positive n_scenes count"
            )

        meta = {
            "id": self.id,
            "version": self.version,
            "stage": "processed",
            "config_hash": self.config_hash,
            "ingest_hash": self.ingest_hash,
            "grumpy_version": gr.__version__,
            "fabric_version": fabric.__version__,
            "molecules": n_scenes,
            "splits": splits.to_dict(),
            "assets": assets.to_dict(),
            "external": external.describe(),
        }
        molecules = _write_grumpy_release(processed_path, batches, meta=meta)
        info(
            f"Processed {self.id} v{self.version} → {processed_path} "
            f"(ingest_hash={self.ingest_hash}, molecules={molecules})"
        )
        data = Data(gr.load(str(processed_path)))
        return iter_batches(data), assets, splits, molecules

    def release(self, *, force: bool = False) -> Path:
        """Ingest, transform, and write a Grumpy Zarr release with provenance.

        Args:
            force: When ``True``, rebuild even if a matching cached release exists.

        Returns:
            Path to the transformed release directory (:attr:`cache_path`).
        """
        release_path = self.cache_path
        if not force:
            meta = _read_release_meta(release_path)
            cache_valid = (
                (release_path / "grumpy.json").is_file()
                and meta.get("transform_hash") == self.hash
            )
            if cache_valid:
                info(
                    f"Transformed cache hit {self.id} v{self.version} → {release_path} "
                    f"(transform_hash={meta['transform_hash']}, molecules={meta['molecules']})"
                )
                self._release_path = release_path
                self._release_meta = meta
                self._data = Data(gr.load(str(release_path)))
                self._assets = Assets.from_dict(meta.get("assets"))
                self._splits = Splits.from_dict(meta.get("splits"))
                return release_path

        if self.cfg.get("parent") is not None:
            data, assets, splits = self._ingest_parent()
            n_scenes = len(data.data)
            batches = iter_batches(data)
        else:
            batches, assets, splits, n_scenes = self._load_or_release_processed(force=force)

        transformed_batches, assets, splits = self.transforms.apply(
            batches, assets, splits, n_scenes=n_scenes
        )
        meta = {
            "id": self.id,
            "version": self.version,
            "stage": "transformed",
            "config_hash": self.config_hash,
            "ingest_hash": self.ingest_hash,
            "transform_hash": self.hash,
            "grumpy_version": gr.__version__,
            "fabric_version": fabric.__version__,
            "molecules": 0,
            "splits": {},
            "assets": assets.to_dict(),
        }
        if self.cfg.get("external") is not None:
            external = build_external(self.cfg.external, config_dir=self.config_dir)
            meta["external"] = external.describe()
            meta["processed_path"] = str(self.processed_cache_path)
        if self.cfg.get("parent") is not None:
            meta["parent"] = str(self.cfg.parent)

        molecules = _write_grumpy_release(release_path, transformed_batches, meta=meta)
        splits = self.transforms.splits
        meta["molecules"] = molecules
        meta["splits"] = splits.to_dict()
        release_json = json.dumps(meta, indent=2, sort_keys=True) + "\n"
        (release_path / "release.json").write_text(release_json)
        info(
            f"Released {self.id} v{self.version} → {release_path} "
            f"(transform_hash={self.hash}, molecules={molecules})"
        )

        self._release_path = release_path
        self._data = Data(gr.load(str(release_path)))
        self._assets = assets
        self._splits = splits
        self._release_meta = meta
        return release_path

    def stage_to_scratch(self, *, force: bool = False) -> Path:
        """Copy the transformed release to the configured scratch path.

        Args:
            force: When ``True``, replace an existing staged copy.

        Returns:
            Scratch directory containing the staged release.

        Raises:
            DatasetError: If scratch is not configured or the dataset is not released.
        """
        scratch_root = Settings.scratch_path()
        if scratch_root is None:
            raise DatasetError(
                "Scratch path is not configured; pass scratch= to Settings or set it in "
                "fabric-config.yaml before calling stage_to_scratch()."
            )
        dest = self.scratch_path
        assert dest is not None
        if not force and (dest / "grumpy.json").is_file():
            info(f"Scratch hit {self.id} v{self.version} → {dest}")
            return dest
        source = self.cache_path
        if not (source / "grumpy.json").is_file():
            raise DatasetError(
                f"Dataset '{self.id}' is not released at {source}; call release() first."
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        copy_tree(source, dest)
        info(f"Staged {self.id} v{self.version} → {dest}")
        return dest

    def release_metadata(self) -> dict[str, Any]:
        """Return provenance metadata from the latest transformed release."""
        if self._release_meta is not None:
            return self._release_meta
        meta = _read_release_meta(self.cache_path)
        if not meta:
            raise DatasetError(f"Dataset '{self.id}' has no release.json at {self.cache_path}")
        self._release_meta = meta
        return meta
