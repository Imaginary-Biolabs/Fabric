"""Fabric settings and storage paths."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import ClassVar

from fabric.utils.errors import ConfigurationError


class _SettingsMeta(type):
    """Metaclass exposing ``Settings.home`` and ``Settings.dataset_path`` class properties."""

    @property
    def dataset_path(cls) -> Path:
        """Active Fabric home directory (alias for :attr:`home`)."""
        return cls._active_home()

    @property
    def home(cls) -> Path:
        """Active Fabric home directory.

        Returns:
            Scoped home from an active context manager, or :attr:`Settings.DEFAULT_HOME`.
        """
        return cls._active_home()


class Settings(metaclass=_SettingsMeta):
    """Runtime settings with optional scoped override.

    Storage is split into tiers so teams can share raw and processed data while
    keeping transformed releases separate:

    * :meth:`raw_path` — external source files (PDB, mmCIF, …)
    * :meth:`processed_path` — ingested Grumpy releases before user transforms
    * :meth:`transformed_path` — releases after the dataset transform chain
    * :meth:`scratch_path` — optional fast local copy for training (``None`` when unset)

    Unset tier paths default to ``{home}/{tier}`` except scratch, which defaults
    to ``None``.

    Example:
        >>> from fabric import Settings
        >>> with Settings(
        ...     home="/tmp/imaginary",
        ...     raw="/mnt/shared/raw",
        ...     processed="/mnt/shared/processed",
        ...     scratch="/local/scratch/me",
        ... ):
        ...     Settings.processed_path()
        PosixPath('/mnt/shared/processed')
    """

    DEFAULT_HOME: ClassVar[Path] = Path("~/.imaginary")
    DEFAULT_WORKERS: ClassVar[int | None] = None
    _context: ClassVar[Settings | None] = None

    def __init__(
        self,
        home: str | Path,
        config_path: str | Path | None = None,
        *,
        workers: int | None = None,
        raw: str | Path | None = None,
        processed: str | Path | None = None,
        transformed: str | Path | None = None,
        scratch: str | Path | None = None,
    ) -> None:
        """Create a settings instance for use as a context manager.

        Args:
            home: Root directory for Fabric config and default storage tiers.
            config_path: Optional path to ``fabric-config.yaml``. Defaults to
                ``{home}/fabric-config.yaml``.
            workers: Parallel worker count for external file parsing.
            raw: Shared raw data root for externals. Defaults to ``{home}/raw``.
            processed: Shared processed (ingested) dataset cache. Defaults to
                ``{home}/processed``.
            transformed: Transformed dataset cache. Defaults to ``{home}/transformed``.
            scratch: Optional fast local path for staged training copies. When
                ``None``, scratch staging is disabled.
        """
        self._home = Path(home)
        if config_path is not None:
            self.config_path = Path(config_path)
        else:
            self.config_path = self._home / "fabric-config.yaml"
        self._workers = workers
        self._raw = Path(raw) if raw is not None else None
        self._processed = Path(processed) if processed is not None else None
        self._transformed = Path(transformed) if transformed is not None else None
        self._scratch = Path(scratch) if scratch is not None else None

    @classmethod
    def _active_home(cls) -> Path:
        """Return the home directory for the active settings context."""
        if cls._context is not None:
            return cls._context._home
        return cls.DEFAULT_HOME

    @classmethod
    def _active(cls) -> Settings | None:
        return cls._context

    @classmethod
    def _resolve_tier_path(cls, attr: str, default_subdir: str) -> Path:
        """Resolve a storage tier from scoped overrides or ``{home}/{subdir}``."""
        ctx = cls._active()
        if ctx is not None:
            override = getattr(ctx, attr)
            if override is not None:
                return override
        return cls._active_home() / default_subdir

    @classmethod
    def current(cls) -> Settings:
        """Return the active scoped settings or a default instance."""
        if cls._context is not None:
            return cls._context
        return cls(home=cls.DEFAULT_HOME)

    @classmethod
    def raw_path(cls) -> Path:
        """Shared root for raw external files (PDB, mmCIF, …).

        Returns:
            Configured raw path or ``{home}/raw``.
        """
        return cls._resolve_tier_path("_raw", "raw")

    @classmethod
    def processed_path(cls) -> Path:
        """Shared cache for ingested datasets before user transforms.

        Returns:
            Configured processed path or ``{home}/processed``.
        """
        return cls._resolve_tier_path("_processed", "processed")

    @classmethod
    def transformed_path(cls) -> Path:
        """Cache for dataset releases after the transform chain.

        Returns:
            Configured transformed path or ``{home}/transformed``.
        """
        return cls._resolve_tier_path("_transformed", "transformed")

    @classmethod
    def datasets_path(cls) -> Path:
        """Alias for :meth:`transformed_path` (legacy name)."""
        return cls.transformed_path()

    @classmethod
    def workers(cls) -> int:
        """Number of parallel workers for external file parsing."""
        if cls._context is not None and cls._context._workers is not None:
            return max(1, int(cls._context._workers))
        if cls.DEFAULT_WORKERS is not None:
            return max(1, int(cls.DEFAULT_WORKERS))
        return max(1, os.cpu_count() or 1)

    @classmethod
    def scratch_path(cls) -> Path | None:
        """Optional fast local directory for staged training copies.

        Returns:
            Configured scratch path, or ``None`` when scratch is not configured.
        """
        if cls._context is not None and cls._context._scratch is not None:
            return cls._context._scratch
        return None

    def raw_dir(self) -> Path:
        """Instance-scoped raw data root."""
        return self._raw if self._raw is not None else self._home / "raw"

    def processed_dir(self) -> Path:
        """Instance-scoped processed cache root."""
        return self._processed if self._processed is not None else self._home / "processed"

    def transformed_dir(self) -> Path:
        """Instance-scoped transformed cache root."""
        return self._transformed if self._transformed is not None else self._home / "transformed"

    def scratch_dir(self) -> Path | None:
        """Instance-scoped scratch path, if configured."""
        return self._scratch

    @classmethod
    def registry_cache_path(cls) -> Path:
        """Directory for cached registry metadata (platform phase)."""
        return cls._active_home() / "registry"

    def __enter__(self) -> Settings:
        """Activate this settings instance for the current scope."""
        if type(self)._context is not None:
            raise ConfigurationError("Settings context is already active")
        type(self)._context = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Deactivate the settings context."""
        type(self)._context = None


def copy_tree(src: Path, dest: Path) -> Path:
    """Copy a release directory tree, replacing ``dest`` when present.

    Args:
        src: Existing release directory.
        dest: Target directory.

    Returns:
        ``dest`` after copy.
    """
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    return dest
