"""Fabric settings and storage paths."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from fabric.utils.errors import ConfigurationError


class _SettingsMeta(type):
    @property
    def dataset_path(cls) -> Path:
        return cls._active_home()

    @property
    def home(cls) -> Path:
        return cls._active_home()


class Settings(metaclass=_SettingsMeta):
    """Runtime settings with optional scoped override.

    ``Settings.dataset_path`` is the Fabric home directory (``~/.imaginary`` by default).
    Storage subdirectories (e.g. ``datasets/``) are derived from this root in later phases.
    """

    DEFAULT_HOME: ClassVar[Path] = Path("~/.imaginary")
    _context: ClassVar[Settings | None] = None

    def __init__(self, home: str | Path, config_path: str | Path | None = None) -> None:
        self._home = Path(home)
        if config_path is not None:
            self.config_path = Path(config_path)
        else:
            self.config_path = self._home / "fabric-config.yaml"

    @classmethod
    def _active_home(cls) -> Path:
        if cls._context is not None:
            return cls._context._home
        return cls.DEFAULT_HOME

    @classmethod
    def current(cls) -> Settings:
        if cls._context is not None:
            return cls._context
        return cls(home=cls.DEFAULT_HOME)

    def datasets_path(self) -> Path:
        return self._active_home() / "datasets"

    def scratch_path(self) -> Path:
        return self._active_home() / "scratch"

    def registry_cache_path(self) -> Path:
        return self._active_home() / "registry"

    def __enter__(self) -> Settings:
        if type(self)._context is not None:
            raise ConfigurationError("Settings context is already active")
        type(self)._context = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        type(self)._context = None
