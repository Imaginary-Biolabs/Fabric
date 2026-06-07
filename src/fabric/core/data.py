"""Fabric data model on Grumpy dataframes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fabric.utils.constants import SCHEMA_LEVELS
from fabric.utils.errors import SchemaError

try:
    from grumpy import GrumpyDataFrame as _GrumpyDataFrame
except ImportError:  # pragma: no cover
    _GrumpyDataFrame = None


@dataclass
class Assets:
    """Mutable sidecar data carried through transform pipelines."""

    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Split:
    """Train/val/test (or custom) index sets."""

    names: tuple[str, ...] = ()
    indices: dict[str, Any] = field(default_factory=dict)


@dataclass
class Data:
    """Thin wrapper around a Grumpy dataframe with schema helpers."""

    frame: Any

    def __init__(self, frame: Any) -> None:
        if _GrumpyDataFrame is None:
            raise TypeError("Data requires grumpy; install grumpy before constructing Data")
        if not isinstance(frame, _GrumpyDataFrame):
            raise TypeError("Data requires a grumpy.GrumpyDataFrame instance")
        self.frame = frame

    def _level_proxy(self, level: str) -> Any:
        if level not in SCHEMA_LEVELS:
            raise SchemaError(f"Unknown schema level '{level}'")
        try:
            return getattr(self.frame, level)
        except AttributeError as exc:
            raise SchemaError(
                f"Schema level '{level}' is not available on this dataframe"
            ) from exc
        except Exception as exc:
            raise SchemaError(f"Cannot access schema level '{level}': {exc}") from exc

    @property
    def scene(self) -> Any:
        return self._level_proxy("scene")

    @property
    def frame_level(self) -> Any:
        return self._level_proxy("frame")

    @property
    def molecule(self) -> Any:
        return self._level_proxy("molecule")

    @property
    def chain(self) -> Any:
        return self._level_proxy("chain")

    @property
    def residue(self) -> Any:
        return self._level_proxy("residue")

    @property
    def atom(self) -> Any:
        return self._level_proxy("atom")


@dataclass
class Batch:
    """One streamed chunk of data."""

    data: Data
    meta: dict[str, Any] = field(default_factory=dict)
