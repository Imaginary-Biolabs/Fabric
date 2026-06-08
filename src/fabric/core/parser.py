"""Parser protocol for biological file formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from fabric.core.data import Data


class Parser(ABC):
    """Read a structure file into Fabric :class:`~fabric.core.data.Data`.

    Implementations must populate all levels of :data:`~fabric.utils.constants.SCHEMA`,
    wrapping missing levels in singleton containers where needed.

    Example:
        >>> from fabric.core.parsers.pdb import PdbParser
        >>> data = PdbParser().parse(Path("structure.pdb"))
    """

    @abstractmethod
    def parse(self, path: Path) -> Data:
        """Parse a structure file from disk.

        Args:
            path: Local path to a supported structure file.

        Returns:
            Data object adhering to Fabric :data:`~fabric.utils.constants.SCHEMA`.

        Raises:
            ParseError: If the file is missing, unreadable, or contains no atoms.
        """
        raise NotImplementedError
