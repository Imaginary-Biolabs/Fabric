"""Local filesystem external adapter."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grumpy as gr
from grumpy import GrumpyDataFrame

from fabric.core.data import Assets, Batch, Data, Splits
from fabric.core.external import External, ExternalLoadResult
from fabric.core.parser import Parser
from fabric.core.parsers.mmcif import MmcifParser
from fabric.core.parsers.pdb import PdbParser
from fabric.utils import constants
from fabric.utils.errors import ExternalError
from fabric.utils.settings import Settings


def parser_for_path(path: Path) -> Parser:
    """Select a structure parser from the file suffix.

    Args:
        path: Local structure file path (``.pdb``, ``.cif``, ``.mmcif``, optionally ``.gz``).

    Returns:
        :class:`~fabric.core.parsers.pdb.PdbParser` or
        :class:`~fabric.core.parsers.mmcif.MmcifParser`.

    Raises:
        ExternalError: If the suffix is not supported.
    """
    suffixes = path.suffixes
    if suffixes and suffixes[-1] == ".gz":
        suffixes = suffixes[:-1]
    ext = "".join(suffixes).lower()
    if ext == ".pdb":
        return PdbParser()
    if ext in {".cif", ".mmcif"}:
        return MmcifParser()
    raise ExternalError(
        f"Unsupported structure file '{path.name}'; expected .pdb, .cif, or .mmcif."
    )


def _parser_for_name(parser_name: str | None, path: Path) -> Parser:
    """Return a parser from an explicit name or file suffix."""
    if parser_name is not None:
        if parser_name.lower() == "pdb":
            return PdbParser()
        if parser_name.lower() in {"mmcif", "cif"}:
            return MmcifParser()
        raise ExternalError(
            f"Unsupported parser '{parser_name}' for LocalAdapter; "
            "use parser='pdb' or parser='mmcif'."
        )
    return parser_for_path(path)


def _combine_frames(frames: list[GrumpyDataFrame]) -> GrumpyDataFrame:
    """Concatenate scene dataframes into one batch-sized Grumpy dataframe."""
    if len(frames) == 1:
        return frames[0]
    with tempfile.TemporaryDirectory(prefix="fabric-batch-") as tmp:
        store = Path(tmp) / "batch.gr"
        gr.save(iter(frames), str(store), chunk_size=constants.GRUMPY_CHUNK_SIZE)
        return gr.load(str(store))


class LocalAdapter(External):
    """Glob local structure files and parse them with BioPandas-backed parsers.

    Files are parsed in parallel (see :meth:`~fabric.utils.settings.Settings.workers`)
    and yielded in batches of :data:`~fabric.utils.constants.GRUMPY_CHUNK_SIZE`
    scenes. Matched file paths are recorded once in ``assets.data["files"]``.

    Args:
        paths: Glob patterns resolved relative to the working directory or
            ``config_dir``.
        config_dir: Directory containing the dataset YAML (used as a search root).
        parser: Optional forced parser name (``"pdb"`` or ``"mmcif"``). When
            ``None``, the parser is chosen from each file extension.

    Example:
        >>> adapter = LocalAdapter(
        ...     ["tests/fixtures/structures/*.pdb"],
        ...     config_dir=Path("tests/fixtures/datasets"),
        ... )
        >>> batches, assets, _ = adapter.load()
        >>> len(next(batches).data.data) >= 1
        True
    """

    def __init__(
        self,
        paths: list[str],
        *,
        config_dir: Path,
        parser: str | None = None,
    ) -> None:
        """Resolve glob patterns and optionally force a parser type."""
        self.paths = list(paths)
        self.config_dir = Path(config_dir)
        self.parser_name = parser
        self._resolved_paths = self._resolve_paths()

    def _resolve_paths(self) -> list[Path]:
        """Expand configured glob patterns to unique absolute file paths.

        Returns:
            Sorted, de-duplicated list of resolved structure files.
        """
        found: list[Path] = []
        seen: set[Path] = set()
        search_roots = (
            Settings.raw_path(),
            Path.cwd(),
            self.config_dir,
            self.config_dir.parent,
        )
        for pattern in self.paths:
            matches: list[Path] = []
            for root in search_roots:
                matches = sorted(root.glob(pattern))
                if matches:
                    break
            for path in matches:
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    found.append(resolved)
        return found

    def _parse_file(self, path: Path) -> Data:
        """Parse one structure file."""
        return _parser_for_name(self.parser_name, path).parse(path)

    def _parse_files(self) -> list[Data]:
        """Parse all matched files, optionally in parallel."""
        paths = self._resolved_paths
        workers = Settings.workers()
        if workers <= 1:
            return [self._parse_file(path) for path in paths]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(self._parse_file, paths))

    def load(self) -> ExternalLoadResult:
        """Return a batch generator plus ingest-time assets and splits.

        Returns:
            ``(batches, assets, splits)`` where ``batches`` yields up to
            :data:`~fabric.utils.constants.GRUMPY_CHUNK_SIZE` scenes per item
            and ``assets`` lists all matched source files once.

        Raises:
            ExternalError: If no files matched the configured glob patterns.
        """
        if not self._resolved_paths:
            patterns = ", ".join(self.paths)
            raise ExternalError(
                f"LocalAdapter matched no files for patterns: {patterns}; "
                "check paths relative to the dataset config or repo root."
            )

        parsed = self._parse_files()
        assets = Assets(
            {
                "files": [str(path) for path in self._resolved_paths],
                "n_scenes": len(parsed),
            }
        )
        splits = Splits()

        def batches() -> Iterator[Batch]:
            frames: list[GrumpyDataFrame] = []
            for data in parsed:
                frames.append(data.data)
                if len(frames) >= constants.GRUMPY_CHUNK_SIZE:
                    yield Batch(data=Data(_combine_frames(frames)))
                    frames = []
            if frames:
                yield Batch(data=Data(_combine_frames(frames)))

        return batches(), assets, splits

    def describe(self) -> dict[str, object]:
        """Return provenance metadata for ``release.json``.

        Returns:
            Adapter name, parser selection, configured patterns, and resolved files.
        """
        return {
            "adapter": "LocalAdapter",
            "parser": self.parser_name or "auto",
            "paths": self.paths,
            "files": [str(path) for path in self._resolved_paths],
        }
