"""PDB structure parser backed by BioPandas."""

from __future__ import annotations

from pathlib import Path

from biopandas.pdb import PandasPdb
from biopandas.pdb.engines import amino3to1dict

from fabric.core.data import Data
from fabric.core.parser import Parser
from fabric.core.parsers.structure import atom_table_to_data
from fabric.utils.errors import ParseError


class PdbParser(Parser):
    """Parse PDB files with BioPandas into Fabric :class:`~fabric.core.data.Data`.

    ATOM records are grouped by chain and nested into the full Fabric
    :data:`~fabric.utils.constants.SCHEMA`, with singleton scene, frame, and
    molecule wrappers for static structures.

    Example:
        >>> from pathlib import Path
        >>> data = PdbParser().parse(Path("structure.pdb"))
        >>> "chain_id" in data.data.columns
        True
    """

    def parse(self, path: Path) -> Data:
        """Read a PDB file and return a Fabric Data object.

        Args:
            path: Local path to a ``.pdb`` structure file.

        Returns:
            Data with one scene per parsed structure, using PDB column names
            (``chain_id``, ``residue_number``, ``x_coord``, etc.).

        Raises:
            ParseError: If the file is missing, unreadable, or contains no ATOM
                records.

        Example:
            >>> from pathlib import Path
            >>> data = PdbParser().parse(Path("mini.pdb"))
            >>> len(data.data) == 1
            True
        """
        path = Path(path)
        if not path.is_file():
            raise ParseError(f"PDB file not found: {path}")

        try:
            pdb = PandasPdb().read_pdb(str(path))
            atoms = pdb.df["ATOM"]
        except Exception as exc:
            raise ParseError(
                f"Failed to read PDB file {path.name} with BioPandas; "
                "check that the file is a valid PDB structure."
            ) from exc

        if atoms is None or atoms.empty:
            raise ParseError(
                f"No ATOM records found in {path.name}; "
                "provide a valid protein structure file."
            )

        molecule_id = path.stem
        return atom_table_to_data(
            atoms,
            scene_id=molecule_id,
            chain_id_col="chain_id",
            residue_number_col="residue_number",
            residue_name_col="residue_name",
            atom_name_col="atom_name",
            coord_cols=("x_coord", "y_coord", "z_coord"),
            residue_name_map=amino3to1dict,
            molecule_id=molecule_id,
        )
