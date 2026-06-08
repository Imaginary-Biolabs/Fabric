"""mmCIF structure parser backed by BioPandas."""

from __future__ import annotations

from pathlib import Path

from biopandas.mmcif import PandasMmcif
from biopandas.pdb.engines import amino3to1dict

from fabric.core.data import Data
from fabric.core.parser import Parser
from fabric.core.parsers.structure import atom_table_to_data
from fabric.utils.errors import ParseError


class MmcifParser(Parser):
    """Parse mmCIF files with BioPandas into Fabric :class:`~fabric.core.data.Data`.

    ATOM records use mmCIF column names (``label_asym_id``, ``label_seq_id``,
    ``Cartn_x``, etc.) and are nested into the full Fabric schema.

    Example:
        >>> from pathlib import Path
        >>> data = MmcifParser().parse(Path("structure.cif"))
        >>> "chain_id" in data.data.columns
        True
    """

    def parse(self, path: Path) -> Data:
        """Read an mmCIF file and return a Fabric Data object.

        Args:
            path: Local path to a ``.cif`` or ``.mmcif`` structure file.

        Returns:
            Data with one scene per parsed structure, using mmCIF label columns.

        Raises:
            ParseError: If the file is missing, unreadable, or contains no ATOM
                records.

        Example:
            >>> from pathlib import Path
            >>> data = MmcifParser().parse(Path("mini.cif"))
            >>> len(data.data) == 1
            True
        """
        path = Path(path)
        if not path.is_file():
            raise ParseError(f"mmCIF file not found: {path}")

        try:
            mmcif = PandasMmcif().read_mmcif(str(path))
            atoms = mmcif.df["ATOM"]
        except Exception as exc:
            raise ParseError(
                f"Failed to read mmCIF file {path.name} with BioPandas; "
                "check that the file is a valid mmCIF structure."
            ) from exc

        if atoms is None or atoms.empty:
            raise ParseError(
                f"No ATOM records found in {path.name}; "
                "provide a valid mmCIF structure file."
            )

        atoms = atoms.copy()
        atoms["label_seq_id"] = atoms["label_seq_id"].astype(int)

        molecule_id = path.stem
        return atom_table_to_data(
            atoms,
            scene_id=molecule_id,
            chain_id_col="label_asym_id",
            residue_number_col="label_seq_id",
            residue_name_col="label_comp_id",
            atom_name_col="label_atom_id",
            coord_cols=("Cartn_x", "Cartn_y", "Cartn_z"),
            residue_name_map=amino3to1dict,
            molecule_id=molecule_id,
        )
