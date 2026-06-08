"""Helpers for nesting flat atom tables into Fabric Grumpy dataframes."""

from __future__ import annotations

import grumpy as gr
import pandas as pd

from fabric.core.data import Data

_STRING_COLUMNS = frozenset(
    {"scene_id", "frame_id", "molecule_id", "chain_id", "residue_label", "atom_label"}
)


def _columns_for_dataframe(columns: dict[str, object]) -> dict[str, object]:
    """Prepare a column mapping for a single :func:`grumpy.dataframe` call.

    Numeric nested lists are passed through for Grumpy to convert. String columns
    need an explicit ``dtype=gr.string`` because nested id/label payloads mix
    shapes Grumpy cannot infer reliably.
    """
    return {
        name: gr.array(value, dtype=gr.string) if name in _STRING_COLUMNS else value
        for name, value in columns.items()
    }


def _nest_chain_atoms(
    chain_rows: pd.DataFrame,
    *,
    chain_id: str,
    residue_number_col: str,
    residue_name_col: str,
    atom_name_col: str,
    coord_cols: tuple[str, str, str],
    residue_name_map: dict[str, str],
) -> tuple[str, list[int], list[str], list[list[str]], list[list[list[float]]]]:
    """Nest atom rows for one chain into residue and atom groupings.

    Args:
        chain_rows: Atom table rows for a single chain, sorted by residue.
        chain_id: Chain identifier string.
        residue_number_col: Column name for residue sequence numbers.
        residue_name_col: Column name for residue names (three-letter codes).
        atom_name_col: Column name for atom labels.
        coord_cols: Tuple of x, y, z coordinate column names.
        residue_name_map: Mapping from three-letter residue codes to one-letter labels.

    Returns:
        Tuple of ``(chain_id, residue_numbers, residue_labels, atom_labels,
        atom_positions)`` suitable for Grumpy array construction.
    """
    residues: list[int] = []
    residue_labels: list[str] = []
    atom_labels: list[list[str]] = []
    atom_positions: list[list[list[float]]] = []
    current_residue: int | None = None

    for row in chain_rows.itertuples(index=False):
        residue_number = int(getattr(row, residue_number_col))
        if current_residue != residue_number:
            current_residue = residue_number
            residues.append(residue_number)
            residue_name = str(getattr(row, residue_name_col))
            residue_labels.append(residue_name_map.get(residue_name, "X"))
            atom_labels.append([])
            atom_positions.append([])
        atom_labels[-1].append(str(getattr(row, atom_name_col)))
        atom_positions[-1].append(
            [
                float(getattr(row, coord_cols[0])),
                float(getattr(row, coord_cols[1])),
                float(getattr(row, coord_cols[2])),
            ]
        )

    return chain_id, residues, residue_labels, atom_labels, atom_positions


def atom_table_to_data(
    atoms: pd.DataFrame,
    *,
    scene_id: str,
    chain_id_col: str,
    residue_number_col: str,
    residue_name_col: str,
    atom_name_col: str,
    coord_cols: tuple[str, str, str],
    residue_name_map: dict[str, str],
    frame_id: str | None = None,
    molecule_id: str | None = None,
) -> Data:
    """Build Fabric :class:`~fabric.core.data.Data` from a flat atom table.

    Static structures are wrapped with singleton scene, frame, and molecule
    levels so the result conforms to :data:`~fabric.utils.constants.SCHEMA`.

    Args:
        atoms: Pandas DataFrame of atom records (one row per atom).
        scene_id: Identifier for the top-level scene.
        chain_id_col: Column holding chain identifiers.
        residue_number_col: Column holding per-chain residue numbers.
        residue_name_col: Column holding residue names.
        atom_name_col: Column holding atom names.
        coord_cols: Tuple of x, y, z coordinate column names.
        residue_name_map: Mapping from three-letter residue codes to one-letter labels.
        frame_id: Optional frame identifier. Defaults to ``"{molecule_id}_f0"``.
        molecule_id: Optional molecule identifier. Defaults to ``scene_id``.

    Returns:
        Data object with nested Grumpy arrays for all schema levels.

    Example:
        >>> import pandas as pd
        >>> from biopandas.pdb.engines import amino3to1dict
        >>> atoms = pd.DataFrame({...})  # doctest: +SKIP
        >>> data = atom_table_to_data(  # doctest: +SKIP
        ...     atoms,
        ...     scene_id="demo",
        ...     chain_id_col="chain_id",
        ...     residue_number_col="residue_number",
        ...     residue_name_col="residue_name",
        ...     atom_name_col="atom_name",
        ...     coord_cols=("x_coord", "y_coord", "z_coord"),
        ...     residue_name_map=amino3to1dict,
        ... )
    """
    molecule_id = molecule_id or scene_id
    frame_id = frame_id or f"{molecule_id}_f0"

    chains: list[str] = []
    chain_groups: dict[str, pd.DataFrame] = {}
    for chain_id, group in atoms.groupby(chain_id_col, sort=False):
        chain_key = str(chain_id)
        chains.append(chain_key)
        chain_groups[chain_key] = group

    chain_id_out: list[str] = []
    residue_number_out: list[list[int]] = []
    residue_label_out: list[list[str]] = []
    atom_label_out: list[list[list[str]]] = []
    atom_pos_out: list[list[list[list[float]]]] = []

    for chain_key in chains:
        (
            chain_label,
            residues,
            residue_labels,
            atom_labels,
            atom_positions,
        ) = _nest_chain_atoms(
            chain_groups[chain_key],
            chain_id=chain_key,
            residue_number_col=residue_number_col,
            residue_name_col=residue_name_col,
            atom_name_col=atom_name_col,
            coord_cols=coord_cols,
            residue_name_map=residue_name_map,
        )
        chain_id_out.append(chain_label)
        residue_number_out.append(residues)
        residue_label_out.append(residue_labels)
        atom_label_out.append(atom_labels)
        atom_pos_out.append(atom_positions)

    columns = {
        "scene_id": [scene_id],
        "frame_id": [[frame_id]],
        "molecule_id": [[[molecule_id]]],
        "chain_id": [[chain_id_out]],
        "residue_number": [[[residue_number_out]]],
        "residue_label": [[[residue_label_out]]],
        "atom_label": [[[atom_label_out]]],
        "atom_pos": [[[atom_pos_out]]],
    }
    return Data(_columns_for_dataframe(columns))
