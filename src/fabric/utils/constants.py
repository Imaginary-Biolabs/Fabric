"""Biological and platform constants.

Module-level constants used across parsers, transforms, and terminal output.

Attributes:
    BRAND_INK_400: Imaginary ink palette (light).
    BRAND_INK_600: Imaginary ink palette (dark), used for progress bar tracks.
    BRAND_GREEN_LIGHT: Imaginary green accent (finished progress).
    BRAND_GREEN_DARK: Imaginary green accent (active progress).
    SCHEMA: Canonical biological nesting levels for Fabric Grumpy dataframes.
    GRUMPY_CHUNK_SIZE: Scene batch size for external ingest and ``gr.save`` chunking.
    AMINO_ACIDS_1: One-letter amino acid alphabet.
    AMINO_ACIDS_3: Three-letter amino acid codes in standard order.
"""

from __future__ import annotations

# Imaginary brand palette (see grumpy/docs/stylesheets/imaginary.css).
BRAND_INK_400 = "#777067"
BRAND_INK_600 = "#484240"
BRAND_GREEN_LIGHT = "#4a6b52"
BRAND_GREEN_DARK = "#9bc4a8"

# Fabric biological schema (see .ai/project/fabric.md):
# scene → frame → molecule → chain → residue/group → atom
SCHEMA: list[str | tuple[str, ...]] = [
    "scene",
    "frame",
    "molecule",
    "chain",
    ("residue", "group"),
    "atom",
]

# Matches gr.save(..., chunk_size=...) default; external adapters batch by scene count.
GRUMPY_CHUNK_SIZE = 1024

AMINO_ACIDS_1 = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACIDS_3 = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]
