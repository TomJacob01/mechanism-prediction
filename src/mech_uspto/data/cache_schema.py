"""Parquet schemas + I/O helpers for the canonical training cache.

Two tables, both written under ``data/cache/parquet/``:

- ``reactions.parquet``: one row per clean reaction. Holds reactant/product
  binaries plus split metadata.
- ``steps.parquet``: one row per elementary step. Holds the pre/post
  molecule binaries and the bond/charge changes that take us between them.

Molecules are stored as ``Mol.ToBinary()`` (RDKit native, ~3x faster to
deserialize than re-parsing SMILES and roughly 2x smaller). Bond and charge
changes are keyed by **atom map number**, not RDKit index — so the cache
survives any future re-alignment / ``AddHs`` choice; the downstream
featurizer resolves maps to indices once at load time.

The schema is the single source of truth — both the build script
(``scripts/build_parquet_cache.py``) and any future ``Dataset`` reader
import from here so the column names/types cannot drift.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


REACTIONS_SCHEMA = pa.schema([
    ("rxn_id",            pa.string()),
    ("n_steps",           pa.int16()),
    ("n_atoms_mapped",    pa.int32()),
    ("mechanistic_class", pa.string()),
    ("mechanistic_label", pa.string()),
    ("data_source",       pa.string()),
    ("reactant_mol",      pa.binary()),
    ("product_mol",       pa.binary()),
    ("split_hash",        pa.uint32()),
])


_BOND_CHANGE_STRUCT = pa.struct([
    ("map_i", pa.int32()),
    ("map_j", pa.int32()),
    ("delta", pa.int8()),
])

_CHARGE_CHANGE_STRUCT = pa.struct([
    ("map_i", pa.int32()),
    ("delta", pa.int8()),
])

STEPS_SCHEMA = pa.schema([
    ("rxn_id",         pa.string()),
    ("step_idx",       pa.int16()),
    ("mol_pre",        pa.binary()),
    ("mol_post",       pa.binary()),
    ("bond_changes",   pa.list_(_BOND_CHANGE_STRUCT)),
    ("charge_changes", pa.list_(_CHARGE_CHANGE_STRUCT)),
    ("arrow_count",    pa.int8()),
])


def split_hash(rxn_id: str) -> int:
    """Deterministic 32-bit hash for train/val/test bucketing.

    SHA-1 of the rxn_id keeps the bucket assignment stable across
    Python versions (``hash()`` is salted per-process).
    """
    h = hashlib.sha1(rxn_id.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


def open_writer(path: Path, schema: pa.Schema) -> pq.ParquetWriter:
    """Open a Zstd-compressed Parquet writer at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return pq.ParquetWriter(path, schema, compression="zstd")


def write_batch(writer: pq.ParquetWriter, rows: list[dict], schema: pa.Schema) -> None:
    """Append ``rows`` to ``writer`` as a single record batch."""
    if not rows:
        return
    table = pa.Table.from_pylist(rows, schema=schema)
    writer.write_table(table)


__all__ = [
    "REACTIONS_SCHEMA",
    "STEPS_SCHEMA",
    "split_hash",
    "open_writer",
    "write_batch",
]
