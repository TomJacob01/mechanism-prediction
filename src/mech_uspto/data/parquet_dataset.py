"""PyG Dataset over the canonical Parquet cache.

Reads from ``data/cache/parquet/{reactions,steps}.parquet`` (produced by
``scripts/build_parquet_cache.py``). Featurizes mol blobs on-read so feature
edits don't require a cache rebuild. Two task modes mirror the old
``MechUSPTODataset``:

- ``stepwise``: one item per elementary step. ``y`` is the bond-Δ matrix
  for that single step, built **directly from the cached map-keyed
  ``bond_changes``** — no kekulize / adjacency diff at train time.
- ``end_to_end``: one item per reaction. ``y`` is the full R→P bond-Δ
  matrix via :class:`DeltaMatrixGenerator`.

Splits are driven by the ``split_hash`` column (SHA1 of ``rxn_id``):
deterministic, repo-portable, and immune to row reordering. Stepwise items
inherit their reaction's bucket, so train/val never share intermediates of
the same reaction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pyarrow.parquet as pq
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data

from mech_uspto.data.featurization import align_atoms, featurize_edges, featurize_nodes
from mech_uspto.data.spectators import SpectatorDetector
from mech_uspto.data.transformations import DeltaMatrixGenerator


VALID_TASK_MODES = ("stepwise", "end_to_end")
VALID_SPLITS = ("train", "val", "test", "all")

# Default split fractions (≈ 80/10/10) over the uint32 hash space.
_U32_MAX = 2**32
_VAL_THRESH = int(_U32_MAX * 0.80)
_TEST_THRESH = int(_U32_MAX * 0.90)


def split_of(h: int) -> str:
    """Map a ``split_hash`` value to one of ``train`` / ``val`` / ``test``."""
    if h < _VAL_THRESH:
        return "train"
    if h < _TEST_THRESH:
        return "val"
    return "test"


def _featurize_binary(blob: bytes, add_hs: bool) -> tuple[Chem.Mol, Data]:
    """Deserialize a cached Mol binary into an aligned Mol + PyG ``Data``."""
    mol = Chem.Mol(blob)
    if add_hs:
        mol = Chem.AddHs(mol)
    aligned = align_atoms(mol)
    x = featurize_nodes(aligned)
    edge_index, edge_attr = featurize_edges(aligned)
    return aligned, Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _delta_from_map_changes(
    aligned_mol: Chem.Mol,
    bond_changes: list[dict],
) -> torch.Tensor:
    """Build the (N, N) bond-Δ tensor from cached map-keyed changes."""
    n = aligned_mol.GetNumAtoms()
    delta = torch.zeros((n, n), dtype=torch.long)
    map_to_idx = {
        a.GetAtomMapNum(): a.GetIdx()
        for a in aligned_mol.GetAtoms() if a.GetAtomMapNum() > 0
    }
    for bc in bond_changes:
        i = map_to_idx.get(bc["map_i"])
        j = map_to_idx.get(bc["map_j"])
        if i is None or j is None:
            continue
        delta[i, j] += int(bc["delta"])
        delta[j, i] += int(bc["delta"])
    return delta


class ParquetMechDataset(Dataset):
    """PyG ``Dataset`` backed by the parquet cache.

    Args:
        parquet_dir: directory containing ``reactions.parquet`` and ``steps.parquet``.
        task_mode: ``"stepwise"`` or ``"end_to_end"``.
        split: ``"train"`` / ``"val"`` / ``"test"`` / ``"all"`` (default ``"all"``).
        add_hs: whether to materialise explicit hydrogens before featurization.
        compute_spectators: attach a per-atom spectator mask + record ratio.
        limit: cap the number of underlying rows (handy for smoke tests).
    """

    def __init__(
        self,
        parquet_dir: str | Path,
        task_mode: str = "stepwise",
        split: str = "all",
        add_hs: bool = True,
        compute_spectators: bool = True,
        limit: Optional[int] = None,
    ):
        if task_mode not in VALID_TASK_MODES:
            raise ValueError(f"task_mode must be one of {VALID_TASK_MODES}, got {task_mode!r}")
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")

        self.task_mode = task_mode
        self.split = split
        self.add_hs = add_hs
        self.compute_spectators = compute_spectators

        parquet_dir = Path(parquet_dir)
        self._reactions_path = parquet_dir / "reactions.parquet"
        self._steps_path = parquet_dir / "steps.parquet"

        rxn_table = pq.read_table(self._reactions_path)

        if task_mode == "end_to_end":
            self._rows = rxn_table.to_pylist()
            self._rxn_split: dict[str, int] = {}
        else:
            # Build a rxn_id -> split_hash map so step rows inherit the rxn split.
            rxn_ids = rxn_table.column("rxn_id").to_pylist()
            split_hashes = rxn_table.column("split_hash").to_pylist()
            self._rxn_split = dict(zip(rxn_ids, split_hashes))
            self._rows = pq.read_table(self._steps_path).to_pylist()

        # Apply split + limit filters.
        if split != "all":
            def _keep(row: dict) -> bool:
                h = (
                    row["split_hash"]
                    if task_mode == "end_to_end"
                    else self._rxn_split.get(row["rxn_id"])
                )
                return h is not None and split_of(h) == split
            self._rows = [r for r in self._rows if _keep(r)]
        if limit is not None:
            self._rows = self._rows[:limit]

        # Populated lazily as items are read.
        self.spectator_ratios: list[float] = []

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> Data:
        row = self._rows[idx]
        if self.task_mode == "stepwise":
            return self._build_stepwise(row)
        return self._build_end_to_end(row)

    # ------------------------------------------------------------------
    # Per-mode builders
    # ------------------------------------------------------------------

    def _build_stepwise(self, row: dict) -> Data:
        aligned_pre, data = _featurize_binary(row["mol_pre"], self.add_hs)
        delta = _delta_from_map_changes(aligned_pre, row["bond_changes"])
        data.y = delta

        if self.compute_spectators:
            mask = SpectatorDetector.identify_spectators(aligned_pre, aligned_pre, delta)
            data.spectator_mask = mask
            self.spectator_ratios.append(SpectatorDetector.compute_spectator_ratio(mask))

        data.reaction_id = row["rxn_id"]
        data.step_id = int(row["step_idx"])
        data.task_mode = "stepwise"
        return data

    def _build_end_to_end(self, row: dict) -> Data:
        aligned_r, data = _featurize_binary(row["reactant_mol"], self.add_hs)
        aligned_p = align_atoms(
            Chem.AddHs(Chem.Mol(row["product_mol"])) if self.add_hs
            else Chem.Mol(row["product_mol"])
        )

        # R→P bond-Δ from adjacency diff (clamped for end-to-end magnitudes).
        delta_f = DeltaMatrixGenerator.delta_from_reactants_products(aligned_r, aligned_p)
        delta = torch.clamp(delta_f.round().long(), -3, 3)
        data.y = delta

        if self.compute_spectators:
            mask = SpectatorDetector.identify_spectators(aligned_r, aligned_p, delta)
            data.spectator_mask = mask
            self.spectator_ratios.append(SpectatorDetector.compute_spectator_ratio(mask))

        data.reaction_id = row["rxn_id"]
        data.step_id = -1
        data.task_mode = "end_to_end"
        return data


__all__ = ["ParquetMechDataset", "split_of", "VALID_TASK_MODES", "VALID_SPLITS"]
