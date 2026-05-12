"""Dual-mode dataset over mech-USPTO-31k reactions."""

from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm.auto import tqdm

from mech_uspto.data.featurization import process_mapped_smiles
from mech_uspto.data.schema import MultiStepReaction
from mech_uspto.data.spectators import SpectatorDetector
from mech_uspto.data.transformations import DeltaMatrixGenerator

VALID_TASK_MODES = ("stepwise", "end_to_end")


class MechUSPTODataset(Dataset):
    """Unified dataset supporting both stepwise and end-to-end task modes.

    Modes:
        ``stepwise``: one data point per elementary step ``S_i → S_{i+1}``.
            Targets Δ_micro ∈ {-1, 0, 1}.
        ``end_to_end``: one data point per full reaction ``S_0 → S_final``.
            Targets Δ_macro ∈ {-2, -1, 0, 1, 2}.
    """

    def __init__(
        self,
        reactions: List[MultiStepReaction],
        task_mode: str = "stepwise",
        add_hs: bool = True,
        compute_spectators: bool = True,
        max_retries: int = 3,
    ):
        if task_mode not in VALID_TASK_MODES:
            raise ValueError(f"Unknown task_mode: {task_mode!r} (expected one of {VALID_TASK_MODES})")

        self.task_mode = task_mode
        self.add_hs = add_hs
        self.compute_spectators = compute_spectators
        self.max_retries = max_retries

        self.data_points: List[Data] = []
        self.spectator_ratios: List[float] = []

        if task_mode == "stepwise":
            self._build_stepwise_dataset(reactions)
        else:
            self._build_end_to_end_dataset(reactions)

        print(f"✅ Loaded {len(self.data_points)} data points in '{task_mode}' mode")
        if self.spectator_ratios:
            avg_spectator = float(np.mean(self.spectator_ratios))
            print(f"   Average spectator ratio: {avg_spectator:.2%}")

    def _build_stepwise_dataset(self, reactions: List[MultiStepReaction]) -> None:
        for rxn in tqdm(reactions, desc="Building stepwise dataset"):
            for step_idx, step in enumerate(rxn.steps):
                try:
                    reactants_mol, reactants_data = process_mapped_smiles(
                        step.reactants_mapped, self.add_hs
                    )
                    products_mol, _ = process_mapped_smiles(
                        step.products_mapped, self.add_hs
                    )

                    delta = DeltaMatrixGenerator.delta_from_reactants_products(
                        reactants_mol, products_mol
                    )

                    graph_data = reactants_data
                    graph_data.y = delta

                    if self.compute_spectators:
                        spectator_mask = SpectatorDetector.identify_spectators(
                            reactants_mol, products_mol, delta
                        )
                        graph_data.spectator_mask = spectator_mask
                        self.spectator_ratios.append(
                            SpectatorDetector.compute_spectator_ratio(spectator_mask)
                        )

                    graph_data.reaction_id = rxn.reaction_id
                    graph_data.step_id = step_idx
                    graph_data.task_mode = "stepwise"

                    self.data_points.append(graph_data)
                except Exception as e:  # noqa: BLE001 - skip bad steps but log
                    print(f"⚠️  Failed to process step {step_idx} in rxn {rxn.reaction_id}: {e}")
                    continue

    def _build_end_to_end_dataset(self, reactions: List[MultiStepReaction]) -> None:
        for rxn in tqdm(reactions, desc="Building end-to-end dataset"):
            try:
                reactants_mol, reactants_data = process_mapped_smiles(
                    rxn.overall_reactants_smi, self.add_hs
                )
                products_mol, _ = process_mapped_smiles(
                    rxn.overall_products_smi, self.add_hs
                )

                delta = DeltaMatrixGenerator.delta_from_reactants_products(
                    reactants_mol, products_mol
                )
                # End-to-end Δ may exceed unit magnitude; clamp to {-2..2}.
                delta = torch.clamp(delta, -2, 2).long()

                graph_data = reactants_data
                graph_data.y = delta

                if self.compute_spectators:
                    spectator_mask = SpectatorDetector.identify_spectators(
                        reactants_mol, products_mol, delta
                    )
                    graph_data.spectator_mask = spectator_mask
                    self.spectator_ratios.append(
                        SpectatorDetector.compute_spectator_ratio(spectator_mask)
                    )

                graph_data.reaction_id = rxn.reaction_id
                graph_data.step_id = -1  # marker for full reaction
                graph_data.task_mode = "end_to_end"

                self.data_points.append(graph_data)
            except Exception as e:  # noqa: BLE001
                print(f"⚠️  Failed to process full reaction {rxn.reaction_id}: {e}")
                continue

    def __len__(self) -> int:
        return len(self.data_points)

    def __getitem__(self, idx: int) -> Data:
        return self.data_points[idx]
