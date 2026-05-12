"""Parser for mech-USPTO-31k JSON files."""

import json
from pathlib import Path
from typing import List

from tqdm.auto import tqdm

from mech_uspto.data.schema import MultiStepReaction, ReactionStep


class MechUSPTOParser:
    """Parser for mech-USPTO-31k JSON / MechSMILES format.

    Expected JSON structure::

        {
            "rxn_id": "rxn_0001",
            "steps": [
                {
                    "id": 0,
                    "reactants": "...",
                    "products": "...",
                    "reactants_mapped": "...",
                    "products_mapped": "...",
                    "mechanism": "..."
                },
                ...
            ],
            "overall_reactants": "...",
            "overall_products": "...",
            "metadata": {...}
        }
    """

    @staticmethod
    def parse_json(json_path: str) -> MultiStepReaction:
        """Parse a single mech-USPTO-31k reaction from disk."""
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        steps = [
            ReactionStep(
                step_id=step_data["id"],
                reactants_smi=step_data.get("reactants", ""),
                products_smi=step_data.get("products", ""),
                reactants_mapped=step_data.get("reactants_mapped", ""),
                products_mapped=step_data.get("products_mapped", ""),
                mechanism_arrow=step_data.get("mechanism", ""),
            )
            for step_data in data.get("steps", [])
        ]

        return MultiStepReaction(
            reaction_id=data["rxn_id"],
            steps=steps,
            overall_reactants_smi=data.get("overall_reactants", ""),
            overall_products_smi=data.get("overall_products", ""),
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def parse_batch(json_dir: str) -> List[MultiStepReaction]:
        """Parse every ``*.json`` file in ``json_dir``.

        Files that fail to parse are logged and skipped.
        """
        reactions: List[MultiStepReaction] = []
        json_files = sorted(Path(json_dir).glob("*.json"))

        for json_file in tqdm(json_files, desc="Parsing mech-USPTO-31k"):
            try:
                reactions.append(MechUSPTOParser.parse_json(str(json_file)))
            except Exception as e:  # noqa: BLE001 - surface but skip bad files
                print(f"⚠️  Failed to parse {json_file}: {e}")
                continue

        return reactions
