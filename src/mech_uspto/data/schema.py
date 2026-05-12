"""Dataclasses representing a mech-USPTO-31k reaction."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ReactionStep:
    """A single elementary step in a multi-step reaction."""

    step_id: int
    reactants_smi: str
    products_smi: str
    reactants_mapped: str
    products_mapped: str
    mechanism_arrow: str  # e.g. "bond_break|proton_shift" or arrow code


@dataclass
class MultiStepReaction:
    """A complete multi-step reaction from mech-USPTO-31k."""

    reaction_id: str
    steps: List[ReactionStep]
    overall_reactants_smi: str
    overall_products_smi: str
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Ensure steps are always ordered by step_id.
        self.steps = sorted(self.steps, key=lambda s: s.step_id)
