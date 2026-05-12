"""Shared pytest fixtures."""

import json
import shutil
from pathlib import Path

import pytest

from mech_uspto.data.dataset import MechUSPTODataset
from mech_uspto.data.schema import MultiStepReaction, ReactionStep

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MOCK_REACTION_PATH = FIXTURES_DIR / "mock_reaction.json"


@pytest.fixture
def mock_reaction_dict() -> dict:
    """Raw JSON dict for the mock reaction."""
    with open(MOCK_REACTION_PATH) as f:
        return json.load(f)


@pytest.fixture
def mock_reaction(mock_reaction_dict) -> MultiStepReaction:
    """Mock reaction parsed into a ``MultiStepReaction``."""
    return MultiStepReaction(
        reaction_id=mock_reaction_dict["rxn_id"],
        steps=[ReactionStep(
            step_id=s["id"],
            reactants_smi=s["reactants"],
            products_smi=s["products"],
            reactants_mapped=s["reactants_mapped"],
            products_mapped=s["products_mapped"],
            mechanism_arrow=s["mechanism"],
        ) for s in mock_reaction_dict["steps"]],
        overall_reactants_smi=mock_reaction_dict["overall_reactants"],
        overall_products_smi=mock_reaction_dict["overall_products"],
        metadata=mock_reaction_dict.get("metadata", {}),
    )


@pytest.fixture
def tmp_json_dir(tmp_path, mock_reaction_dict) -> Path:
    """Temp dir containing a single mock reaction JSON file."""
    target = tmp_path / "rxn_test_001.json"
    shutil.copy(MOCK_REACTION_PATH, target)
    return tmp_path


@pytest.fixture
def stepwise_dataset(mock_reaction) -> MechUSPTODataset:
    return MechUSPTODataset([mock_reaction], task_mode="stepwise", compute_spectators=True)


@pytest.fixture
def end_to_end_dataset(mock_reaction) -> MechUSPTODataset:
    return MechUSPTODataset([mock_reaction], task_mode="end_to_end", compute_spectators=True)
