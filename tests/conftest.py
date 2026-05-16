"""Shared pytest fixtures."""

from pathlib import Path

import pytest

from mech_uspto.data.dataset import MechUSPTODataset
from mech_uspto.data.parser import MechUSPTOParser
from mech_uspto.data.schema import MultiStepReaction

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_CSV_PATH = FIXTURES_DIR / "sample_reactions.csv"


@pytest.fixture
def sample_csv_path() -> Path:
    """Path to the bundled real-data CSV fixture (2 rows from figshare)."""
    return SAMPLE_CSV_PATH


@pytest.fixture
def sample_reactions() -> list[MultiStepReaction]:
    """All reactions parsed from the sample CSV fixture."""
    return MechUSPTOParser.parse_csv_file(str(SAMPLE_CSV_PATH))


@pytest.fixture
def mock_reaction(sample_reactions) -> MultiStepReaction:
    """First reaction from the sample CSV (used for downstream tests)."""
    return sample_reactions[0]


@pytest.fixture
def stepwise_dataset(mock_reaction) -> MechUSPTODataset:
    return MechUSPTODataset([mock_reaction], task_mode="stepwise", compute_spectators=True)


@pytest.fixture
def end_to_end_dataset(mock_reaction) -> MechUSPTODataset:
    return MechUSPTODataset([mock_reaction], task_mode="end_to_end", compute_spectators=True)
