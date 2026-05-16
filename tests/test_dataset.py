"""Tests for ``MechUSPTODataset``."""

import pytest

from mech_uspto.data.dataset import MechUSPTODataset
from mech_uspto.data.schema import MultiStepReaction


def test_stepwise_dataset_builds(stepwise_dataset):
    assert len(stepwise_dataset) >= 1
    sample = stepwise_dataset[0]
    assert sample.task_mode == "stepwise"
    assert sample.y.shape[0] == sample.y.shape[1]
    assert sample.y.shape[0] == sample.x.shape[0]
    assert hasattr(sample, "spectator_mask")
    assert sample.spectator_mask.shape[0] == sample.x.shape[0]


@pytest.mark.xfail(
    reason="Real CSV rows are full multi-step reactions; stepwise Δ ∈ [-1,1] "
    "will only hold after the parser decomposes mechanistic_label into elementary steps.",
    strict=True,
)
def test_stepwise_delta_in_unit_range(stepwise_dataset):
    sample = stepwise_dataset[0]
    assert sample.y.min().item() >= -1
    assert sample.y.max().item() <= 1


def test_end_to_end_dataset_builds(end_to_end_dataset):
    assert len(end_to_end_dataset) >= 1
    sample = end_to_end_dataset[0]
    assert sample.task_mode == "end_to_end"
    assert sample.step_id == -1
    assert sample.y.dtype.is_floating_point is False  # clamped to long


def test_end_to_end_delta_in_extended_range(end_to_end_dataset):
    sample = end_to_end_dataset[0]
    assert sample.y.min().item() >= -3
    assert sample.y.max().item() <= 3


def test_invalid_task_mode_raises(mock_reaction):
    with pytest.raises(ValueError):
        MechUSPTODataset([mock_reaction], task_mode="bogus")


def test_empty_steps_list_yields_empty_dataset(mock_reaction):
    rxn = MultiStepReaction(
        reaction_id="empty",
        steps=[],
        overall_reactants_smi="CCO",
        overall_products_smi="CCO",
    )
    ds = MechUSPTODataset([rxn], task_mode="stepwise", compute_spectators=False)
    assert len(ds) == 0
