"""Tests for graph featurization helpers."""

import pytest
import torch

from mech_uspto.constants import EDGE_FEATURE_DIM, NODE_FEATURE_DIM
from mech_uspto.data.featurization import (
    one_hot_encode,
    process_mapped_smiles,
)


def test_one_hot_encode_known_value():
    encoding = one_hot_encode("b", ["a", "b", "c"])
    assert encoding == [0, 1, 0, 0]  # 3 + unknown bin


def test_one_hot_encode_unknown_value_uses_unknown_bin():
    encoding = one_hot_encode("z", ["a", "b", "c"])
    assert encoding == [0, 0, 0, 1]


def test_process_mapped_smiles_returns_correct_shapes():
    mol, data = process_mapped_smiles("[CH3:1][OH:2]", add_hs=True)
    assert mol.GetNumAtoms() == data.x.shape[0]
    assert data.x.shape[1] == NODE_FEATURE_DIM
    if data.edge_attr.numel() > 0:
        assert data.edge_attr.shape[1] == EDGE_FEATURE_DIM
    assert data.x.dtype == torch.float


def test_process_mapped_smiles_invalid_raises():
    with pytest.raises(ValueError):
        process_mapped_smiles("not a real smiles ###", add_hs=False)
