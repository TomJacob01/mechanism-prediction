"""Tests for ``MechUSPTOParser``."""

import json

import pytest

from mech_uspto.data.parser import MechUSPTOParser


def test_parse_json_returns_reaction(tmp_json_dir):
    rxn_path = tmp_json_dir / "rxn_test_001.json"
    rxn = MechUSPTOParser.parse_json(str(rxn_path))

    assert rxn.reaction_id == "rxn_test_001"
    assert len(rxn.steps) == 1
    assert rxn.steps[0].step_id == 0
    assert rxn.steps[0].reactants_mapped.startswith("[CH3:1]")
    assert rxn.overall_reactants_smi == "CCO.CC(=O)Cl"


def test_parse_batch_loads_all_files(tmp_json_dir, mock_reaction_dict):
    # Add a second reaction to confirm batch parsing.
    second = dict(mock_reaction_dict)
    second["rxn_id"] = "rxn_test_002"
    with open(tmp_json_dir / "rxn_test_002.json", "w") as f:
        json.dump(second, f)

    reactions = MechUSPTOParser.parse_batch(str(tmp_json_dir))
    ids = sorted(r.reaction_id for r in reactions)
    assert ids == ["rxn_test_001", "rxn_test_002"]


def test_parse_batch_skips_malformed_files(tmp_json_dir):
    # Drop an invalid JSON file alongside the valid one.
    (tmp_json_dir / "broken.json").write_text("{not valid json")

    reactions = MechUSPTOParser.parse_batch(str(tmp_json_dir))
    assert len(reactions) == 1
    assert reactions[0].reaction_id == "rxn_test_001"


def test_parse_json_missing_rxn_id_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"steps": []}))
    with pytest.raises(KeyError):
        MechUSPTOParser.parse_json(str(bad))
