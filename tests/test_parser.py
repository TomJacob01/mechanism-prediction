"""Tests for ``MechUSPTOParser`` (CSV format)."""

import pytest

from mech_uspto.data.parser import MechUSPTOParser


# ---------------------------------------------------------------------------
# Inline CSV-row tests (fast, no file IO)
# ---------------------------------------------------------------------------

@pytest.fixture
def csv_row_simple():
    """Single-step reaction (synthetic) used to exercise parse_csv_row."""
    # Quote the mechanism label because it contains commas.
    return (
        '[O:1][CH3:2]>>[OH:1].[CH3:2],'
        '[O:1][CH3:2]>>[OH:1].[CH3:2],'
        'homolysis,'
        '"[(1, 2)]"'
    )


def test_parse_csv_row_returns_multi_step_reaction(csv_row_simple):
    rxn = MechUSPTOParser.parse_csv_row(csv_row_simple)
    assert rxn.reaction_id is not None
    assert len(rxn.steps) >= 1


def test_parse_csv_row_preserves_atom_maps(csv_row_simple):
    rxn = MechUSPTOParser.parse_csv_row(csv_row_simple)
    for step in rxn.steps:
        assert ":" in step.reactants_mapped
        assert ":" in step.products_mapped


def test_parse_csv_row_records_metadata(csv_row_simple):
    rxn = MechUSPTOParser.parse_csv_row(csv_row_simple)
    assert rxn.metadata["mechanistic_class"] == "homolysis"
    assert rxn.metadata["mechanistic_label"] == "[(1, 2)]"


def test_parse_csv_row_rejects_missing_arrow():
    bad_line = "junk,still_junk,class,[(1,2)]"
    with pytest.raises(ValueError, match=r"'>>'"):
        MechUSPTOParser.parse_csv_row(bad_line)


def test_parse_csv_row_rejects_too_few_columns():
    with pytest.raises(ValueError, match=r"at least 4 columns"):
        MechUSPTOParser.parse_csv_row("only,three,cols")


# ---------------------------------------------------------------------------
# Real-data file tests (read the actual figshare CSV sample on disk)
# ---------------------------------------------------------------------------

def test_parse_csv_file_loads_all_rows(sample_csv_path):
    """Parsing the real sample CSV yields one MultiStepReaction per row."""
    reactions = MechUSPTOParser.parse_csv_file(str(sample_csv_path))
    assert len(reactions) == 2  # sample_reactions.csv has 2 rows


def test_parse_csv_file_assigns_deterministic_ids(sample_csv_path):
    reactions = MechUSPTOParser.parse_csv_file(str(sample_csv_path))
    assert reactions[0].reaction_id == "rxn_000000"
    assert reactions[1].reaction_id == "rxn_000001"


def test_parse_csv_file_preserves_mechanistic_class(sample_csv_path):
    reactions = MechUSPTOParser.parse_csv_file(str(sample_csv_path))
    classes = [r.metadata["mechanistic_class"] for r in reactions]
    assert classes == ["Cbz_deprotection", "DCC_condensation"]


def test_parse_csv_file_real_data_has_atom_mapped_smiles(sample_csv_path):
    reactions = MechUSPTOParser.parse_csv_file(str(sample_csv_path))
    for r in reactions:
        step = r.steps[0]
        assert ":" in step.reactants_mapped
        assert ":" in step.products_mapped
        # The updated_reaction column uses many heavy atoms; sanity check length.
        assert len(step.reactants_mapped) > 50
        assert len(step.products_mapped) > 50
