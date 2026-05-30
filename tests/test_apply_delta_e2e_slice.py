"""Regression slice for apply_delta on real USPTO reactions.

Runs the same analysis the e2e verifier does, but on the first 20 reactions
only, and asserts the recoverable pass rate stays above a floor. Turns the
e2e smoke into a CI gate so any future change that breaks bond-Δ application
trips immediately.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Import the verifier as a module (it's a script). The script's _analyse_reaction
# encapsulates the parse/align/Δ-build/apply/compare pipeline.
import importlib.util
import sys

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_apply_delta_e2e.py"
_spec = importlib.util.spec_from_file_location("_verify_e2e", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["_verify_e2e"] = _mod
_spec.loader.exec_module(_mod)

from mech_uspto.data.parser import MechUSPTOParser  # noqa: E402


_CSV = Path(__file__).resolve().parents[1] / "data" / "mech-USPTO-31k.csv"


@pytest.mark.skipif(not _CSV.exists(), reason="USPTO csv not present")
def test_e2e_slice_recoverable_pass_rate():
    reactions = MechUSPTOParser.parse_csv_file(str(_CSV))[:20]
    assert len(reactions) == 20, "expected at least 20 reactions in the dataset"
    statuses: dict[str, int] = {}
    for rxn in reactions:
        result = _mod._analyse_reaction(rxn)
        statuses[result["status"]] = statuses.get(result["status"], 0) + 1
    n_ok = statuses.get("ok", 0)
    n_recov = statuses.get("ok_recoverable", 0)
    recoverable_rate = (n_ok + n_recov) / len(reactions)
    assert recoverable_rate >= 0.95, (
        f"recoverable pass rate {recoverable_rate:.2%} dropped below 95% floor. "
        f"Statuses: {statuses}"
    )
