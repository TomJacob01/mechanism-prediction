"""Quickstart: parse the bundled mock reaction, featurize it, compute its
Δ matrix, and run one forward pass through ``ReactionTransformer`` on CPU.

This script needs no real dataset and no GPU — its purpose is to confirm a
fresh install of ``mech_uspto`` is wired up correctly. Run it from the repo
root:

    python examples/quickstart.py

Expected output: parsed reaction summary, feature shapes, and the predicted
Δ-logit tensor shape.
"""

from __future__ import annotations

from pathlib import Path

import torch
from rdkit import Chem

from mech_uspto import (
    DeltaMatrixGenerator,
    MechUSPTOParser,
    ReactionTransformer,
)
from mech_uspto.data.featurization import process_mapped_smiles


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    fixture = repo_root / "tests" / "fixtures" / "mock_reaction.json"
    if not fixture.exists():
        raise SystemExit(f"Fixture not found: {fixture}")

    print(f"Loading bundled fixture: {fixture.relative_to(repo_root)}")

    # 1. Parse the JSON reaction file.
    reaction = MechUSPTOParser.parse_json(str(fixture))
    print(
        f"Parsed reaction {reaction.reaction_id} "
        f"with {len(reaction.steps)} step(s)."
    )

    # 2. Featurize the first step's reactants.
    step = reaction.steps[0]
    _, data = process_mapped_smiles(step.reactants_mapped)
    print(
        f"Featurization: {data.x.shape[0]} atoms, "
        f"node features {tuple(data.x.shape)}, "
        f"edge index {tuple(data.edge_index.shape)}, "
        f"edge attr {tuple(data.edge_attr.shape)}"
    )

    # 3. Compute the bond-order Δ matrix between reactants and products.
    react_mol = Chem.AddHs(Chem.MolFromSmiles(step.reactants_mapped))
    prod_mol = Chem.AddHs(Chem.MolFromSmiles(step.products_mapped))
    delta = DeltaMatrixGenerator.delta_from_reactants_products(react_mol, prod_mol)
    print(
        f"Δ matrix shape {tuple(delta.shape)}, "
        f"unique values {sorted(set(delta.flatten().tolist()))}"
    )

    # 4. One forward pass through a tiny model on CPU (batch of size 1).
    device = torch.device("cpu")
    model = ReactionTransformer(
        node_in=data.x.shape[1],
        edge_in=data.edge_attr.shape[1],
        hidden_dim=32,
        num_heads=2,
        num_layers=2,
        num_classes=3,  # stepwise mode
    ).to(device)
    model.eval()

    batch_idx = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, mask = model(
            x=data.x.to(device),
            edge_index=data.edge_index.to(device),
            edge_attr=data.edge_attr.to(device),
            batch=batch_idx,
        )
    print(f"Forward pass OK. Logits {tuple(logits.shape)}, mask {tuple(mask.shape)}")
    print("Quickstart complete.")


if __name__ == "__main__":
    main()
