# mech_uspto
Dual-mode (stepwise vs end-to-end) ablation study on the **mech-USPTO-31k**
multi-step reaction dataset. Refactor of an earlier PMechDB proof-of-concept
notebook into an installable Python package.

---

## Research design

**Question.** Can a graph transformer learn multi-step chemistry without ever
seeing the intermediate states?

| Aspect              | Stepwise (micro)                       | End-to-end (macro)                    |
| ------------------- | -------------------------------------- | ------------------------------------- |
| Training data       | Each elementary step ``S_i → S_{i+1}`` | Full reaction ``S_0 → S_final``       |
| Target Δ range      | ``{-1, 0, 1}``                         | ``{-2, -1, 0, 1, 2}``                 |
| Classification head | 3-class                                | 5-class                               |
| Inference           | Autoregressive rollout                 | Single-shot                           |
| Chemical principle  | Step-by-step                           | Implicit "chemical teleportation"     |

**Headline metric.** Final Product Recovery (FPR): fraction of reactions whose
predicted final adjacency matrix matches the ground truth.

---

## Install

Two supported paths. Pick **conda** if you want a one-shot install that
handles RDKit cleanly; pick **pip** if you already have a CUDA-matched
PyTorch wheel installed.

### Option A — conda (recommended)

```bash
conda env create -f environment.yml
conda activate mechuspto
pip install -e ".[dev]"          # editable install + linting/test tools
```

For GPU, edit `environment.yml` and replace `cpuonly` with the matching
`pytorch-cuda=XX.X` package from the
[PyTorch matrix](https://pytorch.org/get-started/locally/).

### Option B — venv + pip

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Unix:    source .venv/bin/activate

# GPU users: install a CUDA-matched torch wheel FIRST, then:
pip install -e ".[dev]"
```

See [`DATA.md`](DATA.md) for how to obtain the mech-USPTO-31k dataset.

---

## Quick start

```powershell
# Run the test suite (uses synthetic mock data, no GPU required)
pytest -q

# Pre-flight sanity check on a real data drop (env, schema, parsing,
# featurization, Δ ranges, spectator stats, mini training step)
python scripts/sanity_check.py --data-dir <path/to/mech-USPTO-31k>

# Train (stepwise mode)
python scripts/train.py --task-mode stepwise --batch-size 32 --num-epochs 100

# Train (end-to-end mode)
python scripts/train.py --task-mode end_to_end --batch-size 32 --num-epochs 100
```

The data directory defaults to the value of the `MECH_USPTO_DATA` environment
variable, falling back to `./data/mech-USPTO-31k`. Override with
`--data-dir <path>`.

---

## Package layout

```
src/mech_uspto/
├── constants.py             # ALLOWED_ELEMENTS, bond types, feature widths
├── data/
│   ├── schema.py            # ReactionStep, MultiStepReaction
│   ├── parser.py            # MechUSPTOParser
│   ├── featurization.py     # one_hot_encode, featurize_nodes/edges, process_mapped_smiles
│   ├── transformations.py   # DeltaMatrixGenerator
│   ├── spectators.py        # SpectatorDetector
│   ├── dataset.py           # MechUSPTODataset
│   └── loaders.py           # collate_fn_with_spectators, create_dataloaders
├── models/
│   ├── heads.py             # DeltaMLP (3- or 5-class)
│   └── transformer.py       # ReactionTransformer (TransformerConv encoder + head)
├── losses/
│   └── focal.py             # MaskedFocalLossWithSpectators
└── training/
    ├── config.py            # Config dataclass
    ├── metrics.py           # MetricsComputer + per-sample helper
    └── engine.py            # TrainingEngine

scripts/train.py             # CLI entrypoint (argparse + main)
tests/                       # pytest suite + fixtures
legacy/PmechDB_POC.ipynb     # archived original notebook
```

The public API is re-exported from the top-level package, so consumer code
can keep importing from `mech_uspto`:

```python
from mech_uspto import (
    MechUSPTOParser,
    MechUSPTODataset,
    create_dataloaders,
    DeltaMLP,
    ReactionTransformer,
    MaskedFocalLossWithSpectators,
)
```

---

## Data format

Each reaction is one JSON file:

```json
{
    "rxn_id": "rxn_0001",
    "steps": [
        {
            "id": 0,
            "reactants": "SMILES",
            "products": "SMILES",
            "reactants_mapped": "atom-mapped SMILES",
            "products_mapped": "atom-mapped SMILES",
            "mechanism": "label or arrow code"
        }
    ],
    "overall_reactants": "SMILES",
    "overall_products": "SMILES",
    "metadata": {"...": "..."}
}
```

Featurization produces 25-dim node features and 6-dim edge features (see
`mech_uspto.constants`). Targets are bond-order Δ matrices of shape
`(N, N)` per sample, padded to `(B, N_max, N_max)` at collation time and
shifted to non-negative class indices inside the training engine.

---

## Key design decisions

1. **Spectator downweighting (not masking).** ~95% of USPTO atoms are
   spectators. Zeroing them out destroys global graph signal, so the loss
   downweights them by `0.1` instead — see `MaskedFocalLossWithSpectators`.
2. **Symmetric Δ predictions.** Bond formation is symmetric (`A-B == B-A`),
   so the head averages `logits[i, j]` and `logits[j, i]`.
3. **Reactants-only inputs.** Both modes feed the encoder the *reactant*
   side (`S_i` for stepwise, `S_0` for end-to-end) — the model is never
   shown products at training time.

---

## Performance expectations (rough, from the original POC)

| Mode       | Final F1 | Final FPR  |
| ---------- | -------- | ---------- |
| Stepwise   | ~0.85    | 70–85 %    |
| End-to-end | ~0.70    | 40–60 %    |

End-to-end converges more slowly but tests whether the model can implicitly
learn intermediate states.

---

## Cluster deployment

See `DEPLOYMENT.md` for SLURM templates, env setup, and monitoring tips on
the cluster (8× A100 partition).

---

## Status & follow-ups

This README replaces the previous `INDEX.md`, `MIGRATION_README.md` and
`SUMMARY.py`, which are no longer maintained. The original PMechDB POC
notebook is preserved at `legacy/PmechDB_POC.ipynb` for reference.

Known follow-ups (out of scope for this refactor):

- Replace broad `except Exception` blocks with specific exceptions, and tally
  dropped samples in dataset construction.
- Replace `print` + emoji output with the `logging` module.
- Add an end-to-end integration test on a tiny synthetic dataset.
- Implement an autoregressive inference loop for stepwise mode and an FPR
  evaluation script.
