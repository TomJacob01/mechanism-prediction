# mech_uspto
Dual-mode (stepwise vs end-to-end) ablation study on the **mech-USPTO-31k**
multi-step reaction dataset. Refactor of an earlier PMechDB proof-of-concept
notebook into an installable Python package.

---

## Research design

**Question.** Can a graph transformer learn multi-step chemistry without ever
seeing the intermediate states? Two training modes (stepwise vs. end-to-end)
target the same forward operator at different time scales, sharing the same
model code and differing only in supervision signal.

Full framing, architecture, loss, metrics, and design rationale: see
[MODEL.md](MODEL.md). Open work: [FUTURE_TASKS.md](FUTURE_TASKS.md).

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
src/mech_uspto/{constants,data,models,losses,training}
scripts/{train,evaluate,sanity_check,plot_history}.py
tests/                  # pytest suite + fixtures
legacy/PmechDB_POC.ipynb
```

Public API is re-exported from the top-level package:

```python
from mech_uspto import (
    MechUSPTOParser, MechUSPTODataset, create_dataloaders,
    DeltaMLP, ReactionTransformer, MaskedFocalLossWithSpectators,
)
```

Per-module breakdown and architecture details: [MODEL.md](MODEL.md).
Data format and how to obtain it: [DATA.md](DATA.md).

---

## Cluster deployment

Training is GPU-agnostic and runs on any single-node CUDA setup.
SLURM submission workflow + sync setup: [CLUSTER_SETUP.md](CLUSTER_SETUP.md).
