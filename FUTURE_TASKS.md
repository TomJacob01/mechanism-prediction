# Future tasks

Backlog of work that is **not** part of the current refactor. Ordered roughly
by impact, with rough effort estimates. Each item has enough detail that you
(or a contributor) can pick it up cold.

Legend: 🔴 blocker for serious training • 🟡 important • 🟢 nice-to-have

---

## 🔴 1. Test-set evaluation script

**What.** `scripts/evaluate.py` that loads a checkpoint, runs the held-out
test split (`train_val_test_split=(0.7, 0.15, 0.15)`), and reports the full
metric bundle plus the headline **Final Product Recovery (FPR)** number.

**Why.** Right now `engine.train()` only computes train + val metrics. The
test split is created by `create_dataloaders` but never used. You can't
report results in a paper without held-out test numbers.

**API sketch.**
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/stepwise_best_ep42.pt \
    --data-dir $MECH_USPTO_DATA \
    --split test \
    --output results/stepwise_test_metrics.json
```

**Touches.** `scripts/evaluate.py` (new), maybe a `MetricsComputer.summary()`
helper that aggregates per-batch metrics into per-dataset numbers.

---

## 🔴 2. Autoregressive rollout for stepwise inference

**What.** A function `predict_full_reaction(model, reactants_smiles)` that
applies the stepwise model repeatedly: predict step 0 → apply Δ →
re-featurize → predict step 1 → … until convergence (no predicted bond
changes) or a max-step budget.

**Why.** Stepwise mode currently can't predict full reactions end-to-end at
inference time, which makes the stepwise-vs-end-to-end FPR comparison in the
README's "Performance expectations" table impossible. Without this, the
ablation study cannot conclude.

**Subtasks.**
- Δ → SMILES: apply a predicted Δ adjacency to an RDKit `Mol` (set bond
  orders, sanitize, reassign Hs).
- Termination criterion: stop when `argmax(logits) == 0` everywhere, or
  after `max_steps` (default 10).
- Beam search variant (optional): keep top-k predictions per step.

**Touches.** new `mech_uspto/inference/rollout.py`, plus an FPR metric in
`training/metrics.py`.

---

## 🔴 3. Data download + format conversion

**What.** Once we know the on-disk format of the figshare archive (see
[`DATA.md`](DATA.md)), write `scripts/download_data.py` and (likely)
`scripts/convert_to_json.py` that turns the upstream tabular release into the
per-reaction JSON layout `MechUSPTOParser` expects.

**Why.** The dataset is the precondition for everything. Without it, no
sanity check, no training, no eval.

**Blocked on.** A human (you) needs to inspect the figshare contents first
and report the actual file format. I won't guess.

---

## 🟡 4. Resume-from-checkpoint flag

**What.** Add `--resume <checkpoint.pt>` to `scripts/train.py`. The
checkpoint already contains `model_state_dict`, `optimizer_state_dict`,
`history`, `best_val_loss`, and `epoch` — restoration is a 10-line change
inside `TrainingEngine.__init__` (or a new `from_checkpoint` classmethod).

**Why.** Cluster jobs get pre-empted. Without resume, a wall-clock kill
loses everything past the last "latest" snapshot.

**Subtasks.**
- Validate `config` matches between checkpoint and current run (warn on
  mismatch, error on incompatible mismatch like `task_mode`).
- Continue `epoch` numbering from where it left off so logs and PR-AUC
  curves line up.

**Touches.** `scripts/train.py`, `training/engine.py`.

---

## 🟡 5. Expose `--checkpoint-dir` in the CLI

**What.** `Config.checkpoint_dir` exists but `scripts/train.py` doesn't
forward a CLI flag for it. Currently checkpoints always land in
`./checkpoints/` relative to the launch directory.

**Why.** On clusters you typically want checkpoints on `$SCRATCH` (fast,
purged) and results on `$HOME` (persistent, slow).

**Touches.** `scripts/train.py` — 3 lines.

---

## 🟡 6. Experiment tracking integration

**What.** Optional W&B (or TensorBoard, or MLflow) hook in
`TrainingEngine.train()`. Behind a `--tracker {none,wandb,tensorboard}`
flag with `none` as default so it stays a soft dependency.

**Why.** Lets you compare runs visually, share dashboards, and run
hyperparameter sweeps. See the explanation in the chat preceding this
file's creation.

**Subtasks.**
- `pip install wandb` as an optional extra in `pyproject.toml`:
  `tracking = ["wandb>=0.16"]`.
- One `tracker.init(vars(config))` call at start of `train()`.
- One `tracker.log({...val_metrics, "epoch": epoch})` per epoch.
- Tracker abstraction in `training/tracking.py` so swapping backends is
  trivial (W&B / TensorBoard / no-op).

**Touches.** `training/tracking.py` (new), `training/engine.py`,
`scripts/train.py`, `pyproject.toml`.

---

## 🟡 7. Replace `print` + emoji with `logging`

**What.** Swap every `print("📂 ...")` for `logger.info("...")` with a
module-level `logger = logging.getLogger(__name__)`. Configure the root
logger in `scripts/train.py` and `scripts/sanity_check.py` based on a
`--log-level` flag (default `INFO`).

**Why.** Cluster logs are easier to grep without emoji, log levels let users
silence noisy output, and you can plug in a file handler for permanent run
logs.

**Touches.** `data/loaders.py`, `data/dataset.py`, `training/engine.py`,
`scripts/*`. Mechanical change but spans most files.

---

## 🟡 8. Deterministic seeding (full)

**What.** Beyond `np.random.seed` + `torch.manual_seed`, also set:
- `torch.cuda.manual_seed_all(seed)`
- `torch.backends.cudnn.deterministic = True` (with a documented speed cost)
- `torch.backends.cudnn.benchmark = False`
- DataLoader `worker_init_fn` that seeds each worker from `seed + worker_id`
- `generator = torch.Generator().manual_seed(seed)` passed to the DataLoader
  for shuffling

**Why.** With `num_workers=4` and unseeded workers, your data ordering and
augmentation (if any) are non-deterministic across runs even with the same
top-level seed. Reproducibility for the paper requires this.

**Touches.** `data/loaders.py`, `scripts/train.py`. ~30 lines.

---

## 🟡 9. Tighten broad `except Exception` blocks

**What.** `MechUSPTOParser.parse_batch` and `MechUSPTODataset.__init__`
catch bare `Exception`, log, and skip. Replace with the specific exceptions
you actually expect (`json.JSONDecodeError`, `KeyError`, RDKit
`Chem.AtomKekulizeException`, `ValueError`) and **count** how many were
dropped — print a summary at the end of dataset construction.

**Why.** A silent 30% drop rate would currently go unnoticed. Bare
`except Exception` also masks real bugs (e.g. a typo in an attribute name
becomes a "skipped sample" instead of a crash).

**Touches.** `data/parser.py`, `data/dataset.py`. ~20 lines.

---

## 🟡 10. End-to-end integration test

**What.** A pytest test that runs 1 epoch of training on a tiny fixture
(say 10 mock reactions), asserts loss decreases, and asserts a checkpoint
file is written.

**Why.** Unit tests catch regressions in individual functions but not in
the wiring between dataloader → model → loss → optimizer → checkpoint.
This is the test that would have caught the `spectator_padded` shape bug
before you noticed it manually.

**Touches.** `tests/test_training_loop.py` (new), maybe a larger
`tests/fixtures/mock_reactions/` directory.

---

## 🟢 11. Configurable model + loss from CLI / YAML

**What.** Currently `scripts/train.py` accepts `--hidden-dim` but not
`--num-layers`, `--num-heads`, `--dropout`, `--gamma-focal`, etc. Either
expose them all as flags or accept a `--config config.yaml` file.

**Why.** Hyperparameter sweeps need it. YAML is friendlier than 15 CLI
flags.

**Touches.** `scripts/train.py`, optionally `training/config.py` to add a
`from_yaml` classmethod.

---

## 🟢 12. Mixed-precision training

**What.** Wrap forward+loss in `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`
and use `torch.amp.GradScaler` for fp16 (or skip the scaler with bf16).

**Why.** ~2× speedup on A100s, half the memory. Lets you fit larger
batches on the same GPU.

**Caveat.** The focal loss currently uses `torch.where(spectator_flat,
0.1, 1.0)` — verify it survives autocast. The unit tests should cover this.

**Touches.** `training/engine.py`. ~10 lines.

---

## 🟢 13. Distributed / multi-GPU training

**What.** Wrap the model in `torch.nn.parallel.DistributedDataParallel`,
launch with `torchrun --nproc_per_node=8`. Adjust the dataloader to use
`DistributedSampler`.

**Why.** With 31k reactions × ~100 epochs, single-A100 training is hours
not days, so this is genuinely 🟢 (nice-to-have) — but on the 8×A100
partition mentioned in `DEPLOYMENT.md`, you'd be wasting 7 GPUs.

**Touches.** `scripts/train.py`, `training/engine.py`,
`data/loaders.py`.

---

## 🟢 14. Hyperparameter sweep config

**What.** A `sweeps/stepwise.yaml` describing a W&B sweep (or Optuna study)
over `hidden_dim ∈ {128, 256, 512}`, `num_layers ∈ {3, 6, 9}`,
`learning_rate` log-uniform `[1e-5, 1e-3]`, `gamma_focal ∈ {1, 2, 2.5, 4}`.

**Why.** Currently the hyperparameters are inherited from the POC. Light
tuning probably moves F1 / PR-AUC by a few points.

**Depends on.** Item 6 (experiment tracking) and item 11 (config
externalization).

---

## 🟢 15. Pretrained checkpoint distribution

**What.** Once you have a result you're happy with, upload the best
checkpoint as a GitHub Release asset (or to HuggingFace Hub) and add a
`scripts/predict.py` that downloads + caches it on first use.

**Why.** Lets others use the model without retraining (which requires
GPU + a day).

**Touches.** new `scripts/predict.py`, GitHub Release process.

---

## 🟢 16. Inference / serving API

**What.** Expose a clean Python API:

```python
from mech_uspto.inference import MechanismPredictor
predictor = MechanismPredictor.from_checkpoint("stepwise_best.pt")
result = predictor.predict("CCO.CC(=O)Cl")
# result: {"steps": [...], "final_smiles": "CCOC(C)=O.Cl",
#          "confidence": 0.87, "fpr_match": True}
```

Optionally a tiny FastAPI server for HTTP predictions.

**Depends on.** Item 2 (rollout).

---

## 🟢 17. Better Δ-matrix visualization

**What.** A `notebooks/visualize_predictions.ipynb` (or
`scripts/visualize.py`) that takes a checkpoint + a reaction and plots
the predicted vs ground-truth Δ matrix as a heatmap, overlaid on the 2D
molecule structure.

**Why.** Debugging-by-eyeballing. Also good figure material for a paper.

---

## 🟢 18. CI hardening

**What.** Add to `.github/workflows/ci.yml`:
- A coverage report (`pytest --cov=mech_uspto --cov-report=xml` + Codecov upload).
- A separate "build" job that does `python -m build` to catch packaging regressions.
- Cache the heavy CPU torch + PyG + RDKit installs across runs (the current cache key only covers `pyproject.toml`, not the wheels themselves).

---

## 🟢 19. Type checking

**What.** Add `mypy` (or `pyright`) to the dev extras and a separate CI job.
Start in lenient mode (`--ignore-missing-imports`) since torch / PyG type
stubs are incomplete; tighten over time.

**Why.** Catches a class of bugs that ruff doesn't, especially around
tensor shapes and `Optional[...]` handling.

---

## 🟢 20. Docs site

**What.** A small Sphinx or MkDocs site under `docs/`, auto-deployed to
GitHub Pages. Auto-generate API docs from docstrings.

**Why.** Probably overkill for a research repo, but worth it if the package
graduates to "tool other people use."

---

## Cross-cutting non-functional debt

- **Hardcoded paths.** `DEFAULT_DATA_DIR = "/cluster/data/mech-USPTO-31k"`
  is a Newton-cluster-specific guess. Drop the fallback and require either
  `MECH_USPTO_DATA` or `--data-dir`.
- **Hardcoded performance numbers in README.** The "Final F1 ~0.85" table
  is from the POC notebook — re-validate after the first real test-set run.
- **No `version.py`.** `mech_uspto.__version__` doesn't exist yet; users
  can't programmatically check what they're running. Add `_version.py` and
  expose from `__init__.py`.
