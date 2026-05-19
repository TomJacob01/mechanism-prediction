# Future tasks

Backlog **not** in the current refactor. Legend: 🔴 blocker · 🟡 important · 🟢 nice-to-have · ✅ done.

---

## Open

| #    | Pri | Title                                                          | Effort  | Touches                                              |
|------|-----|----------------------------------------------------------------|---------|------------------------------------------------------|
| 1    | 🔴  | Autoregressive rollout (stepwise inf.)                         | ~1 wk   | new `inference/rollout.py`, `training/metrics.py`    |
| 1.1  | 🔴  | Define deliverable (final-Δ vs path vs product)                | ~30 m   | none — decision doc                                  |
| 1.2  | 🔴  | 10-mechanism chemistry spike (RDKit apply + sanitize)          | ~2 h    | new `scripts/rollout_spike.py`                       |
| 1.3  | 🔴  | Dataset audit: intermediates available? length distribution?   | ~1 h    | `scripts/sanity_check.py` extension                  |
| 1.4  | 🔴  | Exposure-bias measurement: one-step rollout AUC drop           | ~3 h    | new `scripts/exposure_bias_check.py`                 |
| 1.5  | 🔴  | Implementation (gated on 1.1–1.4 outcomes)                     | TBD     | new `inference/rollout.py`, `training/metrics.py`    |
| 2    | 🟢  | Hyperparameter sweep config                                    | ~2 h    | new `sweeps/*.yaml` (after H1/H2 settled)            |
| H1   | 🟢  | **Biaffine head** (see below)                                  | ~3 h    | `models/heads.py`, `models/transformer.py`, config   |
| H2   | 🟢  | **Two-stage head** (detection + class.)                        | ~half d | `models/heads.py`, `losses/focal.py`, `engine.py`    |

### Notes on the non-obvious ones

**#1 Autoregressive rollout.** The single hardest task in the backlog — much harder than the encoder. Hardness is concentrated in chemistry (applying predicted Δ to RDKit `Mol` without sanitization failures) and metrics (what counts as "correct" when mechanisms have multiple valid orderings), *not* in the model code itself. Subtasks 1.1–1.4 must complete before 1.5 — they determine ~80% of the design.

- **#1.1 Deliverable.** Three candidate questions rollout could answer:
  - **(a) Final-Δ comparison.** Roll out stepwise to convergence; compare cumulative Δ to end-to-end's direct prediction. Only the final pair-wise Δ matters. Matches the README's stepwise-vs-end-to-end FPR claim. *Sidesteps almost every hard chemistry question.* **Default recommendation.**
  - **(b) Path accuracy.** Does predicted step *k* match true step *k*? Mechanistically richer but brutally strict — mechanisms have multiple valid orderings.
  - **(c) Product validity.** Does the final molecule equal the true product? Most permissive; loses mechanistic insight.

- **#1.2 Chemistry spike.** Take 10 mechanisms; manually walk through step-by-step applying *ground-truth* Δ to RDKit (`RWMol` edits → re-perceive valence → `SanitizeMol`). Count how often sanitization fails and on what kinds of steps. Tells you whether rollout is "engineering tedium" or "open research problem" for this dataset.

- **#1.3 Dataset audit.** Confirm whether `mech-USPTO-31k.csv` actually contains per-step intermediates (separate rows or list-valued columns) vs only start + end. Get the mechanism-length distribution (median, 95th, 99th percentile) — sets `max_steps` for #1.5 and tells us if path-accuracy from #1.1(b) is even feasible.

- **#1.4 Exposure-bias check.** Take the trained stepwise model, run one rollout step on the test set, re-featurize the predicted result, measure PR-AUC on step 2. If it drops a lot, the model needs scheduled sampling / DAgger / teacher-forcing during training — meaning rollout is **not** a post-hoc inference component, it's a training-loop change. Big scope difference.

- **#1.5 Implementation.** Once 1.1–1.4 land: write `predict_full_reaction(model, smiles)` loop (predict Δ → apply to `Mol` → re-featurize → next step, until `argmax(logits)==0` everywhere or `max_steps`). Search strategy (greedy vs beam) depends on 1.1's choice. Other open design points: multi-pair-per-step actions (apply atomically vs one-at-a-time), termination (argmax vs probability threshold), sanitization failure policy (skip / backtrack / force-allow), evaluation cadence (every-epoch vs end-only).

**#2 HP sweep.** Defer until H1/H2 are settled and there's one config hitting F1 ≥ 0.5 — sweeping a broken model just finds the least-broken broken config. When the time comes: W&B Sweeps or Optuna, search `lr × weight_decay × warmup_steps × focal_gamma` around the best architectural config.

---

## Head architecture options (deferred)

These trade head parameters / wiring complexity for stronger inductive bias on the bond-change task. Try after #H0 (bias-init + edge features in head, **done**) has saturated.

### H1 — Biaffine head

Replace the concat-MLP's first linear with a per-class bilinear form:
```python
# h: (B, N, D), B_c: (C, D, D), edge_proj: nn.Linear(edge_in, C)
pair  = torch.einsum("bnd,cde,bme->bnmc", h, B_c, h)
logits = pair + edge_proj(edge_dense)              # (B, N, N, C)
logits = (logits + logits.transpose(1, 2)) / 2     # symmetrise
```

Each class `c` gets its own `D × D` interaction matrix `B_c` plus an edge-feature bias. Standard biaffine parser head (Dozat & Manning 2017) — the right inductive bias for pairwise scoring. With `D=256, C=7`: 459k bilinear params vs the current head's 133k (~3.5×; negligible vs ~3M encoder). Forward FLOPs comparable to current when `H ≈ C·D`.

**Gate.** `Config.head_type ∈ {"mlp", "biaffine"}`, default `"mlp"`. New class `BiaffineDeltaHead` in [`heads.py`](src/mech_uspto/models/heads.py); [`transformer.py`](src/mech_uspto/models/transformer.py) picks the head class.

**Test.** Shape + symmetry (mirror existing `DeltaMLP` tests), plus: `B_c = I, edge_proj = 0` ⇒ output recovers `h_i · h_j` per class.

### H2 — Two-stage head (detection + classification)

Decouple the imbalanced binary detection from the rare-class multiclass:

1. **Detector.** Sigmoid head, `P(Δ ≠ 0 | i, j)`. Trained with focal-BCE on the binarised target.
2. **Classifier.** 6-way softmax (2-way stepwise) on the non-zero classes only, `P(Δ = k | Δ ≠ 0, i, j)`. **Teacher-forced**: trained on pairs where the *true* label is non-zero (no detector gating during training).

Inference: `P(Δ=0) = 1 − P(Δ≠0)`; `P(Δ=k≠0) = P(Δ≠0) · P(Δ=k | Δ≠0)`.

**Why.** With 97.5% no-change pairs, the head is dominated by the imbalance — the classifier never gets a clean signal. Decoupling lets the detector specialise (focal-BCE + bias-init to 2.5% positive rate, exact RetinaNet) while the classifier sees a class-balanced positive-only subset.

**Cost.** ~2× head parameters; same inference FLOPs (multiply outputs); two losses with separate masks in `engine.py`.

**Gate.** `Config.head_type = "two_stage"`. New `TwoStageDeltaHead` in `heads.py`, new `TwoStageLoss` in `losses/focal.py`. Metrics need to convert `(P_detect, P_class)` → per-class distribution before the existing PR-AUC / F1 pipeline.

**When to try.** Only after #H0 (bias-init + edge features) has plateaued and the bottleneck is clearly **classifier confusion among rare classes**, not detection failure. If the model is still producing only Δ∈{−1,0,+1} after a converged run with #H0, this is the next move.

**Test.** Round-trip: build head + targets with Δ=2 everywhere on one pair; after a few hundred gradient steps verify `P(detect)→1` and `P(Δ=2 | detect)→1`, product → 1 on that pair, low elsewhere.

---

## ✅ Done

### Data + dataset
- CSV parser for `mech-USPTO-31k.csv` (31,364 rxns), replaced JSON-dir layout.
- 7 output classes for end-to-end (Δ ∈ {−3..+3} → {0..6}); `_shift_targets` handles the `+3` shift.
- `scripts/sanity_check.py` confirmed Δ=0 dominates 97.5%.
- SHA256-keyed disk cache (`cache/{task_mode}_{hash16}.pt`). 10 min cold → 1 sec warm.
- `create_dataloaders` builds **one** dataset and splits via `Subset` (was 3× re-parses).

### Cluster + sync
- `scripts/slurm/train.sbatch`: conda activation that doesn't pollute `$@`, LF endings, partition `newton`, account `cslab`, 24 h walltime, mode-aware default `--class-weights` + `--warmup-steps` injection, auto-eval + auto-plot block at end of training. Verified on Newton L40S / A40.
- Cluster sync: `sync.config.ps1` + `~/Documents/PowerShell/Tools/sync.ps1` (anchored excludes; see [`/memories/cluster.md`](../../memories/cluster.md)).
- `.gitattributes` enforces LF on `*.sbatch`, `*.sh` (Windows CRLF was breaking remote bash). `.gitignore` covers `cache/`, `slurm-logs/`.
- `CLUSTER_SETUP.md §7`: SLURM submission workflow + `mech` / `mlogs` / `mtail` / `mstat` / `msub` bashrc shortcuts.

### Correctness fixes
- **Spectator-mask shape** in focal loss: `(B,N)` per-atom → broadcast to `(B,N,N)` (pair is spectator iff both atoms are).
- **Stepwise target dtype**: `delta.long()` in `_build_stepwise_dataset` (was float, `cross_entropy` needs Long).
- **Collator mutation** (killed job 68205018 ep 2 with `pad(NoneType)`): clone PyG `Data` before stripping fields. Regression: `test_collation_does_not_mutate_input_data`.
- **`no_change_idx` hardcoded to 1** (faked F1=0.976 ep 1 on 7-class): derive `num_classes // 2`. Regressions in `test_metrics.py`.
- **PR-AUC scorer** used `probs[:, 0] + probs[:, -1]` (dropped Δ=±1, ±2 in 7-class) → `1 − probs[:, no_change_idx]`. Regression: `test_pr_auc_uses_full_non_no_change_mass_7class`.
- **PR-AUC averaging bug**: engine was averaging per-batch APs (mathematically invalid). Fixed by pooling raw `(y, score)` buffers across the epoch and computing AP once. Regression: `test_pooled_pr_auc_*`.

### Training stability
- **Resume from checkpoint**: `--resume <ckpt.pt>` on `scripts/train.py`; `engine.load_state` restores model, optimizer, history, best val loss, epoch, `global_step`.
- **Linear LR warmup**: `Config.warmup_steps` (default 0, sbatch injects 130). LR ramps 0 → target over N optimizer steps, then plateau scheduler takes over. Counted in steps so batch-size-agnostic.
- **Gentler class weights**: end-to-end default `[1, 1.5, 3, 1, 3, 1.5, 1]` (was `[1, 2, 4, 1, 4, 2, 1]` then `[2, 4, 16, 1, 16, 4, 2]`) — strong rare-class weights triggered inverted collapse (RxnPreds > targets by ep 8).
- **#8 Deterministic seeding (full).** `seed_everything(seed, deterministic=True)` in [`data/loaders.py`](src/mech_uspto/data/loaders.py) sets `PYTHONHASHSEED`, `random`, `numpy`, `torch` (CPU + all CUDA), and `cudnn.deterministic=True` / `benchmark=False`. DataLoaders pass `worker_init_fn=_seed_worker` and the train loader uses an explicit `torch.Generator().manual_seed(seed)`. `Config.deterministic: bool = True` + `--no-deterministic` CLI flag to opt out for ~5-10% throughput.
- **#12 bf16 mixed precision.** `_forward_and_loss` wrapped in `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` via `_autocast()` helper that returns `contextlib.nullcontext()` on CPU. No `GradScaler` needed. `Config.use_amp: bool = True`, gated to CUDA at runtime. `--no-amp` CLI flag for debugging.

### Metrics + eval
- **Per-class PR-AUC**: `pooled_pr_auc` returns `(overall, per_class)`, surfaced in evaluate.py output.
- **Test-set eval script**: `scripts/evaluate.py` loads checkpoint, runs held-out test split, dumps full metric bundle + confusion CSV. Auto-invoked from sbatch at end of training.
- **History plotting**: `scripts/plot_history.py` auto-invoked from sbatch.
- **First-batch progress print**: `engine.run_epoch` flushes a status line on batch 1 (in addition to every 25 batches) so the slurm log confirms training has started within ~1 s of the first forward pass.

### Head — #H0 bias-init + edge features
- `DeltaMLP(class_prior=...)`: final-layer bias = `log(π) − mean(log(π))`. Mode-aware defaults in `Config.__post_init__` (97.5% Δ=0). Step-0 softmax ≈ prior, eliminates the wake-up oscillation in epochs 1–4.
- `DeltaMLP(edge_dim=...)`: concat dense `(B, N, N, edge_in)` edge features into pair representation. `ReactionTransformer` densifies sparse `edge_attr` via `to_dense_adj`. Default on. Lets the head condition directly on current bond order — non-bonded pairs see explicit zero (the "no input bond" signal), fixing the ~500k false Δ=−1 predictions on no-change pairs from job 68209790.
- DeltaMLP symmetrises `(i,j)` / `(j,i)`; verified by existing test.
