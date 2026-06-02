# Model & design

End-to-end walkthrough of the `mech_uspto` architecture, training loop, and the rationale behind every non-obvious choice. Companion to [README.md](../README.md) (install, quickstart) and [FUTURE_TASKS.md](FUTURE_TASKS.md) (backlog).

---

## 1. Research question and the dual-mode framing

**Question.** Can a graph transformer learn multi-step chemistry without ever seeing intermediate states?

Two training modes target the same forward operator (`reactants → product graph changes`) at different time scales:

| Aspect              | Stepwise                              | End-to-end                            |
| ------------------- | ------------------------------------- | ------------------------------------- |
| Training sample     | One elementary step `Sᵢ → Sᵢ₊₁`       | Full reaction `S₀ → S_final`          |
| Target Δ range      | `{-1, 0, 1}`                          | `{-3, -2, -1, 0, 1, 2, 3}`            |
| Classification head | 3-class                               | 7-class                               |
| Inference           | Autoregressive rollout (TODO #1)      | Single-shot                           |
| Bias                | Step-by-step mechanistic              | Implicit "chemical teleportation"     |

Both modes share the *same model code, featurization, loss, and metrics* — `task_mode` only affects target construction, the head's output dimension, and the class-prior used for bias initialisation. This is what makes the ablation honest: the **only** thing that varies is the supervision signal.

---

## 2. Data representation

### 2.1 From SMILES to graph

1. **Parse** atom-mapped reactant + product SMILES with RDKit.
2. **Featurise** atoms and bonds:
   - Node features (25-dim, [constants.py](src/mech_uspto/constants.py)): element one-hot, formal charge, degree, hybridisation one-hot, aromatic flag, in-ring flag, H-count, chiral tag, radical electrons.
   - Edge features (6-dim): bond-type one-hot (single/double/triple/aromatic), conjugation flag, ring-membership flag.
3. **Δ-matrix target.** For each upper-triangle atom pair `(i, j)` with `i < j`, target = `bond_order_after − bond_order_before` ∈ `{-1, 0, 1}` per step (stepwise) or summed `∈ {-3..3}` (end-to-end).

### 2.2 Spectator atoms

Reagents, catalysts and solvents (~95% of USPTO atoms) don't participate. They're flagged once at parse time:

```python
spectator[i] = True  if atom i has no Δ across any pair
```

Spectators are **not masked** (zeroing them destroys global graph signal — the encoder needs to see them as context) but their loss contribution is downweighted by `spectator_weight=0.1`.

### 2.3 Batching

- PyG `Data` objects are padded to `(N_max, N_max)` Δ-matrices and `(N_max,)` masks at collate time.
- Disk-cached: `cache/{task_mode}_{hash16}.pt` keyed by SHA256 of the raw CSV. Cold parse ~10 min; warm load ~1 s.
- Targets shifted to non-negative class indices inside the engine (`+1` stepwise, `+3` end-to-end) so `cross_entropy` accepts them directly.

---

## 3. Architecture

```
SMILES → RDKit graph → ReactionTransformer ──▶ logits (B, N, N, C)
                       │
                       ├── 3× TransformerConv  (encoder)
                       └── DeltaMLP            (head)
```

### 3.1 Encoder — `ReactionTransformer` ([transformer.py](src/mech_uspto/models/transformer.py))

```
input:  x         (N_total, 25)   sparse node features
        edge_index, edge_attr     sparse edges + 6-dim features
        batch                     graph membership

  node_embedding:  Linear(25 → D)
  edge_embedding:  Linear(6  → D)

  for layer in 1..L:
      h = TransformerConv(D → D/heads, heads=8, edge_dim=D)(h, edge_index, edge_attr=e)
      h = F.dropout(h, p=0.1, training=self.training)
      h = LayerNorm(h + h_residual)

  h_dense, mask = to_dense_batch(h, batch)   # (B, N_max, D), (B, N_max)
output: dense node embeddings + valid-position mask
```

Default `hidden_dim D = 256`, `num_layers L = 6`, `num_heads = 8`. The model is ~3M parameters — bottleneck is dataset size (31k reactions), not capacity.

**Why `TransformerConv`?** Standard message-passing GNNs (GCN, GIN) underperform on bond-prediction because they're permutation-equivariant *within a hop* but cannot weight different neighbours differently based on the query atom. `TransformerConv` (Shi et al. 2021) adds an attention mechanism over neighbours, conditioned on edge features — exactly the right inductive bias when "is this bond about to break?" depends on *which* neighbour you're looking at.

### 3.2 Head — `DeltaMLP` ([heads.py](src/mech_uspto/models/heads.py))

```
input:  h_dense   (B, N, D)
        edge_dense optional (B, N, N, 6)   # densified input bond features

  h_src = h_dense[:, :, None, :].expand(B, N, N, D)   # query atom
  h_dst = h_dense[:, None, :, :].expand(B, N, N, D)   # key atom
  pair  = cat([h_src, h_dst, edge_dense?], dim=-1)    # (B, N, N, 2D + 6)

  logits = Linear(2D + 6 → D) → SiLU → Dropout → Linear(D → C)(pair)

  # Symmetrise: bond formation A-B == B-A
  logits = (logits + logits.transpose(1, 2)) / 2

output: (B, N, N, C) per-pair class logits
```

**Two non-trivial wirings:**

1. **Edge features in head** (`use_edge_features_in_head=True` by default). The head sees the *input* bond order for each pair directly, without relying on the encoder to preserve it through 6 layers of message-passing. Non-bonded pairs get an explicit zero vector — a strong signal against predicting Δ≠0. This single change eliminated ~500k spurious `Δ=−1` predictions on non-bonded pairs in baseline runs.

2. **Class-prior bias init.** The final linear layer's bias is set to `log(π) − mean(log(π))` where `π` is the empirical class prior (97.5% no-change, tiny tails). At step 0 the softmax already matches the data distribution; training only learns deviations. Standard RetinaNet trick (Lin et al. 2017) for severe class imbalance. Removes the early-training "predict reaction everywhere" oscillation entirely.

---

## 4. Loss — `MaskedFocalLossWithSpectators` ([focal.py](src/mech_uspto/losses/focal.py))

Per-pair focal cross-entropy with three masking layers:

$$
\mathcal{L} = -\frac{1}{Z} \sum_{(b,i,j)} m_{bij} \cdot w_{y_{bij}} \cdot s_{bij} \cdot (1 - p_{bij,y_{bij}})^{\gamma} \cdot \log p_{bij,y_{bij}}
$$

| Term | What | Why |
|---|---|---|
| $m_{bij}$ | Pair validity mask (both atoms present after padding) | Padding pairs are noise; never optimise on them |
| $w_{y}$ | Per-class weight (stepwise `[3, 1, 3]`, end-to-end `[1, 1.5, 3, 1, 3, 1.5, 1]`) | Compensate for 97.5% Δ=0 imbalance without triggering inverted collapse |
| $s_{bij}$ | Spectator weight (`0.1` if both atoms are spectators) | Spectators dominate the count; downweight without erasing context |
| $(1 - p)^\gamma$ | Focal modulation (`γ=3.5`) | Down-weight easy examples (high-confidence no-change pairs) so gradient focuses on rare-class learning |
| $Z$ | Sum of $m_{bij} \cdot s_{bij}$ | Normalise so loss magnitude is comparable across batch sizes / atom counts |

The focal exponent `γ=3.5` is unusually high — calibrated for this dataset's severe imbalance. Standard RetinaNet uses 2.0; 3.5 was needed to get gradient signal on Δ=±2, ±3 in end-to-end mode.

---

## 5. Training engine ([engine.py](src/mech_uspto/training/engine.py))

### 5.1 One epoch

```
for batch in train_loader:
    targets = batch.y_padded + shift                  # → non-negative class indices
    with autocast(bf16):                              # CUDA only; no GradScaler needed
        logits, mask = model(...)
        loss = criterion(logits, targets, mask_2d, spectator_mask)
    loss.backward()
    clip_grad_norm_(params, max=1.0)                  # numerical safety
    optimizer.step()
    apply_linear_warmup_to_lr()                       # over first 130 steps
    metrics = MetricsComputer.get_mechanism_metrics(logits, targets, mask)
    accumulate_running_tp_fp_fn(metrics)              # for first/25/last batch print
```

Same loop runs for validation with `is_train=False` (no backward, no optimizer step, `model.eval()` → dropout off, `with torch.no_grad()`-equivalent via no `.backward()`). Metrics are pooled across all val batches and reported at epoch end.

### 5.2 Optimization knobs

| Knob | Default | Rationale |
|---|---|---|
| Optimizer | AdamW | Standard for transformers; weight-decay on |
| `learning_rate` | `1e-4` | Conservative; warmup ramps from 0 |
| `weight_decay` | `1e-5` | Light, primarily to keep weights bounded |
| `warmup_steps` | 130 (injected by sbatch) | Linear ramp `0 → 1e-4` over first 130 optimizer steps. Prevents loss explosion on cold start |
| Scheduler | `ReduceLROnPlateau(patience=5, factor=0.5)` | Conservative; halves LR when val_loss stalls |
| `max_grad_norm` | `1.0` | Standard transformer practice |
| `dropout` | `0.1` (8 sites) | Heavy regularisation needed for 31k-sample dataset; see [FUTURE_TASKS #H3](FUTURE_TASKS.md) |
| Precision | bf16 autocast on CUDA | ~2× speed on A40/L40S; no GradScaler (bf16 has fp32 exponent range) |

### 5.3 Determinism

`seed_everything(seed=42, deterministic=True)` in [data/loaders.py](src/mech_uspto/data/loaders.py) seeds `PYTHONHASHSEED`, `random`, `numpy`, `torch` (CPU + all CUDA devices), and sets `cudnn.deterministic=True`, `benchmark=False`. DataLoaders get `worker_init_fn=_seed_worker` and an explicit `torch.Generator`. Cost: ~5–10% throughput. Required so any "did intervention X help?" comparison isn't just seed noise. `--no-deterministic` opts out.

### 5.4 Checkpointing

- **Selection criterion:** lowest `val_loss` (not F1). Loss is the actual training objective and is smoother than F1 across epochs.
- Saves model + optimizer + history + best metadata → resumable with `--resume <ckpt.pt>`.
- `best` checkpoint overwritten per improvement; `latest` saved every N epochs as a safety net.

---

## 6. Metrics ([metrics.py](src/mech_uspto/training/metrics.py))

All metrics operate over upper-triangle pairs `(i, j) with i < j` after applying the atom mask. "Reaction pair" = any pair with target class ≠ no-change.

| Metric | Aggregation | What it tells you |
|---|---|---|
| **Precision / Recall / F1** | From pooled `tp/fp/fn` across the epoch (not per-batch averaged) | Pair-level detection-and-classification accuracy on reaction pairs |
| **PR-AUC (pooled)** | One AP computed once over the concatenated `(score, target)` arrays from the whole epoch | Threshold-free quality of the "any reaction here?" score. Per-batch APs are non-additive — pooling is the correct aggregation |
| **PR-AUC per class** | Same pooling, one-vs-rest | Diagnose which classes the model can rank vs. not |
| **EM@1** | hits / samples-with-≥1-valid-pair | Whole-sample Δ-matrix top-1 exact-match. PMechDB-comparable proxy. See [FUTURE_TASKS #2](FUTURE_TASKS.md) for the planned product-level upgrade |
| **Per-class counters** | `n_preds_per_class`, `n_targets_per_class` | Class-collapse diagnostic: distinguishes "wrong" from "never even predicted this class" |
| **Confusion matrix** | CSV, written by `evaluate.py` only | Held-out test set only |

### 6.1 Train vs. val metric gap

Train metrics (printed per-batch as `P=… R=… F1=…`) consistently look much worse than val metrics for this model. Three compounding causes:

1. **Dropout in 8 sites.** `(1 − 0.1)^8 ≈ 0.43` end-to-end signal survival per train step. `model.eval()` for val turns all of it off.
2. **Class-weighted loss vs. unweighted metric.** Loss is shaped; metric is not.
3. **Intra-epoch averaging** (minor). Running `tp/fp/fn` pools predictions made by 86 different model states `θ₀ … θ₈₆`.

**The val metric is the correct one** — it reflects the model's deployable state. Train metric is "model + heavy stochastic noise" and isn't what you'd ever run at inference. Not a bug; documented in [FUTURE_TASKS #H3](FUTURE_TASKS.md).

---

## 7. Key design decisions and their evidence

| Decision | Code | Evidence / why |
|---|---|---|
| Spectator down-weighting (0.1) rather than masking | `MaskedFocalLossWithSpectators` | Masking destroys global graph signal; spectators are 95% of atoms and the encoder still needs to attend over them |
| Symmetric `(i,j)` / `(j,i)` predictions | `DeltaMLP.forward` | Bond formation is symmetric; halves effective output count and adds inductive bias |
| Reactants-only inputs (never see products) | `TrainingEngine._forward_and_loss` | Both modes are honest forward predictions; no information leak |
| `no_change_idx = num_classes // 2` (derived, not hardcoded) | `metrics.py`, `losses/focal.py` | A 7-class run with `no_change_idx=1` (the old hardcoded value) faked F1=0.976 on epoch 1 by treating 99.97% of pairs as reaction centers |
| PR-AUC computed once on pooled epoch buffers, not averaged per-batch | `pooled_pr_auc` in `metrics.py` | Per-batch APs are mathematically non-additive; averaging is wrong |
| `"any reaction" score = 1 − P(no_change)` | `_metrics_for_sample` | Earlier `probs[:, 0] + probs[:, -1]` was correct only for 3-class; silently dropped Δ=±1, ±2 in 7-class |
| Class-prior bias init in the head | `DeltaMLP._init_bias_from_prior` | Eliminates early-training oscillation between "no-change everywhere" and "reaction everywhere" |
| Edge features fed directly into the head | `DeltaMLP.forward` with `edge_dense` | Killed ~500k spurious Δ=−1 predictions on non-bonded pairs in baseline |
| Gentler class weights vs. earlier `[2, 4, 16, 1, 16, 4, 2]` | `Config.__post_init__` | Earlier weights caused inverted collapse (`RxnPreds > targets` by epoch 8) |
| Disk-cached datasets keyed by raw-CSV SHA256 | `create_dataloaders` | Cold parse 10 min → warm load 1 s; hash key auto-invalidates when CSV changes |
| Single dataset built once, split via `Subset` | `create_dataloaders` | Old code parsed 3× (train/val/test); 3× wall time waste |

---

## 8. Known limitations

1. **Single-shot inference only.** Stepwise training is honest, but stepwise *inference* (autoregressive rollout: predict Δ → apply to RDKit `Mol` → re-featurize → repeat) is not implemented yet. Blocked on a 4-part dependency chain in [FUTURE_TASKS #1](FUTURE_TASKS.md).
2. **EM@1 is matrix-exact, not product-exact.** SOTA mechanism-prediction papers (Bradshaw 2018, PMechDB) report whether the *predicted product molecule* matches the ground-truth product. Our EM@1 is stricter (matrix symmetries count as wrong) and a strict lower bound on product top-1 accuracy. See [FUTURE_TASKS #2](FUTURE_TASKS.md).
3. **No top-k for k > 1.** Requires Lawler-style beam enumeration over Δ-matrices. Folded into #2.
4. **Heavy dropout (8 sites).** Effective signal survival ~43% per train step. Currently load-bearing (val loss < train loss, no overfitting), but probably suboptimal at convergence. Ablation backlog as [#H3](FUTURE_TASKS.md).
5. **Class collapse on rare end-to-end classes.** Even with focal + class weights, the model effectively never predicts Δ ∈ {±2, ±3} (per-class counters: `c0=0/245`, `c5=0/211` after 14 epochs). Two-stage head ([#H2](FUTURE_TASKS.md)) is the planned remedy.
6. **No early stopping triggered yet.** `patience=20` is more than `num_epochs=15` in current sbatch — effectively disabled. Configured but inactive.
7. **Spectator detection is heuristic** (any atom with zero Δ across all pairs). Mis-labels true reaction atoms in rare cases where a bond breaks and reforms within the same reaction.

---

## 9. Pointers

- High-level overview, install, quickstart → [README.md](README.md)
- Dataset format and how to obtain it → [DATA.md](DATA.md)
- Cluster setup, SLURM workflow → [CLUSTER_SETUP.md](CLUSTER_SETUP.md)
- Open work and rationale → [FUTURE_TASKS.md](FUTURE_TASKS.md)
- Code entry points: [scripts/train.py](scripts/train.py), [scripts/evaluate.py](scripts/evaluate.py), [scripts/plot_history.py](scripts/plot_history.py), [scripts/sanity_check.py](scripts/sanity_check.py)
