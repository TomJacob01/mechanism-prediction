# Future tasks

Backlog **not** in the current refactor. Legend: 🔴 blocker · 🟡 important · 🟢 nice-to-have · ✅ done.

---

## Open — ordered by priority

| #   | Pri | Title                                                  | Effort  | Touches                                                                                |
|-----|-----|--------------------------------------------------------|---------|----------------------------------------------------------------------------------------|
| 1   | 🔴  | Autoregressive rollout (stepwise inference)            | ~1 wk   | new `inference/rollout.py`, `training/metrics.py`                                      |
| 1.1 | 🔴  | └─ Define deliverable (final-Δ vs path vs product)     | ~30 m   | none — decision doc                                                                    |
| 1.2 | 🔴  | └─ 10-mechanism RDKit-apply + sanitize spike           | ~2 h    | new `scripts/rollout_spike.py`                                                         |
| 1.3 | 🔴  | └─ Dataset audit: intermediates? length distribution?  | ~1 h    | `scripts/sanity_check.py` extension                                                    |
| 1.4 | 🔴  | └─ Exposure-bias check: one-step rollout AUC drop      | ~3 h    | new `scripts/exposure_bias_check.py`                                                   |
| 1.5 | 🔴  | └─ Implementation (gated on 1.1–1.4)                   | TBD     | new `inference/rollout.py`, `training/metrics.py`                                      |
| 2   | 🟡  | Product-level top-k accuracy (SOTA-comparable)         | ~1 wk   | new `inference/apply_delta.py`, `training/metrics.py`, `scripts/evaluate.py`           |
| H1  | 🟢  | Biaffine head                                          | ~3 h    | `models/heads.py`, `models/transformer.py`, config                                     |
| H2  | 🟢  | Two-stage head (detection + classification)            | ~half d | `models/heads.py`, `losses/focal.py`, `engine.py`                                      |
| H3  | 🟢  | Dropout ablation (rate × placement)                    | ~2 h    | `models/transformer.py`, `models/heads.py`, sweep config                               |
| H4  | 🟢  | Atom-pair priors in head (graph-distance, etc.)        | ~half d | `data/featurization.py`, `models/heads.py`, `models/transformer.py`                    |
| 3   | 🟢  | Hyperparameter sweep config                            | ~2 h    | new `sweeps/*.yaml` (after H1/H2 settled)                                              |

---

## Notes

### #1 Autoregressive rollout

The single hardest task in the backlog — much harder than the encoder. Hardness is concentrated in chemistry (applying predicted Δ to RDKit `Mol` without sanitization failures) and metrics (what counts as "correct" when mechanisms have multiple valid orderings), *not* in the model code itself. Subtasks 1.1–1.4 must complete before 1.5 — they determine ~80% of the design.

- **1.1 Deliverable.** Three candidate questions rollout could answer:
  - **(a) Final-Δ comparison.** Roll out stepwise to convergence; compare cumulative Δ to end-to-end's direct prediction. Only the final pair-wise Δ matters. Matches the README's stepwise-vs-end-to-end FPR claim. *Sidesteps almost every hard chemistry question.* **Default recommendation.**
  - **(b) Path accuracy.** Does predicted step *k* match true step *k*? Mechanistically richer but brutally strict — mechanisms have multiple valid orderings.
  - **(c) Product validity.** Does the final molecule equal the true product? Most permissive; loses mechanistic insight.
- **1.2 Chemistry spike.** Take 10 mechanisms; manually walk through step-by-step applying *ground-truth* Δ to RDKit (`RWMol` edits → re-perceive valence → `SanitizeMol`). Count how often sanitization fails and on what kinds of steps. Tells you whether rollout is "engineering tedium" or "open research problem" for this dataset. **Prerequisite for #2.**
- **1.3 Dataset audit.** Confirm whether `mech-USPTO-31k.csv` contains per-step intermediates (separate rows or list-valued columns) vs only start + end. Get the mechanism-length distribution (median, 95th, 99th percentile) — sets `max_steps` for 1.5 and tells us if path-accuracy from 1.1(b) is even feasible.
- **1.4 Exposure-bias check.** Take the trained stepwise model, run one rollout step on the test set, re-featurize the predicted result, measure PR-AUC on step 2. If it drops a lot, the model needs scheduled sampling / DAgger / teacher-forcing during training — meaning rollout is **not** a post-hoc inference component, it's a training-loop change. Big scope difference.
- **1.5 Implementation.** Once 1.1–1.4 land: write `predict_full_reaction(model, smiles)` loop (predict Δ → apply to `Mol` → re-featurize → next step, until `argmax(logits)==0` everywhere or `max_steps`). Search strategy (greedy vs beam) depends on 1.1's choice. Other open design points: multi-pair-per-step actions (apply atomically vs one-at-a-time), termination (argmax vs probability threshold), sanitization failure policy (skip / backtrack / force-allow), evaluation cadence (every-epoch vs end-only).

### #2 Product-level top-k accuracy

What mechanism-prediction papers actually headline (Bradshaw 2018 ELECTRO, PMechDB / Tavakoli–Baldi 2024): does any of the top-k *predicted mechanisms*, when applied to the reactants, yield the **correct product molecule** (canonical SMILES match)? This is the right comparison metric — invariant to Δ-matrix symmetries and direct apples-to-apples with the literature.

Pipeline (per val/test sample):

1. **Top-k Δ-matrix enumeration.** Argmax → rank 1. For k > 1: Lawler-style lazy heap over single-pair-swap candidates ordered by Σ log-prob cost.
2. **Apply Δ to reactants.** New helper `apply_delta(rdkit_mol, delta_matrix) → rdkit_mol_product`: walk the upper-tri non-zero entries, edit bond orders / formal charges on an `RWMol`, then `SanitizeMol`. **Reuses the chemistry spike from #1.2** — same `apply Δ to RDKit Mol` primitive, same sanitization-failure handling.
3. **Canonicalise + compare.** `Chem.MolToSmiles(prod, canonical=True)` vs ground-truth product SMILES. Hit if equal.
4. **Aggregate.** `product_top_k = (1/N) Σ 1[hit within top-k]` for k ∈ {1, 5, 10}.

**Cost.** RDKit roundtrip ~1–10 ms/sample; only on val/test, not train. With ~5k val reactions and k=10, adds < 1 min/epoch — negligible.

**Gating.** #1.2 is a prerequisite. If sanitization fails often, the metric is unreliable and #2 blocks on a fix to #1.2's findings.

**Relationship to EM@1.** Once #2 lands, EM@1 (already implemented; see Done §Metrics) becomes a useful internal diagnostic ("how often does the model recover the exact ground-truth matrix?") and `product_top_1` becomes the headline SOTA-comparable number. Keep both.

### #H1 Biaffine head

Replace the concat-MLP's first linear with a per-class bilinear form:

```python
# h: (B, N, D), B_c: (C, D, D), edge_proj: nn.Linear(edge_in, C)
pair  = torch.einsum("bnd,cde,bme->bnmc", h, B_c, h)
logits = pair + edge_proj(edge_dense)              # (B, N, N, C)
logits = (logits + logits.transpose(1, 2)) / 2     # symmetrise
```

Each class `c` gets its own `D × D` interaction matrix `B_c` plus an edge-feature bias. Standard biaffine parser head (Dozat & Manning 2017) — the right inductive bias for pairwise scoring. With `D=256, C=7`: 459k bilinear params vs the current head's 133k (~3.5×; negligible vs ~3M encoder). Forward FLOPs comparable to current when `H ≈ C·D`.

**Gate.** `Config.head_type ∈ {"mlp", "biaffine"}`, default `"mlp"`. New class `BiaffineDeltaHead` in [heads.py](src/mech_uspto/models/heads.py); [transformer.py](src/mech_uspto/models/transformer.py) picks the head class.

**Test.** Shape + symmetry (mirror existing `DeltaMLP` tests), plus: `B_c = I, edge_proj = 0` ⇒ output recovers `h_i · h_j` per class.

### #H2 Two-stage head (detection + classification)

Decouple the imbalanced binary detection from the rare-class multiclass:

1. **Detector.** Sigmoid head, `P(Δ ≠ 0 | i, j)`. Trained with focal-BCE on the binarised target.
2. **Classifier.** 6-way softmax (2-way stepwise) on the non-zero classes only, `P(Δ = k | Δ ≠ 0, i, j)`. **Teacher-forced**: trained on pairs where the *true* label is non-zero (no detector gating during training).

Inference: `P(Δ=0) = 1 − P(Δ≠0)`; `P(Δ=k≠0) = P(Δ≠0) · P(Δ=k | Δ≠0)`.

**Why.** With 97.5% no-change pairs, the head is dominated by the imbalance — the classifier never gets a clean signal. Decoupling lets the detector specialise (focal-BCE + bias-init to 2.5% positive rate, exact RetinaNet) while the classifier sees a class-balanced positive-only subset.

**Cost.** ~2× head parameters; same inference FLOPs (multiply outputs); two losses with separate masks in `engine.py`.

**Gate.** `Config.head_type = "two_stage"`. New `TwoStageDeltaHead` in `heads.py`, new `TwoStageLoss` in `losses/focal.py`. Metrics need to convert `(P_detect, P_class)` → per-class distribution before the existing PR-AUC / F1 pipeline.

**When to try.** Only after #H0 (bias-init + edge features, **done**) has plateaued and the bottleneck is clearly *classifier confusion among rare classes*, not detection failure. If the model is still producing only Δ∈{−1,0,+1} after a converged run with #H0, this is the next move.

**Test.** Round-trip: build head + targets with Δ=2 everywhere on one pair; after a few hundred gradient steps verify `P(detect)→1` and `P(Δ=2 | detect)→1`, product → 1 on that pair, low elsewhere.

### #H3 Dropout ablation (rate × placement)

Current setup applies dropout in **8 places**: 3× `TransformerConv(dropout=0.1)` attention dropout, 3× post-conv `F.dropout(h, p=0.1)` on features, 2× `nn.Dropout(0.1)` in `DeltaMLP`. Crudely independent, that's `(0.9)^8 ≈ 0.43` end-to-end signal survival per train step — i.e. ~57% noise, far more aggressive than a nominal "10% dropout" suggests.

Observed consequence: train metrics are ~half of val metrics across the entire run (ep 10: train F1 0.39 vs val F1 0.68, train P 0.43 vs val P 0.87). Same `MetricsComputer` call — difference is purely `model.train()` (dropout on, 8 sites) vs `model.eval()` (off). Confirmed not a bug; reflects that train predictions live under heavy stochastic noise that never exists at inference.

**Question.** Is the current dropout load-bearing (val F1 drops if reduced) or excessive (val F1 rises if reduced — we're under-using model capacity)?

**Cheapest study.** Single sweep on the long-config run:
- `dropout ∈ {0.0, 0.05, 0.1, 0.2}` (global rate)
- `drop_post_conv ∈ {on, off}` (toggle sites 4–6 — the `F.dropout` after each `TransformerConv`; sites 1–3 inside `TransformerConv` and 7–8 in the head stay)

4 × 2 = 8 runs. Primary signal: best val F1 / EM@1 / PR-AUC per cell. Secondary: the train-vs-val gap closes monotonically as dropout shrinks (sanity check that the mechanism is what we think it is).

**Currently working without it.** Don't run standalone — fold into #3's HP sweep. Val loss is currently *below* train loss and val F1 is still rising at ep 10, so dropout is doing useful regularization for now. Only revisit if val F1 plateaus.

### #H4 Atom-pair priors in head

Motivated by the per-class PR-AUC asymmetry observed in run 68210060: c2 (bond breaking) reaches PR-AUC ≈ 0.98 while c4 (bond formation) sits at 0.14 (≈15× prevalence-baseline lift, vs. ≈65× for c2). The two classes share an architecture but not an input-feature regime: `featurize_edges` only emits non-zero edge features for **existing** RDKit bonds, so the head's `edge_dense[B, N, N, 6]` slot for any non-bonded pair `(i, j)` is all zeros. c2 gets a full bond-type one-hot at the head; c4 gets nothing and has to discover "these atoms want to bond" through node embeddings alone. Deeper encoders (more `TransformerConv` layers) is the brute-force lever — same input, more rounds of message passing — and saturates fast (over-smoothing past 6–8 layers). Atom-pair priors is the structural lever: explicit pair features for *all* `(i, j)`, including non-bonded.

Three tiers, each subsuming the prior:

1. **Graph shortest-path distance (cheapest, recommended first attempt).** For each `(i, j)`, BFS hops along the reactant-bond graph, clipped at e.g. 6, encoded one-hot to `(N, N, 7)`. Pure topology, no 3D, no chemistry knowledge. Compute once at dataset-build time via `Chem.GetDistanceMatrix(mol)` (RDKit, fast) and cache alongside the existing graph data. Concatenate into `edge_dense` before passing to `DeltaMLP`. Expected gain on c4: 2–3× PR-AUC lift (rough estimate based on similar augmentations in pair-scoring literature). Does not help c2 (already feature-rich). Single-feature ablation; easy to reverse.
2. **Atom-pair chemical priors.** Per-pair scalars derivable from per-atom RDKit properties: electronegativity difference, formal-charge sum, lone-pair-count product (proxy for nucleophile×electrophile), HOMO/LUMO proxies (e.g. "is i sp³ with lone pair, j sp² with empty p?"), shared-ring indicator. ≈10 features. Same plumbing as #1.
3. **3D Euclidean distance.** Embed each reactant with `EmbedMolecule + MMFFOptimizeMolecule`, store coordinates, compute pairwise distance at dataset-build time. Single scalar per pair. Adds ~100–500ms per molecule once (cached), and exposes the model to actual geometry. Most informative; most cost; hardest to make robust (some molecules fail to embed).

**Plumbing.** All three flow through the same hook: extend `ReactionTransformer.forward` to densify a per-batch `pair_features` tensor `(B, N, N, k)` and concatenate it onto `edge_dense` before the `DeltaMLP` call. `DeltaMLP.__init__` already takes `edge_dim` as a constructor arg — just bump it by `k`. No new module needed for #1; #2 and #3 want a `data/pair_features.py` module to keep the per-pair logic out of `featurization.py`.

**Gate.** `Config.pair_features ` accepts a list of strings, e.g. `["graph_distance"]`, `["graph_distance", "chemical_prior"]`, `["graph_distance", "chemical_prior", "euclidean"]`. Default `[]` preserves current behaviour.

**When to try.** Only after the rollout work (#1) is unblocked. The pair-prior change is purely additive and orthogonal to the rollout pipeline, so it can run in parallel as a separate experiment, but the headline metric is product-top-k from #2, and there's no point tuning encoder features until rollout is producing a usable end-to-end signal.

**Test.** `pair_features=["graph_distance"]` on a known 4-atom fixture (linear chain) produces the expected distance-matrix one-hot. Forward pass shape unchanged at the head output `(B, N, N, C)`. Concat dimension matches `edge_in + 7` in the head's first linear.

### #3 HP sweep

Defer until H1/H2 are settled and there's one config hitting F1 ≥ 0.5 — sweeping a broken model just finds the least-broken broken config. When the time comes: W&B Sweeps or Optuna, search `lr × weight_decay × warmup_steps × focal_gamma` around the best architectural config.

---

## Completed work

Design decisions and the rationale behind them live in [MODEL.md §7](MODEL.md#7-key-design-decisions-and-their-evidence). Bug-fix history is in `git log`.
