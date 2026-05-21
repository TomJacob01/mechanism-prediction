# Rollout — design

Single source of truth for `FUTURE_TASKS.md` #1 (autoregressive rollout).

🟢 decided · 🟡 default, awaiting review · 🔵 deferred to Phase 2.

---

## 1. Goal

Ship `predict_full_reaction(model, smiles)` that rolls a trained
end-to-end model into a stepwise mechanism, plus two val metrics:

- **`cumulative_delta_match`** (iter 1) — rolled-out cumulative Δ vs
  ground-truth end-to-end Δ.
- **`product_top_k_accuracy`** (iter 2) — canonical SMILES match,
  k ∈ {1, 5, 10}. SOTA-comparable.

Baseline: run **68210060** (F1=0.731, PR-AUC=0.745, c4 PR-AUC=0.14).

## 2. Architecture in one line

MPNN-class encoder (`TransformerConv` ×6) + dense pair head, trained
supervised, deployed as an MDP rollout (state = `Mol`, action = Δ-step,
transition = `apply_delta`). Teacher-forced training is the default;
scheduled sampling is gated on 1.4.

---

## 3. Phase 0 — done

- ✅ Latent shift-arrow bug fixed in `transformations.py:103-122`.
- ✅ `tests/test_transformations.py` — 7 regression tests.
- ✅ `FUTURE_TASKS.md` H4 (atom-pair priors backlog entry).
- ✅ Full `pytest -q` green (90 tests).

## 4. Phase 1 — four de-risking experiments

| Sub | Output | Decision gate |
|---|---|---|
| **1.1** | This doc, frozen | Decisions in §6 reviewed |
| **1.2** `scripts/rollout_spike.py` | `results/rollout_spike.json` per-op sanitize pass rate on 10 reactions | ≥0.90 → true stepwise data; 0.60-0.90 → end-to-end + scheduled sampling; <0.60 → descope |
| **1.3** `scripts/sanity_check.py --audit-mechanisms` | Arrow-count quantiles | Sets `max_steps` (99th pct + 2) |
| **1.4** `scripts/exposure_bias_check.py` | `pr_auc_step1` vs `pr_auc_step2` | ratio ≥0.80 → no training-loop change; <0.80 → 1.5d in scope |

1.4 needs the 68210060 checkpoint accessible (sync down or run on cluster).

## 5. Phase 2 — implementation (gated on Phase 1)

| Sub | Task | Conditional |
|---|---|---|
| 1.5a | Rebuild `_build_stepwise_dataset` from arrows; retrain | 1.2 ≥ 0.90 |
| 1.5b | `inference/rollout.py` — `predict_full_reaction` | always |
| 1.5c | `metrics.py` + `engine.evaluate()` extension | always |
| 1.5d | `--scheduled-sampling-prob` in `scripts/train.py` | 1.4 < 0.80 |

---

## 6. Decisions

| # | Item | Default |
|---|---|---|
| 6.1 🟡 | Atom-map persistence | `apply_delta` re-keys by `GetAtomMapNum()` post-sanitize; caller maintains `map_to_idx` between steps |
| 6.2 🟡 | Hydrogen handling | Keep `add_hs=True`; `apply_delta` does explicit H bookkeeping (don't retrain without Hs) |
| 6.3 🟢 | Bond-order semantics | `b_new = b_current + delta[i,j]`; remove if 0, add if was 0, change otherwise; reject `<0` or `>3` |
| 6.4 🟡 | Formal charges | Heuristic: per-atom Δ-row-sum → `-Σⱼ delta[i,j]` charge change; revisit if 1.2 finds radical-mechanism failures |
| 6.5 🟡 | Multi-pair per step | **Single-pair** (highest-confidence non-zero), recompute logits each step. Atomic-step is iter-2 ablation |
| 6.6 🟢 | Termination | `argmax==0` everywhere OR `step >= max_steps` |
| 6.7 🟢 | Sanitize-fail policy | `skip_step` (default), `terminate`, `force_allow` |
| 6.8 🟢 | Determinism | `eval()` + autocast off + deterministic algos at inference |
| 6.9 🔵 | Featurization cache | Defer to 1.5d |
| 6.10 🔵 | Spectator mask in rollout | Defer to 1.5d |

---

## 7. Pitfalls

- **Atom-map renumbering by RDKit sanitize.** Test `test_apply_delta_preserves_atom_maps`.
- **Featurization mismatch train ↔ rollout.** Same `process_mapped_smiles` codepath. Test enforces 25-d node / 6-d edge / `add_hs=True`.
- **Cumulative-Δ off-by-one or sign error.** Test on 1-step proton-transfer fixture: `cumulative == ground_truth_e2e_delta`.
- **Class-shift inconsistency** between training (`y + 3`) and metrics. One shared helper `_shift_targets`.
- **bf16 / fp32 argmax disagreement.** Force fp32 + deterministic in rollout.
- **`max_steps` too low.** Set from 1.3 audit + 2 buffer; log `status="max_steps"` rate.
- **`skip_step` masking real bugs.** Dump per-failure histogram (op type, RDKit error) every eval.
- **Stale featurization cache** after 1.5a. Add `arrow_apply_version` to cache key.
- **Atom-map collisions** post-`apply_delta`. Defensive assert: maps unique and non-zero.

---

## 8. Out of scope (track in `FUTURE_TASKS.md`)

Beam search (iter 2 only), DAgger, reverse-mode teacher forcing, RL
fine-tuning, atom-pair priors (#H4), biaffine head (#H1), two-stage head
(#H2), 3D coordinate features.

## 9. Open questions

1. Is the 68210060 checkpoint accessible locally, or cluster-only?
2. Iter-2 (`product_top_k`) bundled with iter-1 in Phase 2, or sequential?
3. Pushback on any 🟡 default in §6?
