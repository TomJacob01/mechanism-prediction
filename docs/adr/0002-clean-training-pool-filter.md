# ADR-0002: Train only on the clean-rollout subset

Status: **Accepted** (2026-06-02)

## Context
For each reaction in mech-USPTO-31k, the cache builder (`scripts/build_parquet_cache.py`) runs a full rollout:

1. Parse arrows → group into elementary steps.
2. For each step, apply the Δ to the running molecule using **ground-truth** charge deltas (computed from the recorded product).
3. After all steps, check that the final molecule's canonical SMILES (stereo-blind, leaving-groups stripped) matches the recorded product.

A reaction is **excluded** from the training cache iff *any* of:
- An intermediate step requires the charge-heuristic fallback (rejected with `skip_reason=used_fallback:step=N`).
- An intermediate is "suspect" (canonical-SMILES round-trip isn't a fixed point — aromaticity / tautomer perception drift).
- The final rolled-out product doesn't match the recorded product (`diverged`).

## Decision
**Use only the clean-rollout subset for training.** Currently this is 31,287 of 31,364 reactions (99.75%), with 46 diverged + 31 fallback excluded.

## Rationale
- A reaction whose rollout doesn't reach the recorded product means our parser/grouper/apply_delta chain disagrees with the dataset. Training on those would teach the model wrong intermediates.
- A reaction that requires the heuristic charge fallback means the per-arrow charge deltas don't sum to the observed R→P charge delta — i.e. the curated arrows are charge-inconsistent. The intermediates we'd cache would have arbitrary (heuristic) charges, not ground truth.
- A suspect intermediate carries RDKit perception drift; using it as a training target makes loss values misleading.
- Loss from excluding 0.25% of the corpus is negligible vs. data-quality gain.

## Consequences
- The cache is **smaller** than the raw dataset — every loader and analysis script must read from `data/cache/parquet/`, not from the CSV directly.
- The `used_fallback` schema column was removed in cleanup batch C (2026-06): it was always `False` by construction (any rxn that needed the fallback is rejected upstream). The signal lives instead in `cache_manifest.json:skip_reason_breakdown`.
- Per-class distributions in the cache are very close but not identical to the raw CSV. `scripts/cache_audit.py` reports the divergence.
- If we ever need to evaluate on the *excluded* reactions, run inference from the raw CSV with the same pipeline.

## Validation
- Manifest at `data/cache/parquet/cache_manifest.json` records the keep rate and skip reasons.
- `scripts/verify_cache_rollout.py` re-runs the rollout sanity at any time.
- `scripts/research/bucket_diverged.py` lets us drill into the excluded reactions.

## Related
- [ADR-0001](0001-canonical-chain-rule-grouper.md), [ADR-0003](0003-heuristic-charges-in-apply-delta.md), [ADR-0004](0004-parquet-canonical-cache.md)
