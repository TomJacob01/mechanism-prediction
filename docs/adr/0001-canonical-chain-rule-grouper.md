# ADR-0001: Chain-rule is the canonical arrow grouper

Status: **Accepted** (2026-06-02)

## Context
The mech-USPTO-31k dataset gives a flat ordered list of curly arrows per reaction; the model needs them grouped into **elementary steps** (one Δ matrix per step). Two grouping strategies were prototyped:

1. **Chain-rule** (`group_arrows_into_steps`): two consecutive arrows belong to the same step iff the next arrow's source-atoms overlap with the previous arrow's target-atoms (electron-flow continuity).
2. **Validity-check** (`group_arrows_by_validity`): walk the arrows left to right; close the current step the moment the resulting intermediate has zero virtual transition states (VTS) per the ChRIMP electron-bookkeeping rule.

The validity grouper was added in pursuit of higher "textbook step-count match" (the fraction of reactions where the inferred step count equals a curated reference table in `scripts/_class_audit.py`). Multiple attempts (H-shuttle guards, same-H heuristic) plateaued at ~55% textbook match while introducing class-specific regressions (Cbz/acetal went from 100% to 0% with one heuristic). See session notes May–June 2026.

## Decision
**Chain-rule is canonical.** Validity grouper is scheduled for deletion.

Reasons:
- Chain-rule yields **99.75% clean-rollout acceptance** on the cache build (only 77 reactions excluded out of 31,364). The model trains on the rollout-valid pool, so this is the metric that matters.
- "Textbook step count" is a convention, not a physical truth. Over-splitting an SN2 into 2 steps still produces valid `state_k → state_{k+1}` training pairs with Δ ∈ {-1, 0, 1}; the model can learn the finer granularity.
- Greedy lookahead heuristics on flat arrows are empirically zero-sum: every class-specific fix breaks another class.
- One canonical grouper means one Δ semantics for the model, one set of tests to maintain, and no "which version did this checkpoint train against?" ambiguity.

## Consequences
- `group_arrows_by_validity`, `parse_steps_by_validity`, `valence.py` (only consumed by the validity grouper), and the related `chemistry/__init__` re-exports become dead code and will be removed (cleanup batch B).
- `scripts/_grouper_compare.py` and `scripts/slurm/grouper_compare.sbatch` also become dead.
- If future work needs intermediate-validity checks for a different purpose (e.g. inference-time pruning of candidate Δ matrices), the VTS counter can be resurrected from git history — but a new ADR must justify it.

## Validation
- `scripts/verify_cache_rollout.py` confirms the cached steps round-trip from reactants to products.
- `scripts/cache_audit.py` confirms Δ values stay in {-1, 0, 1} for the stepwise cache.

## Related
- [ADR-0002: Clean training pool filter](0002-clean-training-pool-filter.md)
- [ADR-0004: Parquet canonical cache](0004-parquet-canonical-cache.md)
