# ADR-0003: `apply_delta` uses the row-sum charge heuristic by default

Status: **Accepted** (2026-06-02)

## Context
`apply_delta(mol, delta)` mutates bond orders to produce the next molecular state. After the bond surgery, RDKit's `SanitizeMol` will fail unless every atom has a valid valence — which depends on its formal charge. The function therefore must decide the post-surgery charge.

Three options were considered:

1. **Leave formal charges unchanged.** Most intermediates fail sanitize because, e.g., a newly-bonded N retains its old neutral charge while now having one more bond than its standard valence permits.
2. **Pass ground-truth charge deltas explicitly** (`charge_delta=...`). Requires the caller to know the answer, which defeats the inference-time use case.
3. **Row-sum heuristic.** Set `fc_new[i] = fc_old[i] − Σⱼ Δ[i,j]` (sum over off-diagonal). Implements the conservation rule "every electron pair gained as a bond came from an LP, so own-electron count drops by 1".

## Decision
The default behaviour of `apply_delta` is the **row-sum heuristic**. When ground truth is available (verifier scripts, cache builder), the caller passes `charge_delta=...` and `apply_charge_heuristic=False`.

## Rationale
- The heuristic is correct for the most common pattern (lone-pair attack onto an electrophilic centre): the nucleophile loses one own-electron, the electrophile gains one.
- It's exact for any reaction where Δq is zero on shared atoms (the vast majority — see `results/charge_diagnostic.json`: 99.96% of atoms have zero charge change R→P).
- Where it's wrong (1.2% of reactions, mostly SN2 / amine_oxidation), the cache builder catches the divergence and excludes the reaction. So the training cache only contains reactions where the heuristic would have agreed with ground truth.
- At inference time, the model only emits bond Δ; the heuristic gives a deterministic, well-defined charge assignment without needing a second prediction head. A future "diagonal-Δ head" extension is mentioned in `docs/rollout_design.md` §6.7.

## Consequences
- All training intermediates have charges consistent with the row-sum rule.
- At inference, a model that predicts the right bond Δ automatically gets the right charges (for the 99.75% subset).
- Sanitize-failure rescue chain (`_hypervalent_rescue`, `_h_overvalence_rescue`) handles a small number of edge cases where the heuristic alone isn't enough. These are documented inline in `src/mech_uspto/chemistry/apply_delta.py`.
- For diagnostics, callers should record whether they used the heuristic or ground-truth path (the cache builder records this via `skip_reason=used_fallback:step=N` in the manifest when the fallback fires).

## Validation
- `scripts/charge_diagnostic.py` computes per-atom heuristic accuracy: 33% on changed atoms, but 99.96% of atoms don't change, so overall accuracy is 99.95%.
- `scripts/verify_apply_delta_e2e.py` confirms end-to-end pass rate of 93–96% on the raw CSV.

## Related
- [ADR-0002: Clean training pool filter](0002-clean-training-pool-filter.md)
- `src/mech_uspto/chemistry/apply_delta.py` for the implementation
