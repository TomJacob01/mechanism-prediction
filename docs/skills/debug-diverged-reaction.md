# Skill: Debug a diverged reaction

Use when `cache_manifest.json` reports a higher-than-expected `diverged` count, or when `verify_cache_rollout.json` reports `mismatch` for specific reactions.

## Step 1 — Get the list of failures
```powershell
# From the cache manifest
Get-Content data/cache/parquet/cache_manifest.json | ConvertFrom-Json | Select-Object -ExpandProperty skip_reason_breakdown

# Per-reaction breakdown (uses build_parquet_cache's worker output)
.\.venv\Scripts\python.exe scripts/research/bucket_diverged.py
```
Or from a cache-rollout report:
```powershell
Get-Content results/cache_rollout.json | ConvertFrom-Json | Select-Object -ExpandProperty failure_examples | Format-Table
```

## Step 2 — Reproduce a single divergence
Pick one `rxn_id` from the report, then:
```powershell
.\.venv\Scripts\python.exe -c "
from mech_uspto.data.parser import MechUSPTOParser
from mech_uspto.data.arrow_parser import parse_arrows, parse_steps
rxn = next(r for r in MechUSPTOParser.parse_csv_file('data/mech-USPTO-31k.csv') if r.reaction_id == '<RXN_ID>')
print('class:', rxn.mechanistic_class)
print('label:', rxn.mechanistic_label)
print('R:', rxn.overall_reactants_smi)
print('P:', rxn.overall_products_smi)
"
```

## Step 3 — Trace the rollout
Walk the arrows step by step. The key diagnostic: at which step does the intermediate first diverge?
```powershell
.\.venv\Scripts\python.exe scripts/verify_apply_delta_sequential.py --limit 1 --offset <row_idx> --num-chunks 1
```
Or use the e2e verifier with `--limit 1` and inspect the JSON failure detail.

## Step 4 — Common causes and fixes

| Symptom in failure_examples | Likely cause | Where to look |
|---|---|---|
| `apply_delta:invalid_order` | Δ asks for a bond order outside {0,1,2,3} | `apply_delta.py` (re-check the Δ-from-arrows logic in `arrow_parser.py:arrow_bond_changes`) |
| `apply_delta:sanitize_failed` after rescue | Hypervalent atom that the rescue chain can't fix | `apply_delta.py:_hypervalent_rescue`; add the element/pattern to the rescue table if generally safe, else exclude the reaction class |
| `mismatch` with reasonable-looking intermediate | Stereo descriptor only — see [`scripts/verify_apply_delta_e2e.py`](../../scripts/verify_apply_delta_e2e.py)'s `_canon_strip_unrecoverable` | Confirm the failure is *truly* a chemistry difference, not a stereo annotation gap in the dataset |
| `missing_map` | An arrow references an atom-map not present in `map_to_idx` | Usually a leaving-group atom that should have map ≥ 101 but was assigned <100, or vice versa |
| All steps OK but final canonical SMILES differs | Implicit-H bookkeeping diverged silently | Run `scripts/charge_diagnostic.py` on the single reaction; check whether the row-sum heuristic agrees with ground truth at every step |

## Step 5 — Decide
- If the failure is a **systematic class issue** (>10 reactions of the same class), it's worth fixing in `arrow_parser.py` or `apply_delta.py` and rebuilding the cache (see [rebuild-cache.md](rebuild-cache.md)).
- If it's **idiosyncratic** (<5 reactions across the dataset), leaving it in the excluded pool is correct — these are dataset annotation gaps, not algorithm bugs.
- Update `docs/adr/` if the fix changes a top-level convention; otherwise just commit + push.

## Pitfalls
- **Don't tune to a single failing reaction.** Any heuristic added to make one Cbz reaction succeed has historically broken three others (see ADR-0001 context). Confirm fixes hold across at least 20 random failures in the same class before merging.
- **Stereo mismatches are usually dataset, not us.** The patent's product SMILES often records chirality on atoms whose reactant SMILES has no stereo annotation. apply_delta can't conjure stereo that isn't in the input. See `pass_rate_recoverable` in the e2e verifier output.
