# ADR-0004: Parquet is the canonical training cache

Status: **Accepted** (2026-06-02)

## Context
The earliest version of the training loop parsed the CSV → arrows → steps → tensors **at every dataloader call**, which made each epoch start slow and put RDKit on the hot path of every minibatch. Three storage options were considered:

1. **Re-parse from CSV every epoch** (status-quo at the time).
2. **Pickle the parsed Python objects.**
3. **Parquet with a fixed schema** keyed by atom-map number.

## Decision
Persist the parsed and clean-rolled-out training pool as **two Parquet files** at `data/cache/parquet/`:

- `reactions.parquet` — one row per clean reaction: `rxn_id`, `n_steps`, `n_atoms_mapped`, `mechanistic_class`, `mechanistic_label`, `data_source`, `reactant_mol` (RDKit binary), `product_mol`, `split_hash`.
- `steps.parquet` — one row per elementary step: `rxn_id`, `step_idx`, `mol_pre`, `mol_post`, `bond_changes` (list of `{map_i, map_j, delta}`), `charge_changes` (list of `{map_i, delta}`), `arrow_count`. (`used_fallback` was removed in cleanup batch C; existing caches built before then still carry it and remain readable.)

Plus a `cache_manifest.json` with build metadata (input CSV fingerprint, git HEAD if available, counts, build time).

Schema is defined in [src/mech_uspto/data/cache_schema.py](../../src/mech_uspto/data/cache_schema.py).

## Rationale
- **Speed.** RDKit `Mol.ToBinary()` is ~3× faster to deserialize than `MolFromSmiles` and ~2× smaller on disk. Parquet's columnar layout means the dataloader can `to_pylist()` a single column without touching the others.
- **Schema portability.** Atom maps are stable across re-alignment / `AddHs` choices; the downstream featurizer resolves maps to RDKit indices at load time. No cache rebuild needed when the featurizer changes.
- **Reproducibility.** The manifest records exactly which CSV (size + mtime) and which Git commit produced the cache, so an experiment's training data is recoverable from one JSON file.
- **Audit-friendly.** The whole cache is queryable in one line: `pyarrow.parquet.read_table(...).to_pylist()`. No special tooling.
- **Smaller than pickle.** Zstd compression saves ~3× on disk vs. raw pickle, and pickle locks readers to a specific Python version.

## Consequences
- All training code and analysis scripts read from the parquet cache, not from the CSV directly.
- Adding a new featurization scheme is cheap (no rebuild); adding a new arrow-grouping rule, charge convention, or apply_delta change requires rebuilding the cache.
- The cache is **not** committed to Git — it's regenerated from the CSV. Build time on 64 workers: ~85 seconds.
- Per-row mol binaries inflate disk size (~250 MB on disk for 31k reactions). Acceptable for our scale; would need revisiting at 1M+.

## Validation
- `scripts/verify_cache_rollout.py` re-runs sequential apply_delta over `steps.parquet` and confirms the final intermediate matches `product_mol`.
- `scripts/cache_audit.py` reports Δ ranges, split balance, class distribution, n_steps/n_atoms histograms.
- Round-trip test in `tests/test_parquet_dataset.py` confirms `ParquetMechDataset` returns sane `Data` objects.

## Related
- [ADR-0002: Clean training pool filter](0002-clean-training-pool-filter.md) — what gets into the cache.
- [ADR-0003: Heuristic charges](0003-heuristic-charges-in-apply-delta.md) — how charges are stored.
