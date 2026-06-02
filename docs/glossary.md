# Glossary

Domain vocabulary used throughout this repo. When a term appears in code, prefer the canonical spelling here.

## Chemistry

| Term | Definition |
|---|---|
| **Reactant / R** | Left-hand side of a reaction SMILES. Atom-mapped. |
| **Product / P** | Right-hand side. Same atom maps as the corresponding reactant atoms. |
| **Atom map number** | Integer label on an atom (`[CH3:7]`) that survives a reaction. Maps connect R atoms to their P counterparts. |
| **Leaving group** | Atoms present in R but not P. Identified by atom-map ≥ 101 (MechFinder convention) or by absence from `product_mol`. |
| **Arrow** | One electron-pushing arrow in the curly-arrow mechanism. Encoded as `(src, tgt)` where each side is an `int` (lone pair on atom N), a `[a, b]` list (bond between atoms a, b), or a `float N.k` (kth implicit H of atom N). |
| **Mechanistic class** | Top-level reaction category (e.g. `SN2`, `DCC_condensation`, `Boc_deprotection`). 30+ classes in mech-USPTO-31k. |
| **Mechanistic label** | The flat ordered list of arrows for one reaction, stored as a Python literal in the CSV's `mechanistic_label` column. |
| **Elementary step** | A group of arrows that fire concertedly (e.g. SN2 = 2 arrows in one step, E2 = 3 arrows). Recovered by the grouper from the flat arrow list. |
| **Intermediate** | A molecule between two elementary steps. Must be valence-valid (passes RDKit sanitize). |
| **VTS (virtual transition state)** | An over-valent half-arrow state. Counted by `count_virtual_ts`. A real stable intermediate has 0 VTS atoms. |
| **Spectator** | An atom or fragment that doesn't change between R and P. Identified per-atom by `SpectatorDetector` for loss masking. |

## Bond / charge changes

| Term | Definition |
|---|---|
| **Δ (delta) / bond-order delta** | Per-bond change in bond order between two states. Symmetric `(N, N)` integer tensor where `Δ[i,j]` is the bond-order change between atoms `i` and `j`. |
| **Stepwise Δ** | Δ between two consecutive intermediates. Entries in `{-1, 0, 1}`. |
| **End-to-end Δ** | Δ between R and P directly, summing over all steps. Entries in `{-2, -1, 0, 1, 2}` (rarely ±3). |
| **Δfc / charge delta** | Per-atom change in formal charge. `(N,)` integer tensor. |
| **Charge heuristic** | `apply_delta`'s default: sets `fc_new[i] = fc_old[i] − Σⱼ Δ[i,j]` (row-sum rule). Used when ground-truth Δfc isn't available. See [ADR-0003](adr/0003-heuristic-charges-in-apply-delta.md). |

## Pipeline / data

| Term | Definition |
|---|---|
| **Rollout** | Sequentially applying each elementary step's Δ to a starting molecule. Used by the cache builder to validate end-to-end consistency. |
| **Clean rollout** | A rollout that uses no charge-heuristic fallback, no suspect intermediates, and matches the recorded product (canonical SMILES, stereo-blind). |
| **Fallback** | When apply_delta with strict ground-truth charges fails, retrying with the heuristic. A reaction with any fallback step is excluded from the cache. |
| **Diverged** | A reaction whose final rollout product ≠ recorded product. Excluded from the cache. |
| **Suspect intermediate** | A molecule whose canonical-SMILES round-trip isn't a fixed point — indicates aromaticity / tautomer perception drift. |
| **Parquet cache** | Canonical training input at `data/cache/parquet/{reactions,steps}.parquet`. See [ADR-0004](adr/0004-parquet-canonical-cache.md). |
| **Split hash** | `int.from_bytes(sha1(rxn_id)[:4], "big")`. Maps each reaction to a deterministic train/val/test bucket. |

## Training

| Term | Definition |
|---|---|
| **Task mode** | `"stepwise"` (one item per step, Δ between intermediates) or `"end_to_end"` (one item per reaction, Δ R→P). |
| **Productive fragment** | After applying R→P Δ, the connected fragment(s) containing atoms whose maps appear in P. Filters out leaving groups. |
| **Featurization** | RDKit `Mol` → PyG `Data(x, edge_index, edge_attr)`. Atom features include element, charge, H count, aromaticity, hybridization; bond features include order, ring membership. |

## Conventions

- All bond Δ tensors are **symmetric** (`Δ[i,j] == Δ[j,i]`).
- All formal charges are integers (no fractional charges).
- Atom maps in the CSV are 1-indexed and ≤ 100 for "productive" atoms; ≥ 101 for leaving groups.
- Float arrows `N.k` use `int(N)` as the heavy atom and `k` as the H index (which H of N).
