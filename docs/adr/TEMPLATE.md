# ADR template

When adding a new ADR:
1. Copy this file to `NNNN-short-name.md` (next sequential number).
2. Fill in the sections; delete unused ones.
3. Link from any related ADR and from `.github/copilot-instructions.md` if it changes a top-level convention.
4. Mark status as **Proposed** while under discussion; **Accepted** when merged; **Superseded by ADR-NNNN** when replaced.

---

# ADR-NNNN: Title (verb + object, e.g. "Migrate dataset to Parquet")

Status: **Proposed** (YYYY-MM-DD)

## Context
What problem are we solving? What alternatives exist? What constraints apply? One or two short paragraphs.

## Decision
The single sentence (or two) that someone reading only this section needs to know. Imperative voice: "We will X."

## Rationale
Bullet list of why. Cite numbers / experiments / commit hashes where applicable.

## Consequences
What changes downstream — files affected, conventions enforced, follow-up work created.

## Validation
How we'll know the decision was right (or wrong). Tests, scripts, metrics.

## Related
Links to other ADRs, design docs, or external references.
