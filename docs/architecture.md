# Architecture

End-to-end data and module flow. Companion to [docs/glossary.md](glossary.md) and [docs/repo-map.md](repo-map.md).

## Data pipeline

```mermaid
flowchart LR
    CSV["data/mech-USPTO-31k.csv<br/>(R, P, mechanistic_label, class)"]
    Parser["parser.py<br/>MechUSPTOParser"]
    Arrows["arrow_parser.py<br/>parse_arrows"]
    Grouper["arrow_parser.py<br/>group_arrows_into_steps<br/>(chain-rule)"]
    Apply["chemistry/apply_delta.py<br/>apply_delta"]
    Cache["build_parquet_cache.py<br/>clean-rollout filter"]
    Parquet[("data/cache/parquet/<br/>reactions.parquet<br/>steps.parquet")]

    CSV --> Parser --> Arrows --> Grouper --> Apply --> Cache --> Parquet
```

## Training loop

```mermaid
flowchart LR
    Parquet[("parquet cache")]
    Dataset["parquet_dataset.py<br/>ParquetMechDataset"]
    Featurize["featurization.py<br/>featurize_nodes/edges"]
    Loader["loaders.py<br/>create_dataloaders"]
    Model["models/transformer.py<br/>ReactionTransformer<br/>+ models/heads.py"]
    Loss["losses/focal.py<br/>MaskedFocalLossWithSpectators"]
    Engine["training/engine.py<br/>TrainingEngine"]
    Ckpt[("checkpoints/")]
    Results[("results/")]

    Parquet --> Dataset --> Featurize --> Loader --> Model --> Loss --> Engine
    Engine --> Ckpt
    Engine --> Results
```

## Inference / evaluation

```mermaid
flowchart LR
    Ckpt[("checkpoint.pt")]
    Eval["scripts/evaluate.py"]
    Metrics["training/metrics.py<br/>precision/recall/f1/topk"]
    Json[("results/eval_*.json")]

    Ckpt --> Eval --> Metrics --> Json
```

## Verification harness

```mermaid
flowchart TB
    subgraph Pre-flight
        Sanity["sanity_check.py"]
        Tests["pytest -q"]
    end
    subgraph Cache-correctness
        E2E["verify_apply_delta_e2e.py<br/>(R → apply_delta → P, single shot)"]
        Seq["verify_apply_delta_sequential.py<br/>(Δ chunked into K parts)"]
        Roll["verify_cache_rollout.py<br/>(sequential apply_delta over cached steps)"]
        Audit["cache_audit.py<br/>(Δ range, splits, class balance)"]
    end
    Charge["charge_diagnostic.py<br/>(heuristic accuracy)"]

    Sanity -.before training.-> E2E
    Tests -.before training.-> Roll
    E2E --> Audit
    Seq --> Audit
    Roll --> Audit
```

## Module map

| Package | Responsibility |
|---|---|
| `mech_uspto.constants` | Atom/bond enumerations and shared constants. |
| `mech_uspto.data` | CSV parsing, arrow → step grouping, featurization, dataset, dataloader, parquet cache I/O, spectator detection, Δ matrix transformations. |
| `mech_uspto.chemistry` | `apply_delta` (bond surgery on RDKit Mol) and the valence/VTS counter. |
| `mech_uspto.models` | Graph transformer body + prediction heads. |
| `mech_uspto.losses` | Focal loss with spectator masking. |
| `mech_uspto.training` | Engine (train/val loop), config, metrics, run tracking. |

Per-symbol detail: regenerate [docs/repo-map.md](repo-map.md) with `python scripts/gen_repo_map.py`.
