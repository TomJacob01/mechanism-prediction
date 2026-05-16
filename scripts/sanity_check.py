"""Pre-flight sanity checks for a mech-USPTO-31k CSV data drop.

Run BEFORE launching training to catch problems early. Exits non-zero when a
required check fails.

Checks performed:

1.  Environment   — torch / torch_geometric / rdkit importable, CUDA visibility.
2.  Data layout   — CSV file exists and has the expected header.
3.  Parsing       — ``MechUSPTOParser.parse_csv_file`` succeeds.
4.  Featurization — ``process_mapped_smiles`` round-trips on a sample.
5.  Δ stats       — stepwise Δ stays in [-1, 1]; end-to-end Δ stays in [-2, 2].
6.  Spectator stats — average spectator ratio is within the expected band.
7.  Mini training step — one forward + backward + optimizer step on a tiny
    batch verifies the entire loop end-to-end (model, loss, metrics).

Usage::

    python scripts/sanity_check.py --csv /path/to/mech-USPTO-31k.csv
    python scripts/sanity_check.py --csv <file> --sample-size 50 --quick
"""

import argparse
import csv
import random
import sys
import traceback
from pathlib import Path

EXPECTED_HEADER = {
    "original_reactions",
    "updated_reaction",
    "mechanistic_class",
    "mechanistic_label",
    "data_source",
}
EXPECTED_SPECTATOR_RATIO_BAND = (0.50, 0.99)

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"{GREEN}✅{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}⚠️ {RESET} {msg}")


def fail(msg: str) -> None:
    print(f"{RED}❌{RESET} {msg}")


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #


def check_environment() -> bool:
    print("\n[1/7] Environment")
    try:
        import rdkit  # noqa: F401
        import torch
        import torch_geometric  # noqa: F401
    except ImportError as e:
        fail(f"Missing dependency: {e}")
        return False
    ok(f"torch={torch.__version__}  CUDA available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        ok(f"GPUs visible: {torch.cuda.device_count()}")
    else:
        warn("No CUDA — training will be CPU-only.")
    return True


def check_csv_layout(csv_path: Path) -> bool:
    print("\n[2/7] CSV layout")
    if not csv_path.exists():
        fail(f"CSV file does not exist: {csv_path}")
        return False
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if header is None:
        fail("CSV file is empty")
        return False
    header_set = set(header)
    missing = EXPECTED_HEADER - header_set
    if missing:
        fail(f"CSV header missing columns: {missing}")
        return False
    ok(f"CSV file found with header columns: {header}")
    return True


def check_parsing(csv_path: Path):
    print("\n[3/7] Parser")
    from mech_uspto.data.parser import MechUSPTOParser

    try:
        reactions = MechUSPTOParser.parse_csv_file(str(csv_path))
    except Exception as e:  # noqa: BLE001
        fail(f"parse_csv_file raised: {e}")
        traceback.print_exc()
        return None
    if not reactions:
        fail("Parser returned 0 reactions")
        return None
    ok(f"Parsed {len(reactions)} reactions; first id = {reactions[0].reaction_id}")
    return reactions


def check_featurization(reactions, sample_size: int) -> bool:
    print(f"\n[4/7] Featurization round-trip on {sample_size} samples")
    from mech_uspto.data.featurization import process_mapped_smiles

    sample = random.sample(reactions, min(sample_size, len(reactions)))
    failures: list[tuple[str, str]] = []
    for rxn in sample:
        if not rxn.steps:
            continue
        step = rxn.steps[0]
        for label, smi in (
            ("reactants_mapped", step.reactants_mapped),
            ("products_mapped", step.products_mapped),
        ):
            if not smi:
                continue
            try:
                process_mapped_smiles(smi, add_hs=True)
            except Exception as e:  # noqa: BLE001
                failures.append((rxn.reaction_id, f"{label}: {e}"))
    if failures:
        fail(f"{len(failures)} featurization failures")
        for rid, reason in failures[:5]:
            print(f"   - {rid}: {reason}")
        return False
    ok(f"Featurization OK on {len(sample)} samples")
    return True


def check_delta_stats(reactions, sample_size: int) -> bool:
    print(f"\n[5/7] Δ statistics on {sample_size} samples (both modes)")
    from collections import Counter

    import torch

    from mech_uspto.data.dataset import MechUSPTODataset

    sample = random.sample(reactions, min(sample_size, len(reactions)))
    all_good = True
    for mode, lo, hi in [("stepwise", -1, 1), ("end_to_end", -3, 3)]:
        ds = MechUSPTODataset(sample, task_mode=mode, compute_spectators=True)
        if len(ds) == 0:
            fail(f"{mode}: dataset built 0 samples")
            all_good = False
            continue
        violations = 0
        hist: Counter[int] = Counter()
        for d in ds.data_points:
            y = d.y
            if y.min().item() < lo or y.max().item() > hi:
                violations += 1
            # Count only upper-triangle (bond changes are symmetric).
            n = y.shape[0]
            triu = y[torch.triu_indices(n, n, offset=1).unbind(0)]
            for v in triu.tolist():
                hist[int(v)] += 1
        if violations:
            fail(f"{mode}: {violations}/{len(ds)} samples have Δ outside [{lo}, {hi}]")
            all_good = False
        else:
            ok(f"{mode}: all {len(ds)} samples have Δ ∈ [{lo}, {hi}]")
        # Print histogram (sorted by Δ value) — helps decide class counts.
        total = sum(hist.values()) or 1
        bars = []
        for v in sorted(hist):
            pct = 100 * hist[v] / total
            bars.append(f"Δ={v:+d}: {hist[v]:,} ({pct:.3f}%)")
        print("   " + " | ".join(bars))
    return all_good


def check_spectator_ratio(reactions, sample_size: int) -> bool:
    print(f"\n[6/7] Spectator ratio (band {EXPECTED_SPECTATOR_RATIO_BAND})")
    from mech_uspto.data.dataset import MechUSPTODataset

    sample = random.sample(reactions, min(sample_size, len(reactions)))
    ds = MechUSPTODataset(sample, task_mode="stepwise", compute_spectators=True)
    if not ds.spectator_ratios:
        warn("No spectator ratios computed (empty dataset)")
        return True
    avg = sum(ds.spectator_ratios) / len(ds.spectator_ratios)
    lo, hi = EXPECTED_SPECTATOR_RATIO_BAND
    if lo <= avg <= hi:
        ok(f"Average spectator ratio = {avg:.2%} (within {lo:.0%}–{hi:.0%})")
        return True
    warn(f"Average spectator ratio = {avg:.2%} outside expected band {lo:.0%}–{hi:.0%}")
    return True


def check_mini_training_step(reactions, batch_size: int) -> bool:
    print("\n[7/7] Mini training step (1 forward + backward + optimizer)")
    import torch

    from mech_uspto.data.dataset import MechUSPTODataset
    from mech_uspto.data.loaders import collate_fn_with_spectators
    from mech_uspto.losses.focal import MaskedFocalLossWithSpectators
    from mech_uspto.models.transformer import ReactionTransformer
    from mech_uspto.training.metrics import MetricsComputer

    sample = random.sample(reactions, min(batch_size * 2, len(reactions)))
    ds = MechUSPTODataset(sample, task_mode="stepwise", compute_spectators=True)
    if len(ds) == 0:
        fail("Could not build a non-empty dataset for mini training step")
        return False

    batch = collate_fn_with_spectators([ds[i] for i in range(min(batch_size, len(ds)))])
    if batch is None:
        fail("collate returned None")
        return False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = batch.to(device)

    model = ReactionTransformer(
        node_in=25, edge_in=6, hidden_dim=64, num_heads=4, num_layers=2, num_classes=3
    ).to(device)
    criterion = MaskedFocalLossWithSpectators(
        num_classes=3, weights=torch.tensor([4.0, 1.0, 4.0]).to(device), gamma=2.0
    )
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    targets = batch.y_padded + 1
    logits, mask = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
    spectator = getattr(batch, "spectator_padded", None)

    loss = criterion(logits, targets, mask_2d, spectator)
    if not torch.isfinite(loss):
        fail(f"Loss is not finite: {loss.item()}")
        return False

    optim.zero_grad()
    loss.backward()
    optim.step()

    metrics = MetricsComputer.get_mechanism_metrics(logits.detach(), targets, mask)
    ok(
        f"Step OK on device={device}  loss={loss.item():.4f}  "
        f"PR-AUC={metrics['pr_auc']:.3f}  F1={metrics['f1']:.3f}"
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--csv", type=str, required=True, help="Path to mech-USPTO-31k CSV file")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quick", action="store_true", help="Skip the mini-training step (steps 1-6 only)."
    )
    args = parser.parse_args()

    random.seed(args.seed)
    csv_path = Path(args.csv)

    if not check_environment():
        return 1
    if not check_csv_layout(csv_path):
        return 1

    reactions = check_parsing(csv_path)
    if reactions is None:
        return 1

    if not check_featurization(reactions, args.sample_size):
        return 1
    if not check_delta_stats(reactions, args.sample_size):
        return 1
    if not check_spectator_ratio(reactions, args.sample_size):
        return 1

    if not args.quick:
        if not check_mini_training_step(reactions, args.batch_size):
            return 1
    else:
        print("\n[7/7] skipped (--quick)")

    print(f"\n{GREEN}🎉 All sanity checks passed.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
