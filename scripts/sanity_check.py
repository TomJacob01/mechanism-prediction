"""Pre-flight sanity checks for a mech-USPTO-31k data drop.

Run BEFORE launching training to catch problems early. Exits non-zero when a
required check fails.

Checks performed:

1.  Environment   — torch / torch_geometric / rdkit importable, CUDA visibility.
2.  Data layout   — ``data_dir`` exists and contains ``*.json`` files.
3.  Schema        — random JSON files have the required top-level keys.
4.  Parsing       — ``MechUSPTOParser.parse_batch`` succeeds on a sample.
5.  Featurization — ``process_mapped_smiles`` round-trips on a sample.
6.  Δ stats       — stepwise Δ stays in [-1, 1]; end-to-end Δ stays in [-2, 2].
7.  Spectator stats — average spectator ratio is within the expected band.
8.  Mini training step — one forward + backward + optimizer step on a tiny
    batch verifies the entire loop end-to-end (model, loss, metrics).

Usage::

    python scripts/sanity_check.py --data-dir /path/to/mech-USPTO-31k
    python scripts/sanity_check.py --data-dir <dir> --sample-size 50 --quick
"""

import argparse
import json
import random
import sys
import traceback
from pathlib import Path

REQUIRED_TOP_KEYS = {"rxn_id", "steps"}
REQUIRED_STEP_KEYS = {"id", "reactants_mapped", "products_mapped"}
EXPECTED_SPECTATOR_RATIO_BAND = (0.50, 0.99)  # USPTO molecules are mostly inert.

# ANSI colour codes (no extra deps).
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
    print("\n[1/8] Environment")
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


def check_data_layout(data_dir: Path) -> list[Path] | None:
    print("\n[2/8] Data layout")
    if not data_dir.exists():
        fail(f"data_dir does not exist: {data_dir}")
        return None
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        fail(f"No *.json files found in {data_dir}")
        return None
    ok(f"Found {len(json_files)} JSON files in {data_dir}")
    return json_files


def check_schema(json_files: list[Path], sample_size: int) -> bool:
    print(f"\n[3/8] Schema check on {sample_size} random files")
    sample = random.sample(json_files, min(sample_size, len(json_files)))
    bad: list[tuple[Path, str]] = []
    for path in sample:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:  # noqa: BLE001
            bad.append((path, f"unreadable JSON: {e}"))
            continue
        missing = REQUIRED_TOP_KEYS - set(data)
        if missing:
            bad.append((path, f"missing top-level keys: {missing}"))
            continue
        if not isinstance(data["steps"], list) or not data["steps"]:
            bad.append((path, "empty or malformed 'steps'"))
            continue
        for step in data["steps"]:
            step_missing = REQUIRED_STEP_KEYS - set(step)
            if step_missing:
                bad.append((path, f"step missing keys: {step_missing}"))
                break
    if bad:
        fail(f"{len(bad)}/{len(sample)} files failed schema check")
        for path, reason in bad[:5]:
            print(f"   - {path.name}: {reason}")
        return False
    ok(f"All {len(sample)} sampled files have required keys")
    return True


def check_parsing(data_dir: Path):
    print("\n[4/8] Parser")
    from mech_uspto.data.parser import MechUSPTOParser

    try:
        reactions = MechUSPTOParser.parse_batch(str(data_dir))
    except Exception as e:  # noqa: BLE001
        fail(f"parse_batch raised: {e}")
        traceback.print_exc()
        return None
    if not reactions:
        fail("Parser returned 0 reactions")
        return None
    ok(f"Parsed {len(reactions)} reactions; first id = {reactions[0].reaction_id}")
    return reactions


def check_featurization(reactions, sample_size: int) -> bool:
    print(f"\n[5/8] Featurization round-trip on {sample_size} samples")
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
    print(f"\n[6/8] Δ statistics on {sample_size} samples (both modes)")
    from mech_uspto.data.dataset import MechUSPTODataset

    sample = random.sample(reactions, min(sample_size, len(reactions)))
    all_good = True
    for mode, lo, hi in [("stepwise", -1, 1), ("end_to_end", -2, 2)]:
        ds = MechUSPTODataset(sample, task_mode=mode, compute_spectators=True)
        if len(ds) == 0:
            fail(f"{mode}: dataset built 0 samples")
            all_good = False
            continue
        violations = 0
        for d in ds.data_points:
            if d.y.min().item() < lo or d.y.max().item() > hi:
                violations += 1
        if violations:
            fail(f"{mode}: {violations}/{len(ds)} samples have Δ outside [{lo}, {hi}]")
            all_good = False
        else:
            ok(f"{mode}: all {len(ds)} samples have Δ ∈ [{lo}, {hi}]")
    return all_good


def check_spectator_ratio(reactions, sample_size: int) -> bool:
    print(f"\n[7/8] Spectator ratio (band {EXPECTED_SPECTATOR_RATIO_BAND})")
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
    return True  # warning, not a hard failure


def check_mini_training_step(reactions, batch_size: int) -> bool:
    print("\n[8/8] Mini training step (1 forward + backward + optimizer)")
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

    targets = batch.y_padded + 1  # {-1,0,1} → {0,1,2}
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


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quick", action="store_true", help="Skip the mini-training step (steps 1-7 only)."
    )
    args = parser.parse_args()

    random.seed(args.seed)
    data_dir = Path(args.data_dir)

    if not check_environment():
        return 1

    json_files = check_data_layout(data_dir)
    if json_files is None:
        return 1
    if not check_schema(json_files, args.sample_size):
        return 1

    reactions = check_parsing(data_dir)
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
        print("\n[8/8] skipped (--quick)")

    print(f"\n{GREEN}🎉 All sanity checks passed.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
