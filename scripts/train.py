"""CLI entrypoint for ablation-study training.

Usage::

    python scripts/train.py --task-mode stepwise --num-epochs 100
    python scripts/train.py --task-mode end_to_end --batch-size 16
"""

print("↑ Python started, importing dependencies (torch/pyg/rdkit ≈ 5–15s)...", flush=True)

import argparse
import sys

import torch

from mech_uspto.data.loaders import create_dataloaders, seed_everything
from mech_uspto.training.config import DEFAULT_DATA_PATH, Config
from mech_uspto.training.engine import TrainingEngine
from mech_uspto.training.tracking import make_tracker

print(f"↓ Imports done. torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}", flush=True)


def _device_banner(require_cuda: bool) -> None:
    """Print a loud, unambiguous device banner and optionally abort on CPU.

    Prevents silent CPU fallback — a CPU-only torch wheel on a GPU host is
    one of the most common and least visible failure modes in PyTorch.
    With ``require_cuda=True`` (recommended for cluster jobs), exits with
    code 2 instead of training at ~1/50 the expected throughput.
    """
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print("=" * 70, flush=True)
        print(f"✅ DEVICE: CUDA   |   {name}   |   sm_{cap[0]}{cap[1]}   |   torch CUDA={torch.version.cuda}", flush=True)
        print("=" * 70, flush=True)
        return
    msg_lines = [
        "=" * 70,
        "🚨 DEVICE: CPU   —   torch.cuda.is_available() == False",
        f"   torch={torch.__version__}   torch.version.cuda={torch.version.cuda!r}",
        "   If you intended to use a GPU, the installed torch wheel is the CPU",
        "   build (or the CUDA runtime is missing). Reinstall with:",
        "     pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1",
        "=" * 70,
    ]
    for line in msg_lines:
        print(line, flush=True)
    if require_cuda:
        print("--require-cuda set; aborting before training starts.", flush=True)
        sys.exit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ReactionTransformer on mech-USPTO-31k")
    parser.add_argument(
        "--csv", type=str, default=DEFAULT_DATA_PATH, help="Path to mech-USPTO-31k CSV file"
    )
    parser.add_argument(
        "--task-mode",
        type=str,
        default="stepwise",
        choices=["stepwise", "end_to_end"],
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-weights",
        type=str,
        default=None,
        help=(
            "Comma-separated per-class loss weights "
            "(length must match num_classes for the task mode). "
            "Default: hand-picked weights from Config."
        ),
    )
    parser.add_argument(
        "--gamma-focal",
        type=float,
        default=None,
        help="Focal-loss gamma (default: 3.5 from Config). 0 disables focal weighting.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help=(
            "Linear LR warmup over the first N optimizer steps (0 disables). "
            "~5%% of total train steps is a good starting point."
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint .pt to resume from. Restores model, optimizer, "
            "history, and best-val-loss; continues training from checkpoint epoch + 1."
        ),
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable bf16 autocast on CUDA (default: enabled). Use to debug numerical issues.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help=(
            "Abort with exit code 2 if torch.cuda.is_available() is False. Use on "
            "GPU jobs to fail fast instead of silently falling back to CPU (~50x "
            "slowdown). Recommended for all cluster submissions."
        ),
    )
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help=(
            "Disable cuDNN deterministic mode (default: enabled). Trades reproducibility "
            "for ~5-10%% throughput. Only use after baseline runs are reproduced."
        ),
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="none",
        choices=["none", "wandb"],
        help="Experiment tracker backend (default: none). Use 'wandb' for W&B.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="mech-uspto",
        help="W&B project name (only used when --tracker wandb).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _device_banner(require_cuda=args.require_cuda)

    class_weights = None
    if args.class_weights is not None:
        class_weights = torch.tensor([float(x) for x in args.class_weights.split(",")])

    config_kwargs = dict(
        csv_path=args.csv,
        task_mode=args.task_mode,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    if class_weights is not None:
        config_kwargs["class_weights"] = class_weights
    if args.gamma_focal is not None:
        config_kwargs["gamma_focal"] = args.gamma_focal
    if args.warmup_steps:
        config_kwargs["warmup_steps"] = args.warmup_steps
    if args.no_amp:
        config_kwargs["use_amp"] = False
    if args.no_deterministic:
        config_kwargs["deterministic"] = False

    config = Config(**config_kwargs)

    seed_everything(config.seed, deterministic=config.deterministic)

    print(f"📂 Loading dataset from {config.csv_path}...")
    dataloaders = create_dataloaders(
        config.csv_path,
        task_mode=config.task_mode,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        compute_spectators=True,
    )

    engine = TrainingEngine(
        config,
        tracker=make_tracker(
            args.tracker,
            **({"project": args.wandb_project} if args.tracker == "wandb" else {}),
        ),
    )
    if args.resume is not None:
        engine.load_state(args.resume)
    engine.train(dataloaders["train"], dataloaders["val"])
    engine.save_results()

    print("\n✅ Training completed!")
    print(f"   Best epoch: {engine.best_epoch}")
    print(f"   Best val loss: {engine.best_val_loss:.3e}")
    print(f"   Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
