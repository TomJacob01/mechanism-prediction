"""CLI entrypoint for ablation-study training.

Usage::

    python scripts/train.py --task-mode stepwise --num-epochs 100
    python scripts/train.py --task-mode end_to_end --batch-size 16
"""

import argparse

import numpy as np
import torch

from mech_uspto.data.loaders import create_dataloaders
from mech_uspto.training.config import DEFAULT_DATA_DIR, Config
from mech_uspto.training.engine import TrainingEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ReactionTransformer on mech-USPTO-31k")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = Config(
        data_dir=args.data_dir,
        task_mode=args.task_mode,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    print(f"📂 Loading dataset from {config.data_dir}...")
    dataloaders = create_dataloaders(
        config.data_dir,
        task_mode=config.task_mode,
        batch_size=config.batch_size,
        compute_spectators=True,
    )

    engine = TrainingEngine(config)
    engine.train(dataloaders["train"], dataloaders["val"])
    engine.save_results()

    print("\n✅ Training completed!")
    print(f"   Best epoch: {engine.best_epoch}")
    print(f"   Best val loss: {engine.best_val_loss:.4f}")
    print(f"   Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
