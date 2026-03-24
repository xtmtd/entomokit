"""entomokit classify train — train an AutoGluon image classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "train",
        help="Train an AutoGluon image classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--train-csv", required=True, help="CSV with 'image' and 'label' columns."
    )
    p.add_argument(
        "--images-dir", required=True, help="Directory containing training images."
    )
    p.add_argument(
        "--base-model", default="convnextv2_femto", help="timm backbone name."
    )
    p.add_argument("--out-dir", required=True, help="Output directory.")
    p.add_argument(
        "--augment",
        default="medium",
        help="Augment preset: none/light/medium/heavy or JSON array.",
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs.",
    )
    p.add_argument(
        "--time-limit", type=float, default=1.0, help="Training time limit in hours."
    )
    p.add_argument(
        "--focal-loss",
        action="store_true",
        help="Use focal loss to emphasize hard examples.",
    )
    p.add_argument(
        "--focal-loss-gamma",
        type=float,
        default=1.0,
        help="Gamma value for focal loss weighting.",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device for training.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size used during training.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader worker processes.",
    )
    p.add_argument(
        "--num-threads", type=int, default=0, help="CPU threads for PyTorch (0 = auto)."
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from pathlib import Path
    from src.classification.utils import resolve_augment, select_device, ag_device_map
    from src.classification.trainer import train
    from src.common.cli import save_log

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    save_log(out_dir / "logs", args, log_filename="log.txt")

    augment_transforms = resolve_augment(args.augment)  # raises on invalid
    device = select_device(args.device)

    model_dir = train(
        train_csv=Path(args.train_csv),
        images_dir=Path(args.images_dir),
        base_model=args.base_model,
        out_dir=out_dir,
        augment_transforms=augment_transforms,
        max_epochs=args.max_epochs,
        time_limit_hours=args.time_limit,
        focal_loss=args.focal_loss,
        focal_loss_gamma=args.focal_loss_gamma,
        device=ag_device_map(device),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_threads=args.num_threads,
    )

    print(f"Model saved to: {model_dir}")
