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
        help=(
            "Data augmentation preset or JSON array of AutoGluon transform names. "
            "Presets: none=[resize_shorter_side,center_crop], "
            "light=none+random_horizontal_flip, "
            "medium=light+color_jitter+trivial_augment, "
            "heavy=[random_resize_crop,random_horizontal_flip,random_vertical_flip,"
            "color_jitter,trivial_augment,randaug]. "
            'Custom JSON example: \'["random_resize_crop","color_jitter"]\'.'
        ),
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
        "--resume",
        action="store_true",
        help=(
            "Resume an existing AutoGluon run from checkpoint in --out-dir/AutogluonModels/<base-model>. "
            "When used with a larger --max-epochs, training continues to the new epoch limit."
        ),
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Optimization learning rate (optim.lr).",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
        help="Optimizer weight decay (optim.weight_decay).",
    )
    p.add_argument(
        "--warmup-steps",
        type=float,
        default=0.1,
        help="LR warmup proportion/steps (optim.warmup_steps).",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience checks (optim.patience).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of checkpoints used for model averaging (optim.top_k).",
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
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="User provided device_type of 'cuda', but CUDA is not available. Disabling",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.",
        category=UserWarning,
    )

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
        resume=args.resume,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        patience=args.patience,
        top_k=args.top_k,
        focal_loss=args.focal_loss,
        focal_loss_gamma=args.focal_loss_gamma,
        device=ag_device_map(device),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_threads=args.num_threads,
    )

    print(f"Model saved to: {model_dir}")
