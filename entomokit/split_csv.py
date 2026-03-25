"""entomokit split-csv — split a CSV dataset into train/val/test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from entomokit.help_style import RichHelpFormatter, style_parser, with_examples


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "split-csv",
        help="Split a CSV dataset (image, label) into train/val/test splits.",
        description=with_examples(
            "Split a CSV dataset (image, label) into train/val/test splits.",
            [
                "entomokit split-csv --raw-image-csv data.csv --out-dir ./datasets",
                "entomokit split-csv --raw-image-csv data.csv --mode ratio --known-test-classes-ratio 0.1 --out-dir ./datasets",
            ],
        ),
        formatter_class=RichHelpFormatter,
    )
    style_parser(p)
    p.add_argument(
        "--raw-image-csv",
        required=True,
        help="Input CSV with 'image' and 'label' columns.",
    )
    p.add_argument(
        "--mode",
        choices=["ratio", "count"],
        default="ratio",
        help="Split strategy: use class ratios or explicit class counts.",
    )
    p.add_argument(
        "--unknown-test-classes-ratio",
        type=float,
        default=0.0,
        help="Fraction of classes reserved for unknown-class test split (ratio mode).",
    )
    p.add_argument(
        "--known-test-classes-ratio",
        type=float,
        default=0.1,
        help="Fraction of known classes moved to test split (ratio mode).",
    )
    p.add_argument(
        "--unknown-test-classes-count",
        type=int,
        default=0,
        help="Number of classes reserved for unknown-class test split (count mode).",
    )
    p.add_argument(
        "--known-test-classes-count",
        type=int,
        default=0,
        help="Number of known classes moved to test split (count mode).",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Val split ratio (from train). 0 = no val split.",
    )
    p.add_argument(
        "--val-count",
        type=int,
        default=0,
        help="Val split count (from train). 0 = no val split.",
    )
    p.add_argument(
        "--min-count-per-class",
        type=int,
        default=0,
        help="Drop classes with fewer than this number of images.",
    )
    p.add_argument(
        "--max-count-per-class",
        type=int,
        default=None,
        help="Cap images per class before splitting (None = no cap).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )
    p.add_argument(
        "--out-dir",
        default="datasets",
        help="Directory to write split CSV files and optional copied images.",
    )
    p.add_argument(
        "--images-dir",
        default=None,
        help="Source image directory. Required when --copy-images is set.",
    )
    p.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into out_dir/images/{split}/ subdirectories.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output during splitting.",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from pathlib import Path
    from src.common.cli import setup_shutdown_handler, save_log
    from src.splitting.splitter import DatasetSplitter

    setup_shutdown_handler()

    if args.copy_images and not args.images_dir:
        print(
            "Error: --images-dir is required when --copy-images is set.",
            file=sys.stderr,
        )
        sys.exit(1)

    csv_path = Path(args.raw_image_csv)
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_log(out_dir, args)

    splitter = DatasetSplitter(
        raw_image_csv=str(csv_path),
        out_dir=str(out_dir),
        seed=args.seed,
    )

    results = splitter.split(
        mode=args.mode,
        unknown_test_ratio=args.unknown_test_classes_ratio,
        known_test_ratio=args.known_test_classes_ratio,
        unknown_test_count=args.unknown_test_classes_count,
        known_test_count=args.known_test_classes_count,
        min_count_per_class=args.min_count_per_class,
        max_count_per_class=args.max_count_per_class,
        val_ratio=args.val_ratio,
        val_count=args.val_count,
        copy_images=args.copy_images,
        images_dir=Path(args.images_dir) if args.images_dir else None,
    )

    print(f"All outputs saved in {out_dir}")
    print(
        f"Train: {results['train']}, Val: {results.get('val', 0)}, "
        f"Test known: {results['test_known']}, Test unknown: {results['test_unknown']}"
    )
