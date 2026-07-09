"""entomokit augment — image augmentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from entomokit.help_style import RichHelpFormatter, style_parser, with_examples


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "augment",
        help="Augment images with albumentations presets or a custom policy.",
        description=with_examples(
            "Augment images with presets or a custom policy.",
            [
                "entomokit augment --input-dir ./images --out-dir ./augmented",
                "entomokit augment --input-dir ./images --out-dir ./augmented --preset heavy --multiply 3",
            ],
        ),
        formatter_class=RichHelpFormatter,
    )
    style_parser(p)
    p.add_argument(
        "--input-dir",
        required=True,
        help="Input directory containing images (JPG/PNG/etc).",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for augmented images and manifest.",
    )
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--preset",
        default=None,
        help="Augmentation preset: light, medium, heavy, safe-for-small-dataset.",
    )
    mode_group.add_argument(
        "--policy",
        default=None,
        help="Path to custom augmentation policy JSON file. Mutually exclusive with --preset.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--multiply",
        type=int,
        default=1,
        help="Number of augmented copies to create per input image.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip source images already augmented in --out-dir.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete --out-dir contents and start fresh.",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from src.augment.service import run_augment
    from src.common.cli import save_log, setup_shutdown_handler, check_output_dir, get_shutdown_flag, reset_shutdown_flag

    reset_shutdown_flag()
    setup_shutdown_handler()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    check_output_dir(out_dir, args.resume, args.overwrite)
    save_log(out_dir, args)

    custom = None
    preset = args.preset or "light"
    if args.policy is not None:
        custom = json.loads(Path(args.policy).read_text(encoding="utf-8"))
        preset = None

    try:
        result = run_augment(
            input_dir=input_dir,
            out_dir=out_dir,
            preset=preset,
            custom=custom,
            seed=args.seed,
            multiply=args.multiply,
            skip_existing=args.resume,
            shutdown_flag=get_shutdown_flag(),
        )
    except Exception as exc:
        print(f"Augmentation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if result.success:
        print(f"Augmentation complete -> {out_dir}")
        print(f"  Images processed: {result.manifest['images_processed']}")
        print(f"  Preset:           {result.manifest.get('preset', 'custom')}")
        print(f"  Multiply:         {result.manifest.get('multiply', 1)}")
        print(f"Manifest: {out_dir / 'augment_manifest.json'}")
