"""entomokit synthesize — composite target objects onto background images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from entomokit.help_style import RichHelpFormatter, style_parser, with_examples


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "synthesize",
        help="Composite target objects onto background images.",
        description=with_examples(
            "Composite target objects onto background images.",
            [
                "entomokit synthesize --target-dir ./targets --background-dir ./bg --out-dir ./syn",
                "entomokit synthesize --target-dir ./targets --background-dir ./bg --out-dir ./syn --num-syntheses 20",
            ],
        ),
        formatter_class=RichHelpFormatter,
    )
    style_parser(p)
    # --- I/O ---
    p.add_argument(
        "--target-dir",
        "-t",
        required=True,
        help="Target object images directory (cleaned, with alpha channel).",
    )
    p.add_argument(
        "--background-dir",
        "-b",
        required=True,
        help="Background images directory.",
    )
    p.add_argument(
        "--out-dir",
        "-o",
        required=True,
        help="Output directory.",
    )
    # --- Synthesis parameters ---
    p.add_argument(
        "--num-syntheses",
        "-n",
        type=int,
        default=10,
        help="Number of syntheses per target image.",
    )
    p.add_argument(
        "--area-ratio-min",
        "-a",
        type=float,
        default=0.05,
        help="Minimum area ratio (target area / background area), 0.01-0.50.",
    )
    p.add_argument(
        "--area-ratio-max",
        "-x",
        type=float,
        default=0.20,
        help="Maximum area ratio (target area / background area), 0.01-0.50.",
    )
    p.add_argument(
        "--color-match-strength",
        "-c",
        type=float,
        default=0.5,
        help="Color matching strength 0-1.",
    )
    p.add_argument(
        "--avoid-black-regions",
        "-A",
        action="store_true",
        help="Avoid pure black regions in background.",
    )
    p.add_argument(
        "--rotate",
        "-r",
        type=float,
        default=0.0,
        help="Maximum random rotation degrees (0 = no rotation).",
    )
    # --- Output format ---
    p.add_argument(
        "--out-image-format",
        "-f",
        default="png",
        choices=["png", "jpg"],
        help="Output image format.",
    )
    # --- Annotation output ---
    p.add_argument(
        "--annotation-output-format",
        default="coco",
        choices=["coco", "voc", "yolo"],
        help="Output format for annotations.",
    )
    p.add_argument(
        "--coco-output-mode",
        default="unified",
        choices=["unified", "separate"],
        help="COCO output mode: unified (single annotations.json) or separate (per-image JSON files).",
    )
    p.add_argument(
        "--coco-bbox-format",
        default="xywh",
        choices=["xywh", "xyxy"],
        help="COCO bbox coordinate convention (only used when --annotation-output-format=coco).",
    )
    # --- Misc ---
    p.add_argument(
        "--threads",
        "-d",
        type=int,
        default=4,
        help="Number of parallel workers.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    _ensure_src_on_path()

    import inspect
    import os

    from src.common.cli import (
        setup_shutdown_handler,
        setup_logging,
        save_log,
    )
    from src.synthesis.processor import SynthesisProcessor

    setup_shutdown_handler()

    max_threads = os.cpu_count() or 8
    if args.threads > max_threads * 2:
        print(
            f"Warning: Thread count {args.threads} exceeds recommended maximum {max_threads * 2}",
            file=sys.stderr,
        )

    target_dir = Path(args.target_dir)
    if not target_dir.exists():
        print(f"Error: Target directory does not exist: {target_dir}", file=sys.stderr)
        sys.exit(1)
    if not target_dir.is_dir():
        print(f"Error: Target path is not a directory: {target_dir}", file=sys.stderr)
        sys.exit(1)

    background_dir = Path(args.background_dir)
    if not background_dir.exists():
        print(
            f"Error: Background directory does not exist: {background_dir}",
            file=sys.stderr,
        )
        sys.exit(1)
    if not background_dir.is_dir():
        print(
            f"Error: Background path is not a directory: {background_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir, verbose=args.verbose)
    save_log(output_dir, args)

    # Build constructor kwargs; coco_bbox_format is a Phase-2 param not yet in SynthesisProcessor.
    # Pass it only if the constructor accepts it to avoid TypeError.
    processor_sig = inspect.signature(SynthesisProcessor.__init__)
    constructor_kwargs: dict = dict(
        output_format=args.out_image_format,
        area_ratio_min=args.area_ratio_min,
        area_ratio_max=args.area_ratio_max,
        color_match_strength=args.color_match_strength,
        avoid_black_regions=args.avoid_black_regions,
        rotate_degrees=args.rotate,
        annotation_format=args.annotation_output_format,
        coco_output_mode=args.coco_output_mode,
    )
    if "coco_bbox_format" in processor_sig.parameters:
        constructor_kwargs["coco_bbox_format"] = args.coco_bbox_format

    logger.info("Starting synthesis process")
    try:
        processor = SynthesisProcessor(**constructor_kwargs)
    except Exception:
        logger.exception("Failed to initialize processor")
        sys.exit(1)

    try:
        results = processor.process_directory(
            target_dir=target_dir,
            background_dir=background_dir,
            output_dir=output_dir,
            num_syntheses=args.num_syntheses,
            disable_tqdm=False,
            threads=args.threads,
        )

        logger.info("Processing complete!")
        logger.info(f"  Successfully processed: {results['processed']}")
        logger.info(f"  Failed: {results['failed']}")
        logger.info(f"  Output files: {results['output_files']}")

    except Exception:
        logger.exception("Processing failed")
        sys.exit(1)


def _ensure_src_on_path() -> None:
    """Add project root to sys.path if needed (for non-editable installs)."""
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
