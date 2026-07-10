"""entomokit measure - morphology metrics from mask images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from entomokit.help_style import RichHelpFormatter, style_parser, with_examples


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "measure",
        help="Measure morphology metrics from segmentation masks.",
        description=with_examples(
            "Measure morphology metrics from segmentation masks.",
            [
                "entomokit measure --mask-dir ./segmented/masks --out-dir ./metrics",
                "entomokit measure --mask-dir ./masks --out-dir ./metrics --pixel-size-um 2.5",
            ],
        ),
        formatter_class=RichHelpFormatter,
    )
    style_parser(p)
    p.add_argument("--mask-dir", "-i", required=True, help="Input mask directory.")
    p.add_argument("--out-dir", "-o", required=True, help="Output directory.")
    p.add_argument(
        "--pixel-size-um",
        type=float,
        default=None,
        help="Pixel size in micrometers per pixel (um/px).",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging."
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Append measurements for new masks; skip masks already in metrics.csv.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete --out-dir contents and start fresh.",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from src.common.cli import save_log, setup_logging, setup_shutdown_handler, check_output_dir, reset_shutdown_flag, get_shutdown_flag
    from src.measurement.service import run_batch

    reset_shutdown_flag()
    setup_shutdown_handler()

    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)

    if not mask_dir.exists() or not mask_dir.is_dir():
        print(
            f"Error: --mask-dir does not exist or is not a directory: {mask_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.pixel_size_um is not None and args.pixel_size_um <= 0:
        print("Error: --pixel-size-um must be positive.", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    check_output_dir(out_dir, args.resume, args.overwrite)
    logger = setup_logging(out_dir, verbose=args.verbose)
    save_log(out_dir, args)

    existing_rows: list[dict[str, str]] = []
    if args.resume:
        import csv as _csv
        existing_csv = out_dir / "metrics.csv"
        if existing_csv.exists():
            with existing_csv.open(newline="", encoding="utf-8") as f:
                reader = _csv.DictReader(f)
                existing_rows = list(reader)
            print(f"Resuming: skipping {len(existing_rows)} already-measured masks.")

    try:
        summary = run_batch(
            mask_dir=mask_dir,
            out_dir=out_dir,
            pixel_size_um=args.pixel_size_um,
            existing_rows=existing_rows or None,
            shutdown_flag=get_shutdown_flag(),
        )
    except Exception:
        logger.exception("Measurement failed")
        sys.exit(1)

    logger.info("Measurement complete")
    logger.info(
        "  Total: %s, ok: %s, warn: %s, fail: %s",
        summary["total"],
        summary["ok"],
        summary["warn"],
        summary["fail"],
    )
    logger.info("  Metrics CSV: %s", out_dir / "metrics.csv")
    logger.info("  Summary CSV: %s", out_dir / "metrics_summary.csv")
    logger.info("  Metric definitions: %s", out_dir / "metric_definitions.csv")
    logger.warning(
        "Caution: body_length/body_width are estimates from mask geometry and may be inaccurate when masks include appendages, are truncated at image borders, or contain merged/fragmented regions."
    )
    logger.warning(
        "Please review warn_reason and mask quality before using these values for biological conclusions."
    )
