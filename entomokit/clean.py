"""entomokit clean — image cleaning and deduplication."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from entomokit.help_style import RichHelpFormatter, style_parser, with_examples


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "clean",
        help="Clean and deduplicate images.",
        description=with_examples(
            "Clean and deduplicate images.",
            [
                "entomokit clean --input-dir ./raw --out-dir ./cleaned",
                "entomokit clean --input-dir ./raw --out-dir ./cleaned --dedup-mode phash --phash-threshold 5",
            ],
        ),
        formatter_class=RichHelpFormatter,
    )
    style_parser(p)
    p.add_argument("--input-dir", required=True, help="Input images directory.")
    p.add_argument("--out-dir", required=True, help="Output directory.")
    p.add_argument(
        "--out-short-size",
        type=int,
        default=512,
        help="Resize shorter edge. Use -1 to keep original.",
    )
    p.add_argument(
        "--out-image-format",
        default="jpg",
        choices=["jpg", "png", "tif"],
        help="Output image format for cleaned files.",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=12,
        help="Number of worker threads for image processing.",
    )
    p.add_argument(
        "--keep-exif",
        action="store_true",
        help="Preserve EXIF metadata in output images.",
    )
    p.add_argument(
        "--dedup-mode",
        default="md5",
        choices=["none", "md5", "phash"],
        help="Deduplication strategy: none, exact hash (md5), or perceptual hash (phash).",
    )
    p.add_argument(
        "--phash-threshold",
        type=int,
        default=5,
        help="Maximum pHash distance for treating two images as duplicates.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories in --input-dir.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose progress output.",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from pathlib import Path
    from src.common.cli import setup_shutdown_handler, save_log
    from src.cleaning.processor import ImageCleaner

    setup_shutdown_handler()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        print(
            f"Error: --input-dir does not exist or is not a directory: {input_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    images_subdir = out_dir / "cleaned_images"
    images_subdir.mkdir(parents=True, exist_ok=True)
    save_log(out_dir, args)

    cleaner = ImageCleaner(
        input_dir=str(input_dir),
        output_dir=str(images_subdir),
        out_short_size=args.out_short_size,
        out_image_format=args.out_image_format,
        dedup_mode=args.dedup_mode,
        phash_threshold=args.phash_threshold,
        threads=args.threads,
        keep_exif=args.keep_exif,
    )

    log_path = str(out_dir / "log.txt")
    results = cleaner.process_directory(
        log_path=log_path,
        recursive=args.recursive,
    )

    print(f"Done. Processed {results['processed']} images, {results['errors']} errors.")
    print(f"Cleaned images saved to: {images_subdir}")
