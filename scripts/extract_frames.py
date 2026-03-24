#!/usr/bin/env python3
"""
Video Frame Extraction Tool

Extracts frames from video files with multithreading support,
progress tracking, and customizable output parameters.
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import timedelta

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.cli import (
    setup_shutdown_handler,
    get_shutdown_flag,
    setup_logging,
    save_log,
)
from src.framing.extractor import VideoFrameExtractor


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract frames from video files with multithreading support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input_dir ./videos --out_dir ./frames
  %(prog)s --input_dir ./videos --out_dir ./frames --interval 500 --threads 4
  %(prog)s --input_dir ./videos --out_dir ./frames --max_frames 100 --out_image_format png
  %(prog)s --input_dir ./videos --out_dir ./frames --start_time 5.0 --end_time 30.0
  %(prog)s --input_dir ./videos --out_dir ./frames --start_time 10.0 --interval 500
        """,
    )

    parser.add_argument(
        "--input_dir",
        "-i",
        required=True,
        help="Input directory containing video files",
    )

    parser.add_argument(
        "--out_dir", "-o", required=True, help="Output directory for extracted images"
    )

    parser.add_argument(
        "--out_image_format",
        choices=["jpg", "png", "tif", "pdf"],
        default="jpg",
        help="Output image format (default: jpg)",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of threads for parallel processing (default: 8)",
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum frames to extract per video (default: extract all)",
    )

    parser.add_argument(
        "--start_time",
        type=float,
        default=0.0,
        help="Start time for extraction in seconds (default: 0)",
    )

    parser.add_argument(
        "--end_time",
        type=float,
        default=None,
        help="End time for extraction in seconds (default: video end)",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=1000,
        help="Extraction interval in milliseconds (default: 1000ms = 1 second)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress bar and non-essential output",
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip frames that already exist (resume capability)",
    )

    return parser.parse_args(args)


def main():
    """Main entry point."""
    setup_shutdown_handler()
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    if args.quiet:
        logging.basicConfig(level=logging.ERROR)

    try:
        extractor = VideoFrameExtractor(
            input_dir=args.input_dir,
            output_dir=args.out_dir,
            interval_ms=args.interval,
            image_format=args.out_image_format,
            max_frames=args.max_frames,
            threads=args.threads,
            start_time=args.start_time,
            end_time=args.end_time,
        )

        if not args.quiet:
            print(f"\n{'=' * 60}")
            print("Video Frame Extraction")
            print(f"{'=' * 60}")
            print(f"Input directory: {args.input_dir}")
            print(f"Output directory: {args.out_dir}")
            print(f"Extraction interval: {args.interval}ms")
            print(
                f"Time range: {args.start_time}s - {args.end_time if args.end_time else 'video end'}"
            )
            print(f"Output format: {args.out_image_format}")
            print(f"Threads: {args.threads}")
            print(f"Max frames per video: {args.max_frames or 'all'}")
            print(f"{'=' * 60}\n")

        stats = extractor.extract_all(show_progress=not args.quiet)

        if not args.quiet:
            print(f"\n{'=' * 60}")
            print("Extraction Statistics")
            print(f"{'=' * 60}")
            print(f"Total videos processed: {stats['total_videos']}")
            print(f"Total frames extracted: {stats['total_frames']}")
            if stats.get("skipped_frames", 0) > 0:
                print(f"Skipped existing frames: {stats['skipped_frames']}")
            print(f"Errors: {stats['errors']}")
            print(f"Processing time: {stats['processing_time']:.2f} seconds")
            print(f"{'=' * 60}\n")

        if extractor.errors:
            logger.warning(f"Encountered {len(extractor.errors)} error(s):")
            for video_path, error in extractor.errors:
                logger.error(f"  {video_path.name}: {error}")

        if extractor.errors:
            sys.exit(1)
        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
