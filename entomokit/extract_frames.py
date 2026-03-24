"""entomokit extract-frames — extract frames from video files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "extract-frames",
        help="Extract frames from video files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir",
        "-i",
        required=True,
        help="Input directory containing video files, OR a single video file path.",
    )
    p.add_argument(
        "--out-dir", "-o", required=True, help="Output directory for frames."
    )
    p.add_argument(
        "--out-image-format",
        choices=["jpg", "png", "tif"],
        default="jpg",
        help="Image format for extracted frames.",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of worker threads for frame extraction.",
    )
    p.add_argument(
        "--max-frames", type=int, default=None, help="Max frames to extract per video."
    )
    p.add_argument(
        "--start-time", type=float, default=0.0, help="Start time in seconds."
    )
    p.add_argument("--end-time", type=float, default=None, help="End time in seconds.")
    p.add_argument(
        "--interval",
        type=int,
        default=1000,
        help="Extraction interval in milliseconds.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip frames that already exist (resume).",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging output.",
    )
    p.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output and progress bars.",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    import logging
    from pathlib import Path
    from src.common.cli import setup_shutdown_handler, save_log
    from src.framing.extractor import VideoFrameExtractor

    setup_shutdown_handler()

    input_path = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_log(out_dir, args)

    # Accept single video file OR directory
    if input_path.is_file():
        actual_input_dir = input_path.parent
        single_file = input_path
    elif input_path.is_dir():
        actual_input_dir = input_path
        single_file = None
    else:
        print(f"Error: --input-dir does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    logging.basicConfig(
        level=logging.DEBUG
        if args.verbose
        else (logging.ERROR if args.quiet else logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    extractor = VideoFrameExtractor(
        input_dir=str(actual_input_dir),
        output_dir=str(out_dir),
        interval_ms=args.interval,
        image_format=args.out_image_format,
        max_frames=args.max_frames,
        threads=args.threads,
        start_time=args.start_time,
        end_time=args.end_time,
    )

    # Single file mode: set filter attribute for get_video_files
    if single_file is not None:
        extractor._single_file_filter = single_file.name

    stats = extractor.extract_all(show_progress=not args.quiet)

    if not args.quiet:
        print(f"Total videos processed: {stats['total_videos']}")
        print(f"Total frames extracted: {stats['total_frames']}")
        print(f"Errors: {stats['errors']}")
