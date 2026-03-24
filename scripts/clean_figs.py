#!/usr/bin/env python3
"""
Image cleaning and deduplication script.

Usage:
    python scripts/clean_figs.py --input_dir images/raw/ --out_dir images/cleaned/
"""

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.cli import setup_shutdown_handler, get_shutdown_flag, save_log
from src.cleaning.processor import ImageCleaner


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean and deduplicate images"
    )
    
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing input images.")
    parser.add_argument("--out_dir", type=str, required=True, help="Folder containing output images.")

    parser.add_argument("--out_short_size", type=int, default=512, 
                       help="Shorter size of output images. Use -1 to keep original size. Default: 512")
    parser.add_argument("--out_image_format", type=str, default="jpg", 
                       choices=["jpg", "png", "tif", "pdf"], help="Output image format. Default: jpg")

    parser.add_argument("--threads", type=int, default=12, help="Number of threads to use. Default: 12")

    parser.add_argument("--keep_exif", action="store_true", help="Keep EXIF data in output images.")

    parser.add_argument("--dedup_mode", type=str, default="md5", 
                       choices=["none", "md5", "phash"], help="Deduplication mode. Default: md5")
    parser.add_argument("--phash_threshold", type=int, default=5, help="Phash threshold. Default: 5")
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    parser.add_argument("--log_path", type=str, default=None, 
                       help="Path to log file. Default: 'log.txt' in output directory")

    return parser.parse_args(args)


def main():
    """Main entry point."""
    setup_shutdown_handler()
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_subdir = output_dir / "cleaned_images"
    images_subdir.mkdir(parents=True, exist_ok=True)
    
    log_path = args.log_path if args.log_path else str(output_dir / "log.txt")
    
    save_log(output_dir, args, log_filename="log.txt")
    
    try:
        cleaner = ImageCleaner(
            input_dir=args.input_dir,
            output_dir=str(images_subdir),
            out_short_size=args.out_short_size,
            out_image_format=args.out_image_format,
            dedup_mode=args.dedup_mode,
            phash_threshold=args.phash_threshold,
            threads=args.threads,
            keep_exif=args.keep_exif
        )
        
        results = cleaner.process_directory(log_path=log_path)
        
        print(f"Done. Processed {results['processed']} images, {results['errors']} errors.")
        print(f"Cleaned images saved to: {images_subdir}")
        print(f"Log saved to: {log_path}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
