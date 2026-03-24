#!/usr/bin/env python3
"""
Segmentation script for insect extraction.

Usage:
    # SAM3 with alpha channel
    python scripts/segment.py \\
        --input_dir images/clean_insects/ \\
        --out_dir outputs/insects_clean/ \\
        --sam3-checkpoint models/sam3_hq_vit_h.pt \\
        --segmentation-method sam3 \\
        --device auto \\
        --hint "insect" \\
        --threads 12
    
    # SAM3 with bounding box only
    python scripts/segment.py \\
        --input_dir images/clean_insects/ \\
        --out_dir outputs/insects_bbox/ \\
        --sam3-checkpoint models/sam3_hq_vit_h.pt \\
        --segmentation-method sam3-bbox \\
        --out-image-format jpg
    
    # Otsu method (no SAM3 checkpoint required)
    python scripts/segment.py \\
        --input_dir images/clean_insects/ \\
        --out_dir outputs/insects_otsu/ \\
        --segmentation-method otsu \\
        --out-image-format jpg
    
    # GrabCut method (no SAM3 checkpoint required)
    python scripts/segment.py \\
        --input_dir images/clean_insects/ \\
        --out_dir outputs/insects_grabcut/ \\
        --repair-strategy sam3-fill
"""

import argparse
import logging
import os
import signal
import sys
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
# Use the script's directory to find the project root
SCRIPTS_DIR = Path(__file__).resolve().parent
# Project root is one level up from scripts directory  
PROJECT_ROOT = SCRIPTS_DIR.parent

# The project root may already be in sys.path from .pth files
# but we need to ensure it's at the FRONT to take precedence over site-packages
if sys.path[0] != str(PROJECT_ROOT):
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.cli import setup_shutdown_handler, get_shutdown_flag, setup_logging, save_log
from src.segmentation.processor import SegmentationProcessor


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Segment insects from images using multiple methods',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--hint', '-t',
        default='insect',
        help='Text prompt for segmentation (default: "insect")'
    )
    
    parser.add_argument(
        '--input_dir', '-i',
        required=True,
        help='Input directory containing images'
    )
    
    parser.add_argument(
        '--out_dir', '-o',
        required=True,
        help='Output directory for segmented images'
    )
    
    parser.add_argument(
        '--segmentation-method',
        default='sam3',
        choices=['sam3', 'sam3-bbox', 'otsu', 'grabcut'],
        help='Segmentation method'
    )
    
    parser.add_argument(
        '--sam3-checkpoint', '-c',
        required=False,
        help='Path to SAM3 checkpoint file (required for sam3/sam3-bbox methods)'
    )
    
    parser.add_argument(
        '--device', '-d',
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device for inference (default: auto)'
    )
    
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.0,
        help='Minimum confidence score for masks (default: 0.0 = no filtering)'
    )
    
    parser.add_argument(
        '--repair-strategy', '-r',
        default=None,
        choices=['opencv', 'sam3-fill', 'black-mask', 'lama'],
        help='Repair strategy for filling holes (default: None)'
    )
    
    parser.add_argument(
        '--padding-ratio',
        type=float,
        default=0.0,
        help='Padding ratio for bounding box (default: 0.0 = no padding)'
    )
    
    parser.add_argument(
        '--lama-model',
        default=None,
        help='Path to LaMa model checkpoint directory (default: models/big-lama/models/best.ckpt)'
    )
    parser.add_argument(
        '--lama-mask-dilate',
        type=int,
        default=0,
        help='Number of dilation iterations for LaMa mask (default: 0)'
    )
    
    parser.add_argument(
        '--out-image-format', '-f',
        default='png',
        choices=['png', 'jpg'],
        help='Output image format (default: png)'
    )
    
    parser.add_argument(
        '--threads', '-n',
        type=int,
        default=8,
        help='Number of parallel workers (default: 8)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--annotation-output-format',
        default='coco',
        choices=['coco', 'voc', 'yolo'],
        help='Output format for annotations: coco (default), voc, yolo'
    )
    
    parser.add_argument(
        '--coco-output-mode',
        default='unified',
        choices=['unified', 'separate'],
        help='COCO output mode: unified (single annotations.json) or separate (per-image JSON files)'
    )
    
    return parser.parse_args(args)


def main():
    """Main entry point."""
    setup_shutdown_handler()
    args = parse_args()
    lama_logger = logging.getLogger('imagekit.lama')
    lama_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    max_threads = os.cpu_count() or 8
    if args.threads > max_threads * 2:
        print(f"Warning: Thread count {args.threads} exceeds recommended maximum {max_threads * 2}", file=sys.stderr)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    has_images = any(
        f.suffix.lower() in image_extensions 
        for f in input_dir.iterdir() 
        if f.is_file()
    )
    if not has_images:
        print(f"Warning: No image files found in input directory: {input_dir}", file=sys.stderr)
    
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_log(output_dir, args)
    logger = setup_logging(output_dir, args.verbose)
    
    logger.info("Starting segmentation process")
    try:
        processor = SegmentationProcessor(
            sam3_checkpoint=args.sam3_checkpoint,
            device=args.device,
            hint=args.hint,
            repair_strategy=args.repair_strategy,
            confidence_threshold=args.confidence_threshold,
            padding_ratio=args.padding_ratio,
            segmentation_method=args.segmentation_method,
            lama_model=args.lama_model,
            lama_mask_dilate=max(0, args.lama_mask_dilate),
            annotation_format=args.annotation_output_format,
            coco_output_mode=args.coco_output_mode
        )
    except Exception:
        logger.exception("Failed to initialize processor")
        sys.exit(1)
    
    try:
        results = processor.process_directory(
            input_dir=args.input_dir,
            output_dir=output_dir,
            num_workers=args.threads,
            disable_tqdm=False,
            output_format=args.out_image_format,
            shutdown_flag=get_shutdown_flag()
        )
        
        logger.info(f"Processing complete!")
        logger.info(f"  Successfully processed: {results['processed']}")
        logger.info(f"  Failed: {results['failed']}")
        logger.info(f"  Output files: {len(results['output_files'])}")
        
    except Exception:
        logger.exception("Processing failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
