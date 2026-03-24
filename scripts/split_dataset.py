#!/usr/bin/env python3
"""
Dataset splitting script.

Usage:
    python scripts/split_dataset.py --raw_image_csv data/images.csv --mode ratio --out_dir datasets/
"""

import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.cli import setup_shutdown_handler, save_log
from src.splitting.splitter import DatasetSplitter


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Dataset Splitting Script: ratio-split or count-split mode."
    )
    
    parser.add_argument('--raw_image_csv', type=str, required=True, 
                       help='Path to the raw CSV file containing image and label columns.')
    
    parser.add_argument('--mode', type=str, choices=['ratio', 'count'], default='ratio', 
                       help='Split mode: ratio or count.')
    
    parser.add_argument('--unknown_test_classes_ratio', type=float, default=0, 
                       help='Ratio of total samples for Open-set unknown classes. Default: 0')
    parser.add_argument('--known_test_classes_ratio', type=float, default=0.1, 
                       help='Ratio of known class samples for Closed-set test. Default: 0.1')
    
    parser.add_argument('--unknown_test_classes_count', type=int, default=0, 
                       help='Number of samples for Open-set unknown classes. Default: 0')
    parser.add_argument('--known_test_classes_count', type=int, default=0, 
                       help='Number of samples for Closed-set known classes. Default: 0')
    parser.add_argument('--min_count_per_class', type=int, default=0, 
                       help='Minimum samples per class for train set.')
    parser.add_argument('--max_count_per_class', type=int, default=None,
                       help='Maximum samples per class for train set.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--out_dir', type=str, default='datasets', 
                       help='Output directory. Default: datasets')
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

    return parser.parse_args(args)


def main():
    """Main entry point."""
    setup_shutdown_handler()
    args = parse_args()
    
    csv_path = Path(args.raw_image_csv)
    if not csv_path.exists():
        print(f"Error: CSV file does not exist: {csv_path}", file=sys.stderr)
        sys.exit(1)
    
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_log(output_dir, args, log_filename="log.txt")
    
    try:
        splitter = DatasetSplitter(
            raw_image_csv=args.raw_image_csv,
            out_dir=args.out_dir,
            seed=args.seed
        )
        
        results = splitter.split(
            mode=args.mode,
            unknown_test_ratio=args.unknown_test_classes_ratio,
            known_test_ratio=args.known_test_classes_ratio,
            unknown_test_count=args.unknown_test_classes_count,
            known_test_count=args.known_test_classes_count,
            min_count_per_class=args.min_count_per_class,
            max_count_per_class=args.max_count_per_class
        )
        
        total = results['test_unknown'] + results['test_known'] + results['train']
        print(f"All outputs saved in {output_dir}")
        print(f"Total samples: {total}")
        print(f"  Train: {results['train']}")
        print(f"  Test (known): {results['test_known']}")
        print(f"  Test (unknown): {results['test_unknown']}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
