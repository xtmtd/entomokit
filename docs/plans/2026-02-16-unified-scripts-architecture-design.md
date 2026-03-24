# Unified Scripts Architecture Design

**Date:** 2026-02-16 (Updated 2026-02-18 with synthesize progress bars) (Updated 2026-02-18 with COCO annotations)  
**Status:** ✅ Completed  
**Last Updated:** 2026-02-18

## Overview

This document describes the unified architecture for organizing the insect image processing pipeline scripts. The goal is to maintain script independence while sharing common utilities and dependencies.

## Architecture Principles

### 1. Script Independence
Each script remains fully independent and can be:
- Run standalone without installing other scripts' dependencies
- Imported as a module in other code
- Deployed individually

### 2. Shared Common Library
Common utilities are extracted to `src/common/`:
- `cli.py` - Command-line infrastructure (shutdown handler, logging, argument parsing)
- `validators.py` - Directory/file validation, image/video detection
- `logging.py` - Standardized logging configuration

### 3. Domain-Driven Organization
Each script category has its own domain module:
- `src/segmentation/` - Insect segmentation (SAM3, Otsu, GrabCut)
- `src/framing/` - Video frame extraction
- `src/cleaning/` - Image cleaning and deduplication
- `src/splitting/` - Dataset splitting

### 4. Flexible Dependencies
Dependencies are managed via extras in `setup.py`:
```bash
pip install .[segmentation]    # For segment.py, synthesize.py
pip install .[cleaning]        # For clean_figs.py
pip install .[video]           # For extract_frames.py
pip install .[data]            # For split_dataset.py
pip install .[dev]             # Development dependencies
```

## Directory Structure

```
imagekit/
├── scripts/              # CLI entry points (thin wrappers)
│   ├── __init__.py
│   ├── segment.py
│   ├── synthesize.py
│   ├── clean_figs.py
│   ├── extract_frames.py
│   └── split_dataset.py
├── src/
│   ├── __init__.py
│   ├── common/          # Shared utilities
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── validators.py
│   │   └── logging.py
│   ├── segmentation/    # Segmentation domain
│   │   ├── __init__.py
│   │   └── processor.py
│   ├── framing/         # Video framing domain
│   │   ├── __init__.py
│   │   └── extractor.py
│   ├── cleaning/        # Image cleaning domain
│   │   ├── __init__.py
│   │   └── processor.py
│   └── splitting/       # Dataset splitting domain
│       ├── __init__.py
│       └── splitter.py
├── docs/
│   └── plans/
├── old_scripts/         # Original scripts (kept for reference)
├── tests/
├── data/
├── models/
├── outputs/
├── requirements.txt     # Base dependencies only
└── setup.py             # With extras_require
```

## Script Structure

Each script follows this pattern (thin wrapper ~30-50 lines):

```python
#!/usr/bin/env python3
"""Script description."""

import argparse
from src.common.cli import setup_shutdown_handler, get_shutdown_flag, setup_logging, save_log
from src.{domain}.{module} import {ProcessorClass}


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='...')
    # ... argument definitions
    return parser.parse_args(args)


def main():
    """Main entry point."""
    setup_shutdown_handler()
    args = parse_args()
    
    processor = {ProcessorClass}(...)
    results = processor.process(...)
    
    save_log(output_dir, args)
    logger.info(f"Processing complete: {results}")


if __name__ == '__main__':
    main()
```

## Usage

### Direct Script Execution
```bash
python scripts/segment.py --input_dir images/ --out_dir outputs/
python scripts/clean_figs.py --input_dir images/ --out_dir cleaned/
python scripts/extract_frames.py --input_dir videos/ --out_dir frames/
python scripts/split_dataset.py --raw_image_csv data.csv --mode ratio
```

### Installed Package
```bash
pip install -e .
entomokit-segment --input_dir images/ --out_dir outputs/
entomokit-clean --input_dir images/ --out_dir cleaned/
entomokit-extract --input_dir videos/ --out_dir frames/
entomokit-split --raw_image_csv data.csv --mode ratio
```

### Selective Installation
```bash
# Only segmentation (no video, no data dependencies)
pip install .[segmentation]

# Only cleaning
pip install .[cleaning]

# Only video extraction
pip install .[video]

# Only data splitting
pip install .[data]

# Development (all)
pip install .[dev]
```

## Dependencies

### Core (always required)
- numpy>=1.24.0
- Pillow>=10.0.0
- tqdm>=4.65.0

### Extras
- **segmentation**: torch, torchvision, opencv-python
- **cleaning**: imagehash, lama-contrasted (optional for LaMa strategy)
- **video**: opencv-python
- **data**: pandas

## Benefits

1. **Independent Maintenance**: Each script can be modified without affecting others
2. **Flexible Installation**: Users only install what they need
3. **Code Reuse**: Common utilities centralized in `src/common/`
4. **Domain Organization**: Related functionality grouped logically
5. **Easy Testing**: Each domain can be tested independently

## Migration Status

### ✅ Completed (2026-02-16)

1. **Directory structure created**
    - ✅ `scripts/` directory with CLI entry points
    - ✅ `src/common/` shared utilities (cli.py, validators.py, logging.py)
    - ✅ Domain modules (segmentation/, framing/, cleaning/, splitting/)

 2. **Script migrations completed**
    - ✅ `scripts/segment.py` - Updated with common CLI infrastructure
    - ✅ `scripts/extract_frames.py` - Updated with common CLI infrastructure
    - ✅ `scripts/clean_figs.py` - Updated with common CLI infrastructure
    - ✅ `scripts/split_dataset.py` - Updated with common CLI infrastructure
    - ✅ `scripts/synthesize.py` - Added with progress bars and black region avoidance (2026-02-18)
    - ✅ `scripts/synthesize.py` - Added COCO annotations generation (2026-02-18)

3. **Documentation updated**
    - ✅ README.md - Added all scripts with usage examples
    - ✅ requirements.txt - Updated with all dependencies
    - ✅ setup.py - Entry points configured for all scripts

 4. **COCO annotations support** (2026-02-18)
    - ✅ Added --coco-output flag to synthesize.py
    - ✅ Added --coco-output-dir parameter for custom output directory
    - ✅ COCO annotations include images, annotations, categories
    - ✅ Annotations contain: bbox, segmentation, area, mask_area, scale_ratio, rotation_angle, original paths
    - ✅ Extended COCOMetadataManager for synthesis-specific metadata
    - ✅ Works with both multiprocessing and sequential processing

 5. **Repair strategies added** (2026-02-16)
    - ✅ `black-mask`: Pure black [0,0,0] fill for future compositing
    - ✅ `LaMa`: WACV 2022 Fourier-based inpainting for high-quality results

### 🔄 Ongoing / Future

- Integration tests for pipeline chaining
- Configuration files for complex workflows
- Docker support with optional dependencies
- Additional validation tests

## Future Enhancements

Possible future improvements:
- Add integration tests for pipeline chaining
- Add configuration files for complex workflows
- Add CLI command chaining (e.g., `entomokit-pipeline segment then clean`)
- Add Docker support with optional dependencies
