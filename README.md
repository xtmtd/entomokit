# Insect Dataset Toolkit

A Python-based toolkit for building insect image datasets with multiple processing capabilities including segmentation, frame extraction, image cleaning, dataset splitting, and image synthesis.

## Overview

This project provides five main scripts:
- `segment.py` - Segments insects from clean background images using multiple segmentation methods
- `extract_frames.py` - Extracts frames from video files for data collection
- `clean_figs.py` - Cleans and deduplicates images with consistent naming
- `split_dataset.py` - Splits datasets into train/test/unknown classes for ML training
- `synthesize.py` - Composites target objects onto background images with advanced positioning

## Features

- **Multiple Segmentation Methods**: SAM3 (with alpha channel), SAM3-bbox (cropped), Otsu thresholding, GrabCut
- **Flexible Repair Strategies**: OpenCV morphological operations or SAM3-based hole filling
- **Video Frame Extraction**: Multithreaded extraction from multiple video formats with progress tracking
- **Image Cleaning**: Resize, deduplicate (MD5/Phash), and standardize image naming
- **Dataset Splitting**: Ratio or count-based train/test/unknown class splits with stratification
- **Image Synthesis**: Advanced compositing with rotation, color matching, and black region avoidance
- **Input Validation**: Validates input directories, image files, and parameter constraints
- **Graceful Shutdown**: Handles Ctrl+C to finish current image before exiting
- **Parallel Processing**: Multi-threaded image processing with configurable worker count
- **Progress Tracking**: Progress bars for synthesis with tqdm support
- **Comprehensive Logging**: Detailed logging with verbose mode and log file output

## Installation

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## Project Structure

```
.
├── scripts/              # CLI scripts
│   ├── segment.py        # Segmentation script
│   ├── extract_frames.py # Video frame extraction
│   ├── clean_figs.py     # Image cleaning and deduplication
│   ├── split_dataset.py  # Dataset splitting
│   └── synthesize.py     # Image synthesis script
├── src/
│   ├── common/          # Shared utilities (CLI, logging, validation)
│   ├── segmentation/    # Segmentation domain logic
│   ├── framing/         # Video framing domain logic
│   ├── cleaning/        # Image cleaning domain logic
│   ├── splitting/       # Dataset splitting domain logic
│   └── synthesis/       # Image synthesis domain logic
│   └── lama/            # LaMa inpainting implementation
├── tests/               # Test files
├── data/                # Data directory (large files ignored)
├── models/              # Model weights (large files ignored)
├── outputs/             # Output files (ignored)
├── requirements.txt     # Python dependencies
├── setup.py             # Package setup
├── CHANGES_SUMMARY.md   # This change summary
└── README.md            # This file
```

## Usage

### 1. Segment Script

Segment insects from images using multiple methods (SAM3, Otsu, GrabCut). Optionally generates annotations in COCO, VOC, or YOLO format.

**Annotations**: When `--annotation-output-format` is specified, generates annotation files with object metadata (bbox, segmentation, area) in original image coordinates.

#### Basic Usage

```bash
# SAM3 with alpha channel (transparent background)
python scripts/segment.py     --input_dir images/clean_insects/     --out_dir outputs/insects_clean/     --sam3-checkpoint models/sam3_hq_vit_h.pt     --segmentation-method sam3     --device auto     --hint "insect"

# With COCO annotations (unified, default)
python scripts/segment.py     --input_dir images/clean_insects/     --out_dir outputs/insects_clean/     --sam3-checkpoint models/sam3_hq_vit_h.pt     --segmentation-method sam3     --annotation-output-format coco     --coco-output-mode unified

# With VOC Pascal annotations
python scripts/segment.py     --input_dir images/clean_insects/     --out_dir outputs/insects_clean/     --sam3-checkpoint models/sam3_hq_vit_h.pt     --segmentation-method sam3     --annotation-output-format voc

# With YOLO annotations
python scripts/segment.py     --input_dir images/clean_insects/     --out_dir outputs/insects_clean/     --sam3-checkpoint models/sam3_hq_vit_h.pt     --segmentation-method sam3     --annotation-output-format yolo
```

#### All Parameters

**Required Parameters:**
- `--input_dir`, `-i`: Input directory containing images
- `--out_dir`, `-o`: Output directory for segmented images

**Optional Parameters:**
- `--hint`, `-t`: Text prompt for segmentation (default: "insect")
- `--segmentation-method`: Segmentation method
  - `sam3` - SAM3 segmentation with alpha channel (transparent background)
  - `sam3-bbox` - SAM3 segmentation with cropped bounding box (no alpha)
  - `otsu` - Otsu thresholding (no SAM3 checkpoint required)
  - `grabcut` - GrabCut algorithm (no SAM3 checkpoint required)
- `--sam3-checkpoint`, `-c`: Path to SAM3 checkpoint file (required for sam3/sam3-bbox methods)
- `--device`, `-d`: Device for inference (`auto`, `cpu`, `cuda`, `mps`)
- `--confidence-threshold`: Minimum confidence score for masks (0.0-1.0, default: 0.0)
- `--repair-strategy`, `-r`: Repair strategy for filling holes
  - `opencv` - OpenCV morphological operations
  - `sam3-fill` - SAM3-based hole filling
  - `lama` - LaMa-based hole filling
- `--padding-ratio`: Padding ratio for bounding box (0.0-0.5, default: 0.0)
- `--out-image-format`, `-f`: Output image format (`png`, `jpg`)
- `--threads`, `-n`: Number of parallel workers (default: 8)
- `--verbose`, `-v`: Enable verbose logging
- `--annotation-output-format`: Annotation output format (`coco`, `voc`, `yolo`)
- `--coco-output-mode`: COCO output mode (`unified`, `separate`)
  - `unified` - Single `annotations.json` file
  - `separate` - Individual JSON files per image

#### Usage Examples

**1. SAM3 with Alpha Channel (Transparent Background)**

```bash
python scripts/segment.py     --input_dir images/clean_insects/     --out_dir outputs/insects_clean/     --sam3-checkpoint models/sam3_hq_vit_h.pt     --segmentation-method sam3     --device auto     --hint "insect"     --threads 12
```

**2. SAM3 with Bounding Box Only (No Alpha)**

```bash
python scripts/segment.py     --input_dir images/clean_insects/     --out_dir outputs/insects_bbox/     --sam3-checkpoint models/sam3_hq_vit_h.pt     --segmentation-method sam3-bbox     --out-image-format jpg
```

**3. Otsu Thresholding (No SAM3 Checkpoint Required)**

```bash
python scripts/segment.py     --input_dir images/clean_insects/     --out_dir outputs/insects_otsu/     --segmentation-method otsu     --out-image-format jpg
```

**4. GrabCut Algorithm (No SAM3 Checkpoint Required)**

```bash
python scripts/segment.py     --input_dir images/clean_insects/     --out_dir outputs/insects_grabcut/     --segmentation-method grabcut     --repair-strategy sam3-fill
```

**5. With Repair Strategy and Padding**

```bash
python scripts/segment.py     --input_dir images/clean_insects/     --out_dir outputs/insects_repaired/     --sam3-checkpoint models/sam3_hq_vit_h.pt     --segmentation-method sam3     --repair-strategy opencv     --padding-ratio 0.1     --confidence-threshold 0.5     --verbose
```

**6. With COCO Annotations (Unified Mode)**

```bash
python scripts/segment.py     --input_dir images/clean_insects/     --out_dir outputs/insects_clean/     --sam3-checkpoint models/sam3_hq_vit_h.pt     --segmentation-method sam3     --annotation-output-format coco     --coco-output-mode unified
```

**Output Structure with Annotations:**
```
output_dir/
├── cleaned_images/           # Segmented output images
│   ├── image_01.png
│   └── image_02.png
├── annotations/              # COCO/VOC annotations
│   └── annotations.json      # (unified COCO mode - single file for all images)
│   ├── image_01.json         # (separate COCO mode - one JSON per input image)
│   └── image_01.xml          # (VOC/YOLO - one annotation file per input image)
└── labels/                   # YOLO annotations
    ├── image_01.txt
    └── ...
```

**Annotation Format Notes:**
- **bbox**: `[x, y, w, h]` in original image coordinates
- **segmentation**: `[[x1, y1, x2, y2, ...]]` in original image coordinates
- **area**: Bounding box area (width × height)
- **mask_area**: Actual mask pixel count
- **file_name**: Points to cleaned output image (e.g., `cleaned_images/image.png`)
- **VOC/YOLO**: One annotation file per input image (all objects from that image in one file)

### 2. Extract Frames Script

Extract frames from video files with multithreading support.

#### Basic Usage

```bash
# Extract frames every 1 second (default)
python scripts/extract_frames.py     --input_dir videos/     --out_dir frames/

# Custom interval (every 500ms)
python scripts/extract_frames.py     --input_dir videos/     --out_dir frames/     --interval 500

# Limit frames per video
python scripts/extract_frames.py     --input_dir videos/     --out_dir frames/     --max_frames 100

# Custom output format (PNG)
python scripts/extract_frames.py     --input_dir videos/     --out_dir frames/     --out_image_format png

# Adjust thread count
python scripts/extract_frames.py \
    --input_dir videos/ \
    --out_dir frames/ \
    --threads 4

# Extract frames from specific time range (5s to 30s)
python scripts/extract_frames.py \
    --input_dir videos/ \
    --out_dir frames/ \
    --start_time 5.0 \
    --end_time 30.0

# Extract frames starting from 10 seconds to end
python scripts/extract_frames.py \
    --input_dir videos/ \
    --out_dir frames/ \
    --start_time 10.0
```

**Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input_dir` | Input directory containing video files | Required |
| `--out_dir` | Output directory for extracted images | Required |
| `--interval` | Extraction interval in milliseconds | 1000 (1s) |
| `--start_time` | Start time for extraction in seconds | 0 |
| `--end_time` | End time for extraction in seconds | video end |
| `--out_image_format` | Output format (jpg/png/tif/pdf) | jpg |
| `--threads` | Number of parallel threads | 8 |
| `--max_frames` | Maximum frames per video | All |

**Output Structure:**

```
output_dir/
└── video_name/
    ├── video_name_01.jpg
    ├── video_name_02.jpg
    └── ...
```

### 3. Clean Figures Script

Clean and deduplicate images with consistent naming and format.

#### Basic Usage

```bash
# Basic cleaning with MD5 deduplication
python scripts/clean_figs.py     --input_dir images/raw/     --out_dir images/cleaned/

# Resize to shorter side 512px
python scripts/clean_figs.py     --input_dir images/raw/     --out_dir images/cleaned/     --out_short_size 512

# Use phash for perceptual deduplication
python scripts/clean_figs.py     --input_dir images/raw/     --out_dir images/cleaned/     --dedup_mode phash     --phash_threshold 5

# Convert to PNG format
python scripts/clean_figs.py     --input_dir images/raw/     --out_dir images/cleaned/     --out_image_format png     --threads 16
```

**Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input_dir` | Input directory containing images | Required |
| `--out_dir` | Output directory for cleaned images | Required |
| `--out_short_size` | Shorter size of output images (-1 for original) | 512 |
| `--out_image_format` | Output format (jpg/png/tif/pdf) | jpg |
| `--threads` | Number of parallel threads | 12 |
| `--keep_exif` | Keep EXIF data in output | No |
| `--dedup_mode` | Deduplication mode (none/md5/phash) | md5 |
| `--phash_threshold` | Phash threshold for similarity | 5 |

### 4. Split Dataset Script

Split datasets into train/test/unknown classes for machine learning.

#### Basic Usage

```bash
# Ratio-based split (default)
python scripts/split_dataset.py     --raw_image_csv data/images.csv     --mode ratio     --out_dir datasets/

# With unknown test classes (open-set)
python scripts/split_dataset.py     --raw_image_csv data/images.csv     --mode ratio     --unknown_test_classes_ratio 0.1     --known_test_classes_ratio 0.1     --out_dir datasets/

# Count-based split
python scripts/split_dataset.py     --raw_image_csv data/images.csv     --mode count     --unknown_test_classes_count 50     --known_test_classes_count 100     --min_count_per_class 10     --max_count_per_class 100     --out_dir datasets/

# Custom random seed
python scripts/split_dataset.py     --raw_image_csv data/images.csv     --mode ratio     --seed 42     --out_dir datasets/
```

**Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--raw_image_csv` | Path to raw CSV with image and label columns | Required |
| `--mode` | Split mode: ratio or count | ratio |
| `--unknown_test_classes_ratio` | Ratio for unknown test classes | 0 |
| `--known_test_classes_ratio` | Ratio for known test classes | 0.1 |
| `--unknown_test_classes_count` | Count for unknown test classes | 0 |
| `--known_test_classes_count` | Count for known test classes | 0 |
| `--min_count_per_class` | Min samples per class for train | 0 |
| `--max_count_per_class` | Max samples per class for train | None |
| `--seed` | Random seed for reproducibility | 42 |
| `--out_dir` | Output directory | datasets |

**Output Structure:**

```
output_dir/
├── train.csv              # Training samples
├── test.known.csv         # Known class test samples
├── test.unknown.csv       # Unknown class test samples (if configured)
└── class_count/
    ├── class.count        # Total class statistics
    ├── class.train.count  # Training class counts
    ├── class.test.known.count
    └── class.test.unknown.count
```


### 5. Synthesize Script

Composite target objects onto background images with rotation, color matching, and intelligent positioning. Optionally generates annotations in COCO, VOC, or YOLO format with target object metadata.

**Annotations**: When `--annotation-output-format` is specified, generates annotation files with object metadata (bbox, segmentation, area, scale_ratio, rotation_angle) in synthesized image coordinates.

#### Basic Usage

```bash
# Basic synthesis with 10 variations per target
python scripts/synthesize.py     --target-dir images/targets/     --background-dir images/backgrounds/     --out-dir outputs/synthesized/     --num-syntheses 10

# With COCO annotations (unified, default)
python scripts/synthesize.py     --target-dir images/targets/     --background-dir images/backgrounds/     --out-dir outputs/synthesized/     --num-syntheses 10     --annotation-output-format coco     --coco-output-mode unified

# With COCO separate files per image
python scripts/synthesize.py     --target-dir images/targets/     --background-dir images/backgrounds/     --out-dir outputs/synthesized/     --num-syntheses 10     --annotation-output-format coco     --coco-output-mode separate

# With VOC Pascal annotations
python scripts/synthesize.py     --target-dir images/targets/     --background-dir images/backgrounds/     --out-dir outputs/synthesized/     --num-syntheses 10     --annotation-output-format voc

# With YOLO annotations
python scripts/synthesize.py     --target-dir images/targets/     --background-dir images/backgrounds/     --out-dir outputs/synthesized/     --num-syntheses 10     --annotation-output-format yolo

# With black region avoidance
python scripts/synthesize.py     --target-dir images/targets/     --background-dir images/backgrounds/     --out-dir outputs/synthesized/     --avoid-black-regions

# With rotation and parallel processing
python scripts/synthesize.py     --target-dir images/targets/     --background-dir images/backgrounds/     --out-dir outputs/synthesized/     --rotate 30     --threads 4
```

**Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--target-dir`, `-t` | Target object images directory (cleaned, with alpha channel) | Required |
| `--background-dir`, `-b` | Background images directory | Required |
| `--out-dir`, `-o` | Output directory | Required |
| `--num-syntheses`, `-n` | Number of syntheses per target image | 10 |
| `--area-ratio-min`, `-a` | Minimum area ratio (target/background area) | 0.05 |
| `--area-ratio-max`, `-x` | Maximum area ratio (target/background area) | 0.20 |
| `--color-match-strength`, `-c` | Color matching strength (0-1) | 0.5 |
| `--avoid-black-regions`, `-A` | Avoid pure black regions in background | No |
| `--rotate`, `-r` | Maximum random rotation degrees | 0 |
| `--out-image-format`, `-f` | Output image format (png/jpg) | png |
| `--threads`, `-d` | Number of parallel workers | 4 |
| `--verbose`, `-v` | Enable verbose logging | No |
| `--annotation-output-format` | Annotation output format (`coco`, `voc`, `yolo`) | coco |
| `--coco-output-mode` | COCO output mode (`unified`, `separate`) | unified |

**Features:**
- **Rotation Support**: Random rotation within specified degrees
- **Black Region Avoidance**: Automatically positions targets away from dark areas
- **Color Matching**: Matches target color to background using LAB histogram matching
- **Auto Scaling**: Automatically scales targets to fit within background
- **Parallel Processing**: Multi-threaded synthesis with progress bar
- **Multiple Annotation Formats**: COCO JSON, VOC XML, YOLO TXT

**Output Structure:**

Without annotations:
```
output_dir/
└── images/
    ├── target_name_01.png
    ├── target_name_02.png
    └── ...
```

With COCO annotations (`--annotation-output-format coco`):
```
output_dir/
├── images/
│   ├── target_name_01.png
│   ├── target_name_02.png
│   └── ...
├── annotations/              # COCO/VOC annotations
│   └── annotations.json      # (unified COCO mode - single file for all images)
│   ├── target_name_01.json   # (separate COCO mode - one JSON per input image)
│   └── target_name_01.xml    # (VOC/YOLO - one annotation file per input image)
└── labels/                   # YOLO annotations
    ├── target_name_01.txt
    └── ...
```

**Annotation Format Notes:**
- **bbox**: `[x, y, w, h]` in synthesized image coordinates
- **segmentation**: `[[x1, y1, x2, y2, ...]]` in synthesized image coordinates
- **area**: Bounding box area (width × height)
- **mask_area**: Actual mask pixel count
- **scale_ratio**: Target/background area ratio used
- **rotation_angle**: Rotation angle in degrees
- **file_name**: Points to synthesized image (e.g., `images/target_name_01.png`)
- **VOC/YOLO**: One annotation file per input image (all objects from that image in one file)

## Input Validation and Error Handling

All scripts include comprehensive input validation:

**Common Validations:**
- **Directory Validation**: Checks if input directory exists and contains expected files
- **Parameter Validation**: Validates numeric ranges, format choices, and required arguments
- **File Format Validation**: Supports standard image and video formats

**Script-Specific Validations:**
- **segment.py**: Validates SAM3 checkpoint existence (when required), image format compatibility
- **clean_figs.py**: Validates image files can be opened, handles corrupted files gracefully
- **extract_frames.py**: Validates video files can be opened with OpenCV
- **split_dataset.py**: Validates CSV structure (image and label columns exist)
- **synthesize.py**: Validates target images (with alpha channel), background images, and output directory; validates COCO output directory creation

**Error Handling:**
- Detailed error messages with logging
- Graceful shutdown on Ctrl+C
- Partial results are saved on interruption

## Graceful Shutdown

Press `Ctrl+C` at any time to trigger a graceful shutdown:
- Current image processing completes
- Results are saved
- Clean exit with status message

## Progress Tracking

All scripts support progress tracking:
- **Synthesize script**: Shows progress bar for synthesis operations (requires `tqdm`)
- Progress bars automatically disabled when running in headless environments
- Can be explicitly disabled with script-specific flags where applicable

## Logging

All scripts save logs to `log.txt` in the output directory:
- Command used and timestamp
- All parameter values
- Processing progress and results
- Errors and warnings

Enable verbose mode with `--verbose` for detailed debugging information.

## Model Requirements

### SAM3 Model

For SAM3-based methods (`sam3`, `sam3-bbox`), you need:
- SAM3 checkpoint file (e.g., `sam3_hq_vit_h.pt` or `sam3.pt`)
- Download from [SAM3 on Hugging Face](https://huggingface.co/facebook/sam3)

### LaMa Model

For LaMa-based inpainting repair (`--repair-strategy lama`), you need:
- LaMa Big-Lama model checkpoint (e.g., `models/big-lama/models/best.ckpt`)
- Configuration file (e.g., `models/big-lama/config.yaml`)

#### Download LaMa Model

The LaMa model can be downloaded from the official Google Drive folder:

**Option 1: Download from Google Drive (Recommended)**

1. Visit [LaMa Big-Lama Model on Google Drive](https://drive.google.com/drive/folders/1B2x7eQDgecTL0oh3LSIBDGj0fTxs6Ips?usp=sharing)
2. Download the **big-lama** model files
3. Extract and place the files in the correct locations:
   ```
   models/big-lama/
   ├── config.yaml          # LaMa configuration
   └── models/
       └── best.ckpt        # Model weights (large file, ~150MB)
   ```

**Option 2: Using the models directory**

The project includes a `.gitkeep` file in the `models/` directory to ensure it's tracked by git. The actual model files are large and should be downloaded separately.

**Option 3: Verify Download**

After downloading, verify the model structure:

```bash
ls -lh models/big-lama/config.yaml
ls -lh models/big-lama/models/best.ckpt
```

For `otsu` and `grabcut` methods, no external model is required.

## Documentation Updates

### 2026-02-18 - Multi-Format Annotation Output Support

**Documentation Updated:**
- README.md - Added annotation output documentation for both segment.py and synthesize.py
- README.cn.md - Added Chinese annotation output documentation
- CHANGES_SUMMARY.md - Added annotation format changes
- docs/plans/2026-02-18-annotation-fixes.md - Complete implementation plan with segment.py updates

**Annotation Output Features:**
- **Multiple Formats**: COCO JSON, VOC Pascal XML, YOLO TXT
- **COCO Modes**: Unified (single file) or Separate (per-image files)
- **Segment Script**: Annotations in original image coordinates
- **Synthesize Script**: Annotations in synthesized image coordinates
- **Output Structure**: Annotations folder is sibling to images folder
- **Polygon Support**: VOC and YOLO formats support polygon segmentation

**segment.py:**
- New arguments: `--annotation-output-format`, `--coco-output-mode`
- bbox/segmentation in original image coordinates
- file_name points to cleaned output image

**synthesize.py:**
- Updated arguments: `--annotation-output-format` replaces `--coco-output`
- bbox/segmentation in synthesized image coordinates
- Removed redundant source path information for portability

### 2026-02-16 - Unified Scripts Architecture

**Documentation Updated:**
- README.md - Added all 4 scripts with comprehensive usage examples
- requirements.txt - Verified all dependencies
- setup.py - Verified entry points and extras
- docs/plans/2026-02-16-unified-scripts-architecture-design.md - Status: Completed

**Scripts Documented:**
1. `segment.py` - Segmentation (SAM3, Otsu, GrabCut)
2. `extract_frames.py` - Video frame extraction
3. `clean_figs.py` - Image cleaning and deduplication
4. `split_dataset.py` - Train/test/unknown class splitting
5. `synthesize.py` - Image synthesis (rotation, color matching, black region avoidance, COCO annotations)

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
