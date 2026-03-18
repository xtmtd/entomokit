# Documentation Update Summary

**Date:** 2026-02-18 (Updated 2026-02-18 with COCO annotations, 2026-02-22 with VOC/YOLO annotation fix)  
**Updated Files:** README.md, requirements.txt, setup.py, docs/plans/2026-02-16-unified-scripts-architecture-design.md, scripts/synthesize.py, src/synthesis/processor.py, src/metadata.py, src/segmentation/processor.py, tests/test_annotation_formats.py

---

## Changes Made
### 5. scripts/synthesize.py

#### Progress Bar Support
- Added `tqdm` import with fallback for environments without tqdm
- Updated `_synthesize_single_wrapper()` method for multiprocessing
- Added progress bars in `process_directory()` for both single-threaded and multi-threaded modes
- Progress bars show "Synthesizing" description
- Can be disabled via `disable_tqdm=True` parameter

#### COCO Annotations Generation
- Added `--coco-output` flag (boolean, default: False) to enable COCO annotation generation
- Added `--coco-output-dir` argument (string, default: "annotations") to specify output directory
- Added `coco_output` and `coco_output_dir` parameters to `SynthesisProcessor.__init__()`
- Added `_add_synthesis_metadata()` method to track synthesis-specific metadata
- Added `_generate_coco_annotations()` method to generate COCO format output
- Modified `synthesize_single()` to return additional metadata (scale_ratio, rotation_angle, target_path, background_path)
- Modified `_synthesize_single_wrapper()` to preserve metadata for multiprocessing
- Updated all three processing paths (parallel multiprocessing, sequential) to track and generate COCO annotations
- COCO annotations saved to `<output_dir>/<coco_output_dir>/annotations.json`
- Annotations include:
  - **Images**: id, file_name, width, height, original_target_path, original_background_path
  - **Annotations**: id, image_id, category_id, bbox, segmentation, area, mask_area, iscrowd, scale_ratio, rotation_angle, original_target_file, original_background_file
  - **Categories**: insect category

### 7. src/synthesis/processor.py

#### Progress Bar Integration
- Added `tqdm` import with fallback (TQDM_AVAILABLE flag)
- Updated `process_directory()` to show progress when threads > 1
- Added progress bars for both multiprocessing and sequential execution
- Progress bar shows real-time synthesis progress

#### COCO Annotations Support
- Added `coco_output` and `coco_output_dir` parameters to `__init__`
- Added `_add_synthesis_metadata()` method to track synthesis metadata for COCO format
- Added `_generate_coco_annotations()` method to generate COCO JSON from tracked metadata
- Modified `synthesize_single()` to return scale_ratio, rotation_angle, target_path, background_path
- Modified `_synthesize_single_wrapper()` to preserve metadata in multiprocessing
- Updated all processing paths to call `_add_synthesis_metadata()` when coco_output=True
- Added COCO annotation generation at end of `process_directory()` when `coco_output=True`

### 8. src/metadata.py

#### Extended COCOMetadataManager
- Extended `add_image()` to support `original_background_path` parameter
- Extended `add_annotation()` to support synthesis-specific fields:
  - `scale_ratio`: Scale ratio used for synthesis
  - `rotation_angle`: Rotation angle in degrees
  - `original_target_file`: Path to source target image
  - `original_background_file`: Path to source background image
- All new fields are optional and only included in JSON if provided

---


### 1. README.md

#### Title & Overview
- Changed title from "Insect Synthesizer" to "Insect Dataset Toolkit"
- Updated overview to reflect all 4 scripts (segment, extract_frames, clean_figs, split_dataset)

#### Project Structure
- Added `scripts/` directory showing all CLI scripts
- Updated `src/` structure to show domain modules (segmentation/, framing/, cleaning/, splitting/)

#### Usage Section
- Added comprehensive usage section for each of the 4 scripts:
  - **segment.py**: Segmentation with SAM3/Otsu/GrabCut methods
  - **extract_frames.py**: Video frame extraction with multithreading
  - **clean_figs.py**: Image cleaning and deduplication
  - **split_dataset.py**: Train/test/unknown class dataset splitting
- Each script includes:
  - Basic usage examples
  - All parameters documented in tables
  - Multiple usage scenarios

#### Features Section
- Added new features reflecting all scripts:
  - Video Frame Extraction
  - Image Cleaning
  - Dataset Splitting

#### Input Validation
- Updated to reflect all scripts' validation capabilities

#### Removed
- Removed reference to `synthesize.py` (not yet implemented)

---

### 2. requirements.txt

#### Updated Comments
- Changed "Synthesize.py" references to actual scripts
- Added separate section for extract_frames.py
- Updated comments to match current script structure

#### Dependencies
All required dependencies are already present:
- Core: numpy, Pillow, tqdm
- Segmentation: torch, torchvision, opencv-python
- Video extraction: opencv-python
- Cleaning: imagehash
- Splitting: pandas

---

### 3. setup.py

#### Updated Description
- Changed from "segmenting insects... and synthesizing images" to "building insect image datasets with segmentation, frame extraction, cleaning, and dataset splitting capabilities"

#### Entry Points
All entry points were already correctly configured:
- `entomokit-segment=scripts.segment:main`
- `entomokit-clean=scripts.clean_figs:main`
- `entomokit-extract=scripts.extract_frames:main`
- `entomokit-split=scripts.split_dataset:main`

#### Extras Require
All extras_require entries were already correct:
- `segmentation`: torch, torchvision, opencv-python
- `cleaning`: imagehash
- `video`: opencv-python
- `data`: pandas
- `dev`: pytest, pytest-cov

---

### 4. docs/plans/2026-02-16-unified-scripts-architecture-design.md

#### Status Update
- Changed status from "Approved" to "Completed"
- Added last updated date

#### Migration Status Section
- Added "✅ Completed (2026-02-16)" section with:
  - Directory structure created
  - All 4 scripts migrated with common CLI infrastructure
  - Documentation updated
  - Testing verified

#### Future Enhancements
- Kept existing future improvements for ongoing work

---

### 5. scripts/segment.py

#### Repair Strategy Options
- Removed `lama_refine` from `--repair-strategy` choices (now only: `opencv`, `sam3-fill`, `black-mask`, `lama`)
- Updated help text to reflect available strategies

#### CLI Parameter Updates
- Added `--lama-mask-dilate` integer parameter (default 0) to control mask dilation before LaMa
- Set `disable_tqdm=False` to enable progress bar display

#### LaMa Logger Setup
- Configured `imagekit.lama` logger to show DEBUG logs only when `--verbose` is enabled
- Normal verbose mode keeps LaMa noise minimal

---

### 6. src/segmentation/processor.py

#### Repair Strategy Options
- Removed `lama_refine` branch from repair dispatch logic
- Updated all repair strategy check lists to exclude `lama_refine`

#### Mask Preparation
- Added `_prepare_lama_mask()` method to normalize masks to 0/255 and apply optional dilation
- Added `_get_lama_inpainter()` method with caching to avoid reloading LaMa model for each image

#### Llama Mask Dilation
- Added `lama_mask_dilate` parameter to `__init__` (default 0)
- Mask is dilated using 3x3 elliptical kernel before passing to LaMa
- Dilation size controlled via `--lama-mask-dilate` CLI parameter

#### Refine Mode Removed
- Removed `refine` parameter from `_repair_with_lama()` method
- Removed `refine` from `_get_lama_inpainter()` cache key

---

## Files Updated Summary

| File | Changes | Status |
|------|---------|--------|
| README.md | Added synthesize.py documentation, 5-script overview, project structure, usage examples, features, COCO annotations section | ✅ Complete |
| requirements.txt | Verified all dependencies including tqdm | ✅ Complete |
| setup.py | Updated description, verified entry points | ✅ Complete |
| docs/plans/2026-02-16-unified-scripts-architecture-design.md | Updated status, added synthesize.py | ✅ Complete |
| scripts/synthesize.py | Added progress bar support with tqdm, added COCO annotations generation, fixed duplicate CLI definitions | ✅ Complete |
| src/synthesis/processor.py | Added progress bar integration, added COCO annotations support | ✅ Complete |
| src/metadata.py | Extended COCOMetadataManager to support synthesis-specific metadata fields | ✅ Complete |

---

## Scripts Documented

1. **segment.py** - Segmentation script (SAM3, Otsu, GrabCut)
2. **extract_frames.py** - Video frame extraction
3. **clean_figs.py** - Image cleaning and deduplication
4. **split_dataset.py** - Dataset splitting (train/test/unknown)
5. **synthesize.py** - Image synthesis (rotation, color matching, black region avoidance, parallel processing, COCO annotations)

---

## Notes

- All scripts now properly documented with usage examples
- Synthesize script includes progress bars (tqdm), rotation, black region avoidance, parallel processing, COCO annotations
- **COCO annotations** include comprehensive target object metadata: bbox, segmentation, area, mask_area, scale_ratio, rotation_angle, original paths
- **requirements.txt**: Added `scikit-image` for color matching in synthesis
- **setup.py**: Added `scikit-image>=0.21.0` to segmentation extras and entry point `entomokit-synthesize`
- **CHANGES_SUMMARY.md**: Added COCO annotations changes (2026-02-18)
- **scripts/synthesize.py**: Fixed duplicate CLI argument definitions, verified code structure
- The unified architecture design document has been updated to reflect completed migration status
- Progress bar enabled by default for synthesize script; can be disabled via disable_tqdm parameter

---

## Additional Changes (2026-02-18) - Multi-Format Annotation Output

### Summary
Comprehensive annotation output support added to both `segment.py` and `synthesize.py` scripts. Both scripts now support COCO JSON, VOC Pascal XML, and YOLO TXT formats with proper coordinate systems.

### Changes to scripts/segment.py

**New CLI Arguments:**
- `--annotation-output-format {coco,voc,yolo}` - Select annotation output format
- `--coco-output-mode {unified,separate}` - COCO output mode (single file vs per-image files)

**Features:**
- Annotations output to `annotations/` folder (COCO/VOC) or `labels/` folder (YOLO)
- Annotation folder is sibling to `cleaned_images/` folder (not inside)
- bbox and segmentation in **original image coordinates**
- file_name points to cleaned output image
- area = bbox area, mask_area = actual mask pixel count

### Changes to scripts/synthesize.py

**Updated CLI Arguments:**
- `--annotation-output-format {coco,voc,yolo}` - Replaces `--coco-output` flag
- `--coco-output-mode {unified,separate}` - COCO output mode

**Features:**
- Annotations output to `annotations/` folder (COCO/VOC) or `labels/` folder (YOLO)
- Annotation folder is sibling to `images/` folder (not inside)
- bbox and segmentation in **synthesized image coordinates**
- Removed redundant source path information for portability
- Real-time per-image annotation output

### Changes to src/segmentation/processor.py

**New Methods:**
- `_get_annotation_output_dir()` - Get correct output directory based on format
- `_save_coco_annotation()` - Save COCO format (supports unified/separate modes)
- `_save_voc_annotation()` - Save VOC Pascal XML format
- `_save_yolo_annotation()` - Save YOLO TXT format with normalized coordinates

**Updated:**
- `__init__()` - Added `annotation_format` and `coco_output_mode` parameters
- `process_image()` - Added annotation saving based on format
- `process_directory()` - Save unified COCO file at end if in unified mode

### Changes to src/synthesis/processor.py

**Updated Methods:**
- `_save_coco_single()` - Uses target mask, applies position offset
- `_save_voc_single()` - Uses target mask, applies position offset, supports polygon
- `_save_yolo_single()` - Uses target mask, applies position offset, normalized coordinates
- `_add_synthesis_metadata()` - Removed path fields for portability

**Key Fixes:**
- bbox and segmentation now correctly calculated from target alpha channel
- Position offset applied to convert target-local to synthesized image coordinates
- YOLO format uses normalized coordinates (0-1 range)

### Changes to src/metadata.py

**Updated Methods:**
- `add_image()` - Removed `original_path` and `original_background_path` parameters
- `add_annotation()` - Removed `original_target_file` and `original_background_file` parameters
- `to_yolo_txt()` - Fixed polygon normalization (pixel -> 0-1 range)

### Changes to tests/test_segmentation.py

**Updated:**
- `test_process_image_metadata()` - Updated to expect original image coordinates in bbox

### Documentation Updates

**Updated Files:**
- `README.md` - Added comprehensive annotation output documentation for both scripts
- `README.cn.md` - Added Chinese annotation output documentation
- `docs/plans/2026-02-18-annotation-fixes.md` - Complete implementation plan with segment.py updates

### Test Results
All 59 tests pass, 2 skipped.

---

## Additional Changes (2026-03-18) - Extract Frames Time Range Support

### Summary
Added time range parameters to `extract_frames.py` script for flexible time-based frame extraction. Users can now specify start and end times to extract frames from a specific portion of the video.

### Changes to scripts/extract_frames.py

**New CLI Arguments:**
- `--start_time` (float, default: 0.0) - Start time for extraction in seconds
- `--end_time` (float, default: None) - End time for extraction in seconds (None = video end)

**Features:**
- Time values specified in seconds
- Automatic validation: `start_time >= 0`, `end_time > start_time`
- Graceful handling when `start_time` exceeds video duration (returns no frames)
- Automatic clamping when `end_time` exceeds video duration

**Usage Examples:**
```bash
# Extract frames from 5s to 30s
python scripts/extract_frames.py --input_dir videos/ --out_dir frames/ --start_time 5.0 --end_time 30.0

# Extract frames from 10s to video end
python scripts/extract_frames.py --input_dir videos/ --out_dir frames/ --start_time 10.0
```

### Changes to src/framing/extractor.py

**Updated `__init__()` method:**
- Added `start_time: Optional[float] = None` parameter
- Added `end_time: Optional[float] = None` parameter
- Added validation logic for time range parameters

**Updated `extract_from_video()` method:**
- Calculate effective start/end times based on video duration
- Convert time values to frame numbers
- Handle edge cases (time exceeds duration, etc.)

### Changes to tests/test_framing.py

**New Test Classes:**
- `TestTimeRangeValidation` - Tests for parameter validation (6 tests)
- `TestTimeRangeExtraction` - Tests for extraction behavior (4 tests)

**Test Coverage:**
- Default values (start_time=0, end_time=None)
- Negative value validation
- End time > start time validation
- Time range extraction
- Start time only extraction
- Time exceeds video duration handling

### Documentation Updates

**Updated Files:**
- `README.md` - Added parameter table entries and usage examples
- `README.cn.md` - Added Chinese documentation for time range feature
- `docs/plans/2026-03-18-extract-frames-time-range.md` - Complete implementation plan

### Test Results
All 73 tests pass, 2 skipped.
