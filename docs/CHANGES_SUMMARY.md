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

---

## Additional Changes (2026-03-26) — Doctor/Augment + Workflow Ordering + Version 0.1.4

### Summary

Added `doctor` and `augment` command modules, aligned CLI/README command ordering to the dataset workflow, switched classify dependency guidance to AutoMM (`autogluon.multimodal>=1.4.0`), added top-level `--version/-v`, and bumped package version to `0.1.4`.

### CLI and command modules

- Added `entomokit/augment.py` and augmentation runtime modules under `src/augment/`.
  - Migrated detcli-style augmentation flow.
  - Renamed I/O args to entomokit style: `--input-dir`, `--out-dir`.
  - Added `--multiply` support.
- Added `entomokit/doctor.py` and diagnostics logic in `src/doctor/service.py`.
  - Reports Python/device/package status.
  - Emits install/upgrade recommendations including `autogluon.multimodal>=1.4.0`.
- Updated top-level command registration order in `entomokit/main.py`:
  - `extract-frames`, `segment`, `synthesize`, `clean`, `augment`, `split-csv`, `classify`.
- Added top-level version flags in `entomokit/main.py`:
  - `--version`, `-v`.

### Dependency and version metadata

- Updated `setup.py`:
  - Package version: `0.1.3` -> `0.1.4`.
  - Added `augment` extra with `albumentations>=1.4.0`.
  - Updated classify extra to `autogluon.multimodal>=1.4.0`.
- Updated `requirements.txt` with:
  - `albumentations>=1.4.0`
  - `autogluon.multimodal>=1.4.0`

### Documentation

- Updated `README.md` and `README.cn.md`:
  - Added `augment` and `doctor` command docs.
  - Reorganized command introductions to the requested workflow order.
  - Added AutoMM install guidance and official link.
  - Added version usage examples (`entomokit --version`, `entomokit -v`).

### Tests

- Added `tests/test_doctor_augment_cli.py`.
- Extended `tests/test_main_cli.py` with:
  - Top-level command order assertion.
  - `--version` and `-v` behavior checks.
- Updated `tests/test_package_version.py` to assert `0.1.4`.

### Verification

Executed and passed:

```bash
pytest tests/test_main_cli.py tests/test_doctor_augment_cli.py tests/test_package_version.py -q
pytest tests/test_cli_help_texts.py tests/test_classify_predict_cli.py tests/test_classify_trainer.py -q
python -m entomokit.main -h
python -m entomokit.main doctor
python -m entomokit.main augment --help
python -m entomokit.main --version
python -m entomokit.main -v
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

---

## Additional Changes (2026-03-24) — Phase 1 & 2: Unified CLI Framework + Annotation Format Alignment

### Summary

Refactored all standalone scripts into a unified `entomokit <command>` CLI (inspired by `detcli`), and aligned annotation output directories/formats with the detcli convention using the `supervision` library.

### Phase 1 — Unified CLI Framework

**New entry point:** `entomokit` (single command replacing 5 separate scripts)

**Migrated commands:**

| Command | Original script | New extras |
|---------|----------------|------------|
| `entomokit segment` | `scripts/segment.py` | `--coco-bbox-format xywh/xyxy` |
| `entomokit extract-frames` | `scripts/extract_frames.py` | Accepts single video file path |
| `entomokit clean` | `scripts/clean_figs.py` | `--recursive` subdirectory scan |
| `entomokit split-csv` | `scripts/split_dataset.py` | `--val-ratio`, `--val-count`, `--copy-images`, `--images-dir` |
| `entomokit synthesize` | `scripts/synthesize.py` | `--coco-bbox-format xywh/xyxy` |

**New files:**
- `entomokit/main.py` — top-level dispatcher
- `entomokit/segment.py`, `extract_frames.py`, `clean.py`, `split_csv.py`, `synthesize.py` — CLI modules
- `entomokit/classify/__init__.py` — stub for Phase 3

**`setup.py` changes:**
- Single entry point: `entomokit = entomokit.main:main`
- Added `classify` extras_require group (AutoGluon, timm, pytorch-grad-cam, umap-learn, onnxruntime)

### Phase 2 — Annotation Format Alignment (detcli convention)

**New file:** `src/common/annotation_writer.py` — unified COCO/YOLO/VOC writer using `supervision` library

**Output layout now matches detcli:**

| Format | Layout |
|--------|--------|
| COCO | flat dir, `annotations.coco.json`, bbox xywh (or xyxy via `--coco-bbox-format`) |
| YOLO | `images/` + `labels/` + `data.yaml` (class names quoted) |
| VOC | `JPEGImages/` + `Annotations/` + `ImageSets/Main/default.txt` |

**Updated processors:** `src/segmentation/processor.py`, `src/synthesis/processor.py`
- Per-image YOLO/VOC saves in `process_image()`; COCO unified save at end of `process_directory()`

**New test file:** `tests/test_annotation_writer.py` — 8 unit tests for COCO/YOLO/VOC

### Test Results
All 86 tests pass, 2 skipped.

---

## Additional Changes (2026-03-24) — Phase 3: `classify` Command Group

### Summary

Implemented the full `entomokit classify` command group with 6 subcommands for AutoGluon image classification workflows.

### New files

**Business logic (`src/classification/`):**

| File | Purpose |
|------|---------|
| `utils.py` | Device selection, augment preset parsing, CSV helpers |
| `trainer.py` | AutoGluon `MultiModalPredictor` training |
| `predictor.py` | Inference — AutoGluon and ONNX |
| `evaluator.py` | Evaluation metrics (accuracy, F1, MCC, AUC, ROC) |
| `embedder.py` | Embedding extraction (timm / AG), quality metrics, UMAP |
| `cam.py` | GradCAM heatmap generation (CNN + ViT, 6 methods) |
| `exporter.py` | ONNX export via `MultiModalPredictor.export_onnx()` |

**CLI layer (`entomokit/classify/`):**

| Module | Command |
|--------|---------|
| `train.py` | `entomokit classify train` |
| `predict.py` | `entomokit classify predict` |
| `evaluate.py` | `entomokit classify evaluate` |
| `embed.py` | `entomokit classify embed` |
| `cam.py` | `entomokit classify cam` |
| `export_onnx.py` | `entomokit classify export-onnx` |
| `__init__.py` | Registers all 6 subcommands (replaces stub) |

### Command reference

```
entomokit classify train
    --train-csv     CSV with image/label columns
    --images-dir    Training images directory
    --base-model    timm backbone (default: convnextv2_femto)
    --out-dir       Output directory
    --augment       Preset (none/light/medium/heavy) or JSON array
    --max-epochs    Max training epochs (default: 50)
    --time-limit    Training time limit in hours (default: 1.0)
    --focal-loss    Enable focal loss
    --focal-loss-gamma  Focal loss gamma (default: 1.0)
    --device        auto/cpu/cuda/mps
    --batch-size    (default: 32)

entomokit classify predict
    --input-csv / --images-dir  (mutually exclusive)
    --model-dir / --onnx-model  (mutually exclusive)
    --out-dir       Output directory (predictions/predictions.csv)

entomokit classify evaluate
    --test-csv      CSV with image/label columns
    --images-dir    Images directory
    --model-dir / --onnx-model  (mutually exclusive)
    --out-dir       Output directory (logs/evaluations.txt)
    Metrics: accuracy, precision/recall macro+micro, F1, MCC, ROC-AUC

entomokit classify embed
    --images-dir    Input images directory
    --base-model    timm backbone (used if --model-dir not given)
    --model-dir     AutoGluon predictor for fine-tuned embeddings
    --label-csv     CSV for supervised metrics + UMAP coloring
    --visualize     Generate UMAP PDF (requires --label-csv)
    --umap-*        UMAP hyperparameters
    Metrics: NMI, ARI, Recall@1/5/10, kNN Acc k1/5/20, Linear Probing, mAP@R, Purity, Silhouette

entomokit classify cam
    --label-csv     Optional CSV with image/label columns
    --images-dir    Images directory
    --model-dir / --base-model  (mutually exclusive; no --onnx-model)
    --cam-method    gradcam/gradcampp/layercam/scorecam/eigencam/ablationcam
    --arch          cnn/vit (auto-detected if not set)
    --save-npy      Save raw CAM arrays as .npy (creates arrays/)
    --dump-model-structure  Write model_layers.txt for target-layer selection
    Output: figures/, cam_summary.csv, and arrays/ only with --save-npy

entomokit classify export-onnx
    --model-dir     AutoGluon predictor directory
    --out-dir       Output directory
    --opset         ONNX opset version (default: 17)
    --input-size    Model input size in pixels (default: 224)
```

### Design notes
- `cam` and `embed` do **not** support ONNX (GradCAM requires PyTorch hooks)
- All CLI args use hyphen-style (`--input-dir`, `--base-model`, etc.)
- `--augment` accepts presets (`none/light/medium/heavy`) or a raw JSON array of transform names
- Business logic modules are importable independently of the CLI layer

### New test file
- `tests/test_classify_utils.py` — 8 unit tests for `resolve_augment()`

### Test Results
All 94 tests pass, 2 skipped.

---

## Additional Changes (2026-03-25) — `classify predict` Input Flexibility + Missing-Image Diagnostics

### Summary

Updated `entomokit classify predict` to support mixed input workflows where CSV rows contain image names (not absolute paths), while keeping guardrails for ambiguous input sources.

### Changes to `entomokit/classify/predict.py`

- Removed parser-level mutual exclusion between `--input-csv` and `--images-dir`.
- Added `_resolve_predict_inputs()` to resolve input source with explicit rules:
  - CSV only: use CSV directly if paths are readable.
  - CSV + images dir: resolve CSV image values under `--images-dir` when CSV values are not directly readable paths.
  - Images dir only: scan directory images for prediction.
  - Ambiguous case (CSV already readable + non-empty images dir): abort with validation error.
- Added `PredictInputError` for structured input validation failures.
- Added friendly diagnostics for missing images referenced by CSV:
  - writes complete missing list to `<out-dir>/logs/missing_images.txt`
  - prints stderr pointer to the generated file.

### Tests

- Added `tests/test_classify_predict_cli.py` coverage for:
  - parser accepts `--input-csv` with `--images-dir`
  - CSV names resolved via `--images-dir`
  - clear error when `--images-dir` is missing for unresolved CSV paths
  - ambiguity rejection when CSV already has readable paths and images dir also has images
  - images-dir-only scan path
  - missing-image diagnostic file output in `run()`

### Verification

Executed:

```bash
pytest tests/test_classify_predict_cli.py tests/test_main_cli.py tests/test_cli_help_texts.py
```

Result: 10 passed.

---

## Additional Changes (2026-03-25) — `classify evaluate` Metrics Expansion + CSV Output

### Summary

Improved `entomokit classify evaluate` for better metric coverage and easier downstream parsing.

### Changes to `src/classification/evaluator.py`

- Added reusable `compute_classification_metrics()` helper.
- Expanded common classification metrics to include:
  - `balanced_accuracy`
  - `precision_weighted`, `recall_weighted`, `f1_weighted`
  - `quadratic_kappa`
  - `roc_auc_ovr` (in addition to `roc_auc_ovo`)
- Unified AutoGluon and ONNX evaluation paths to use the same metric computation.
- Added safe ROC-AUC handling for binary/multiclass and fallback to `NaN` when unavailable.

### Changes to `entomokit/classify/evaluate.py`

- `evaluations.txt` is no longer written under `logs/`.
- Metrics are now saved directly in `--out-dir` as `evaluations.csv`.
- CSV format is two columns: `metric,value`.

### Tests

- Added `tests/test_classify_evaluate_cli.py`:
  - verifies common metric keys are produced
  - verifies `classify evaluate` writes `evaluations.csv` under output root (not `logs/evaluations.txt`)

### Verification

Executed:

```bash
pytest tests/test_classify_evaluate_cli.py tests/test_classify_predict_cli.py tests/test_main_cli.py tests/test_cli_help_texts.py
```

Result: 12 passed.

---

## Additional Changes (2026-03-25) — Apple Silicon Training Stability + Output Directory Cleanup

### Summary

Fixed multiple issues with `entomokit classify train` to work correctly on Apple Silicon (MPS) machines, improved usability with tunable training parameters, and unified output directory naming conventions.

### Changes to `src/classification/trainer.py`

**NVML stub for non-NVIDIA machines:**
- Auto-installs a no-op `nvidia_smi` stub module on Apple Silicon / CPU machines
- Prevents `NVMLError_LibraryNotFound` from AutoGluon's internal GPU detection

**Fixed AutoGluon API change:**
- `MultiModalPredictor.fit()` no longer accepts `max_epochs` as keyword argument
- Changed to use `hyperparameters={"optim.max_epochs": N}` format

**Auto problem type detection:**
- Automatically infers `problem_type` from label count
- 2 classes → `"binary"`, ≥3 classes → `"multiclass"`

**Focal loss fix:**
- Removed `focal_loss.alpha = -1` setting (caused `RuntimeError` with binary classification)
- Let AutoGluon handle focal loss alpha internally

**CUDA AMP warning suppression:**
- Sets `env.precision=32` for non-CUDA devices
- Suppresses spurious `User provided device_type of 'cuda'` warnings on MPS/CPU

**New training parameters:**
- `--learning-rate` (default: `1e-4`)
- `--weight-decay` (default: `1e-3`)
- `--warmup-steps` (default: `0.1`)
- `--patience` (default: `10`)
- `--top-k` (default: `3`)

**Resume training:**
- Added `--resume` flag to continue training from existing model directory checkpoint

### Changes to `entomokit/classify/train.py`

**Documentation improvements:**
- Expanded `--augment` help text with preset details and JSON array examples
- All new parameters documented with AutoGluon defaults

**Warning filters:**
- Added warning filter for `torch.cuda.amp` warnings on non-CUDA devices

### Changes to `src/common/cli.py`

**ANSI cleanup in logs:**
- `_TeeStream.write()` now strips ANSI CSI sequences (cursor movement, colors)
- Prevents `\x1b[A` artifacts in log files from progress bars

### Changes to `src/common/annotation_writer.py`

**COCO output simplification:**
- Removed `images_directory_path` parameter from `dataset.as_coco()` call
- COCO mode no longer copies source images into output directory

**YOLO data.yaml enhancement:**
- Added `train: images` line to `data.yaml` for YOLO format

### Changes to `src/segmentation/processor.py`

**Output directory rename:**
- Changed output directory from `cleaned_images/` to `images/`
- Aligns with detcli convention

### Deleted files

- `README.cn.md` — removed Chinese documentation (consolidate to English only)

### New test file

- `tests/test_classify_trainer.py` — 8 tests covering:
  - `max_epochs` hyperparameter format
  - Binary vs multiclass problem type detection
  - Resume training functionality
  - NVML stub installation
  - CUDA warning filtering

### Updated test files

- `tests/test_annotation_writer.py` — added COCO no-copy test, YOLO `train: images` test
- `tests/test_segmentation.py` — updated assertions for `images/` directory name, code formatting

### Test Results

All tests pass after changes.

---

## Additional Changes (2026-03-25) — ONNX Export/Inference Compatibility and Docs Update

### Summary

Fixed ONNX export API mismatches with current AutoGluon, resolved ONNX inference input-feed errors, normalized ONNX output path layout, and updated user-facing docs for ONNX runtime requirements.

### Changes to `entomokit/classify/export_onnx.py`

- Removed obsolete `--input-size` argument (not part of current AutoGluon `export_onnx` API).
- Added optional `--sample-image` argument for explicit trace input.
- Updated help text to clarify trace behavior.

### Changes to `src/classification/exporter.py`

- Switched to current AutoGluon export kwargs:
  - `data=...`
  - `path=...`
  - `opset_version=...`
- Added image column auto-detection from `assets.json` (`image_path` column lookup).
- Added temporary trace image generation when `--sample-image` is not provided.
- Added cleanup of temporary trace image after export.
- Added compatibility handling for nested export artifacts and normalized final output to:
  - `--out-dir/model.onnx`
- Added sidecar metadata output:
  - `--out-dir/label_classes.json`

### Changes to `src/classification/predictor.py`

- Added explicit `onnxruntime` import error guidance with install commands.
- Added ONNX path normalization and existence checks.
- Fixed ONNX input feed to support multi-input models:
  - auto-feeds `*_valid_num` tensors
  - auto-expands image tensor to 5D when required
- Improved output tensor selection to pick logits tensor reliably.
- Added `label_classes.json` loading for label-name mapping.
- Prediction output now includes:
  - `prediction_index` (numeric class index)
  - `prediction` (class name when sidecar exists, else numeric index)

### Changes to `src/classification/evaluator.py`

- Added ONNX label mapping support during evaluation:
  - maps string labels in CSV to class indices via `label_classes.json`
  - uses `prediction_index` for metric computation when mapping is available

### Changes to CLI help and docs

- Updated ONNX help text in:
  - `entomokit/classify/predict.py`
  - `entomokit/classify/evaluate.py`
- Updated `README.md`:
  - ONNX examples use `--onnx-model runs/onnx/model.onnx`
  - added `onnxruntime` installation note
  - documented trace behavior and optional `--sample-image`
  - documented `label_classes.json` sidecar and `prediction_index`
- Updated `requirements.txt` to include `onnxruntime`.

### Tests added/updated

- Added `tests/test_classify_export_onnx_cli.py`.
- Added `tests/test_classification_predictor_onnx.py`.
- Updated `tests/test_classify_evaluate_cli.py` with ONNX label-mapping coverage.

### Verification

Executed and passed:

```bash
pytest -q tests/test_classify_export_onnx_cli.py tests/test_classification_predictor_onnx.py tests/test_classify_evaluate_cli.py tests/test_cli_help_texts.py tests/test_main_cli.py
entomokit classify export-onnx --model-dir out/train/AutogluonModels/convnextv2_femto/ --out-dir out/onnx
entomokit classify evaluate --test-csv out/split/test.known.csv --images-dir data/Epidorcus/images/ --onnx-model out/onnx/model.onnx --out-dir out/onnx-eval --device mps
entomokit classify predict --input-csv out/split/test.known.csv --images-dir data/Epidorcus/images/ --onnx-model out/onnx/model.onnx --out-dir out/onnx-predict --device mps
```

---

## Additional Changes (2026-03-25) — `classify embed` Warning Cleanup + Output Path Fixes

### Summary

Fixed CUDA autocast warning spam on non-CUDA machines, suppressed sklearn numerical warnings during embedding metric computation, and corrected output paths for `entomokit classify embed`.

### Changes to `entomokit/main.py`

- Added `_ensure_project_root_on_path()` to prioritize local project root over similarly named site-packages.
- Prevents unintended import of external `src` packages that trigger CUDA-related warnings at import time.

### Changes to `entomokit/classify/embed.py`

- Added warning filters for CUDA autocast and GradScaler warnings on non-CUDA devices.
- `metrics.csv` now saved directly to `--out-dir` (previously saved to `--out-dir/logs/`).
- Added terminal output `Metrics saved to: <path>` after printing metric values.

### Changes to `src/classification/embedder.py`

- Added warning suppression for CUDA autocast/GradScaler warnings in `extract_embeddings_ag()`.
- Suppressed sklearn matmul RuntimeWarnings inside `compute_embedding_metrics()`.
- Added explicit float32 conversion for AutoGluon embeddings.
- Changed LogisticRegression solver to `"liblinear"` for stability with small datasets.

### Changes to `README.md`

- Updated `classify embed` output docs from `logs/metrics.csv` to `metrics.csv`.

### New test file

- `tests/test_classify_embed_cli.py` — verifies:
  - `metrics.csv` written to output root directory
  - Terminal includes `Metrics saved to:` message
  - `UMAP saved to:` message present

### Verification

Executed and passed:

```bash
entomokit classify embed --images-dir out/split/images/test_known/ --out-dir out/embed --model-dir out/train/AutogluonModels/convnextv2_femto/ --label-csv out/split/test.known.csv --visualize
entomokit classify embed --images-dir out/split/images/test_known/ --out-dir out/embed-default --base-model convnextv2_femto
pytest -q tests/test_classify_embed_cli.py tests/test_classify_evaluate_cli.py tests/test_classify_predict_cli.py
```

Result: 10 passed, no CUDA/sklearn warnings on stdout.

---

## Additional Changes (2026-03-25) — `classify cam` Optional Labels + Layer Dump + Output Cleanup

### Summary

Improved `entomokit classify cam` usability by making labels optional, avoiding empty `arrays/` directories when not needed, and adding a model-layer dump file for easier `--target-layer-name` selection.

### Changes to `entomokit/classify/cam.py`

- Made `--label-csv` optional.
- Added `--dump-model-structure` flag.
- Updated `--images-dir` help text to support both CSV-driven and directory-scan workflows.
- Forwarded optional label CSV and model-structure dump flag to runtime `run_cam()`.

### Changes to `src/classification/cam.py`

- Added `collect_image_label_rows()`:
  - If `--label-csv` is provided, reads `image,label` from CSV.
  - If omitted, scans `--images-dir` recursively for image files and uses empty labels.
- Updated output directory creation:
  - Always creates `figures/`.
  - Creates `arrays/` only when `--save-npy` is enabled.
- Added model structure export via `write_model_structure()` to generate `model_layers.txt`.
- Enhanced `--target-layer-name` resolution:
  - Supports numeric indexing (e.g., `blocks.11.norm1`).
  - Supports `ModuleDict` key traversal.
- Fixed name-collision bug where boolean `dump_model_structure` shadowed callable helper.

### Changes to `README.md`

- Updated `classify cam` example to work without mandatory `--label-csv`.
- Documented that `arrays/` is generated only with `--save-npy`.
- Documented `--dump-model-structure` and `model_layers.txt` usage.

### New test file

- `tests/test_classification_cam.py` — verifies:
  - `arrays/` is not created when `--save-npy` is omitted.
  - No-label image-directory scan path.
  - CLI runner forwards optional label CSV and dump flag correctly.
  - Parser accepts missing `--label-csv`.
  - `run_cam()` writes `model_layers.txt` when dump flag is enabled.

### Verification

Executed and passed:

```bash
pytest tests/test_classification_cam.py
pytest tests/test_classification_cam.py tests/test_classify_embed_cli.py
pytest tests/test_cli_help_texts.py tests/test_main_cli.py
entomokit classify cam --images-dir data/Epidorcus/images/ --out-dir out/cam --model-dir out/train/AutogluonModels/convnextv2_femto/ --num-classes 2 --cam-method scorecam --dump-model-structure --max-images 1
```

---

## Additional Changes (2026-03-25) — Remove Legacy Folders + Bump Version to 0.1.3

### Summary

Cleaned up obsolete migration artifacts by removing legacy implementation folders and updated package version directly from `0.1.1` to `0.1.3`.

### Removed

- `scripts/` (legacy standalone CLI scripts no longer used)
- `add_functions/` (historical classification/GradCAM source scripts already integrated into `src/classification/` and `entomokit/classify/`)

### Versioning

- Updated `setup.py` package version:
  - `0.1.1` → `0.1.3`
- Skipped `0.1.2` per release sequencing decision.

### Tests

- Added `tests/test_package_version.py` to assert published setup version is `0.1.3`.
- Removed obsolete `tests/test_segment_cli.py` (it depended on deleted legacy `scripts/segment.py` entrypoint).

### Verification

Executed and passed:

```bash
pytest -q tests/test_package_version.py
pytest -q tests/test_main_cli.py tests/test_cli_help_texts.py
```

---

## Additional Changes (2026-03-25) — CLI Help UI Refresh (Examples + Boxed Sections)

### Summary

Refined CLI help presentation to better match the `detection` tool style by adding quick command examples near the top and boxed section titles for readability.

### New file

- `entomokit/help_style.py`
  - Added shared helpers for consistent help rendering:
    - `RichHelpFormatter` (defaults + raw text newlines)
    - `with_examples()` (injects `Quick examples` blocks)
    - `style_parser()` (sets `[ Options ]` / `[ Commands ]` headings)

### Changes to top-level CLI

- `entomokit/main.py`
  - Added top-level `Quick examples` section.
  - Updated parser styling to use boxed section titles.

### Changes to command help pages

Applied the same help style to command groups and subcommands:

- Base commands:
  - `entomokit/segment.py`
  - `entomokit/extract_frames.py`
  - `entomokit/clean.py`
  - `entomokit/split_csv.py`
  - `entomokit/synthesize.py`
- Classification group and subcommands:
  - `entomokit/classify/__init__.py`
  - `entomokit/classify/train.py`
  - `entomokit/classify/predict.py`
  - `entomokit/classify/evaluate.py`
  - `entomokit/classify/embed.py`
  - `entomokit/classify/cam.py`
  - `entomokit/classify/export_onnx.py`

### Tests

- `tests/test_main_cli.py`
  - Added assertions for:
    - top-level quick examples and boxed sections
    - `segment --help` quick examples and boxed options
    - `classify --help` quick examples and boxed commands/options
    - `classify train --help` quick examples and boxed options

### Verification

Executed and passed:

```bash
pytest -q tests/test_main_cli.py
pytest -q tests/test_cli_help_texts.py
pytest -q tests/test_classify_predict_cli.py tests/test_classify_trainer.py
pytest -q tests/test_classify_embed_cli.py tests/test_classify_evaluate_cli.py tests/test_classify_export_onnx_cli.py tests/test_classification_cam.py
python -m entomokit.main --help
python -m entomokit.main segment --help
python -m entomokit.main classify --help
python -m entomokit.main classify train --help
```

---

## Additional Changes (2026-03-26) — AI Assistant Skill for Guided Workflows

### Summary

Added `entomokit-workflow` skill to enable AI assistants (OpenCode, Claude Code, Codex, etc.) to guide non-CLI users through the EntomoKit pipeline with conversational workflow orchestration.

### New files

**Skill structure (`skills/entomokit-workflow/`):**

| File | Purpose |
|------|---------|
| `SKILL.md` | Main skill definition with core rules, phases, and protocols |
| `scripts/export_cli_schema.py` | Exports machine-readable CLI parameter schema |
| `scripts/run_guarded_step.py` | Guarded execution bridge for conversational runs |
| `scripts/resolve_data_dir.py` | Dynamic demo data path discovery |
| `references/workflow.md` | Per-phase execution script |
| `references/command-profiles.md` | Parameter defaults per command |
| `references/csv-validation.md` | Strict CSV checks and fixes |
| `references/error-catalog.md` | Error mapping and repair actions |
| `references/teaching-playbook.md` | Opt-in demo flows using sample data |
| `references/path-resolution.md` | Path resolution rules |
| `references/dialog-templates.md` | Mandatory pre-run and post-run cards |
| `references/progress-memory.md` | Session state management |
| `references/release-checklist.md` | Packaging readiness checks |

### Skill capabilities

**Core Rules:**
- Run `entomokit doctor` before any substantive step
- Every command requires explicit user approval with full parameter card
- Parameter names/options loaded from runtime CLI schema (not memory)
- Never write outputs into repository `data/` or mixed root paths
- Create dedicated run roots under `runs/runNNN/`
- Prefer `entomokit <command>` for all workflow actions

**Workflow Phases:**
- Phase 0: `doctor`
- Phase 1: `extract-frames` (optional) → `clean` (required) → `segment`/`synthesize`/`augment` (optional)
- Phase 2: CSV teaching + validation + `split-csv`
- Phase 3: `classify train` → `predict` → `evaluate` → `embed`/`cam` → `export-onnx`

**Features:**
- Parameter validation gate (unknown/missing/invalid params blocked)
- Label strategy confirmation before CSV generation
- Device selection guardrail (prefer faster backend)
- Demo visibility prompts at key checkpoints
- AutoMM split policy guidance (train + test.known default)

### Documentation

**Updated files:**
- `README.md` — Added "AI Assistant Integration (Skills)" section with:
  - What is the skill
  - Installation instructions for OpenCode, Claude Code, Codex
  - Usage examples
  - Feature table
