# Insect Dataset Toolkit (EntomoKit)

[中文文档](README.cn.md) | **English**

A Python-based toolkit for building insect image datasets. Provides a unified `entomokit` CLI with commands for frame extraction, segmentation, synthesis, cleaning, augmentation, dataset splitting, AutoMM classification, and environment diagnostics. Includes an `entomokit-workflow` skill for AI assistants (OpenCode, Claude Code, Codex) to guide non-CLI users through the pipeline.

## Overview

All functionality is accessed through a single entry point:

```
entomokit <command> [options]
```

| Command | Description |
|---------|-------------|
| `extract-frames` | Extract frames from video files |
| `segment` | Segment insects from images (SAM3, Otsu, GrabCut) |
| `synthesize` | Composite insects onto background images |
| `clean` | Clean and deduplicate images |
| `augment` | Augment images with presets or custom albumentations policy |
| `split-csv` | Split datasets into train/val/test CSVs |
| `classify train` | Train an AutoMM image classifier |
| `classify predict` | Run inference (AutoGluon or ONNX) |
| `classify evaluate` | Evaluate model performance |
| `classify embed` | Extract embeddings + UMAP + quality metrics |
| `classify cam` | Generate GradCAM heatmaps |
| `classify export-onnx` | Export model to ONNX format |
| `doctor` | Diagnose environment and missing dependencies |

## Features

- **Unified CLI**: Single `entomokit` entry point — no more per-script invocations
- **Multiple Segmentation Methods**: SAM3 (with alpha channel), SAM3-bbox (cropped), Otsu thresholding, GrabCut
- **Flexible Repair Strategies**: OpenCV morphological operations, SAM3-based or LaMa hole filling
- **Annotation Output**: COCO JSON, VOC Pascal XML, YOLO TXT
- **Video Frame Extraction**: Multithreaded extraction with time range support
- **Image Cleaning**: Resize, deduplicate (MD5/Phash), and standardize image naming; recursive mode
- **Image Augmentation**: Albumentations-based preset/custom augmentation with deterministic seeds
- **Dataset Splitting**: Ratio or count-based train/val/test splits with stratification
- **Image Synthesis**: Advanced compositing with rotation, color matching, and black region avoidance
- **AutoMM Classification**: Train, predict, evaluate, embed, GradCAM, and ONNX export
- **Environment Diagnostics**: `doctor` command reports missing/outdated dependencies and install suggestions
- **Embedding Quality Metrics**: NMI, ARI, Recall@K, kNN accuracy, mAP@R, Silhouette, UMAP visualization
- **Parallel Processing**: Multi-threaded image processing with configurable worker count
- **Comprehensive Logging**: Detailed logging with verbose mode and log file output
- **AI Assistant Integration**: `entomokit-workflow` skill for guided conversational workflows with OpenCode, Claude Code, Codex, etc.

## Requirements

- Python 3.8+
- Operating Systems: Linux, macOS, Windows

## Installation

Recommended: use an isolated Python environment to avoid dependency conflicts with your system/site-packages.

### Deployment Mode A (Recommended): Isolated Environment

Choose one of the following:

**Option 1: conda**

```bash
conda create -n entomokit python=3.11 -y
conda activate entomokit
pip install -e .
```

**Option 2: uv + venv**

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

**Option 3: stdlib venv + pip**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Basic Installation

```bash
pip install -e .
```

### Deployment Mode B (Not Recommended): Direct Global pip

You can install directly into the current Python environment, but this may cause dependency conflicts with other projects:

```bash
pip install -e .
```

### With Classification Support

For classification commands (AutoMM, timm, GradCAM, UMAP):

```bash
pip install -e ".[classify]"
```

AutoMM official install reference:
https://auto.gluon.ai/stable/install.html

### With Segmentation Support

For SAM3-based segmentation:

```bash
pip install -e ".[segmentation]"
```

### With Video Processing

For video frame extraction:

```bash
pip install -e ".[video]"
```

### With Image Cleaning

For perceptual hash deduplication:

```bash
pip install -e ".[cleaning]"
```

### With Augmentation

For `entomokit augment`:

```bash
pip install -e ".[augment]"
```

### Development Installation

```bash
pip install -e ".[dev,classify,segmentation,video,cleaning,augment]"
```

## Project Structure

```
.
├── entomokit/              # Unified CLI package
│   ├── main.py             # Entry point dispatcher
│   ├── segment.py          # entomokit segment
│   ├── extract_frames.py   # entomokit extract-frames
│   ├── synthesize.py       # entomokit synthesize
│   ├── clean.py            # entomokit clean
│   ├── augment.py          # entomokit augment
│   ├── split_csv.py        # entomokit split-csv
│   ├── doctor.py           # entomokit doctor
│   ├── help_style.py       # Rich help formatting
│   └── classify/           # entomokit classify *
│       ├── train.py
│       ├── predict.py
│       ├── evaluate.py
│       ├── embed.py
│       ├── cam.py
│       └── export_onnx.py
├── src/
│   ├── common/             # Shared utilities (CLI, annotation_writer, logging, validators)
│   ├── classification/     # AutoGluon classification logic
│   ├── segmentation.py     # Segmentation domain logic
│   ├── framing/            # Video framing domain logic
│   ├── cleaning/           # Image cleaning domain logic
│   ├── augment/            # Image augmentation domain logic
│   ├── splitting/          # Dataset splitting domain logic
│   ├── synthesis/          # Image synthesis domain logic
│   ├── doctor/             # Environment diagnostics
│   ├── sam3/               # SAM3 model implementation
│   └── lama/               # LaMa inpainting implementation
├── tests/                  # Test files
├── data/                   # Data directory (large files ignored)
├── models/                 # Model weights (large files ignored)
├── docs/                   # Plans, specs, change summaries
├── requirements.txt        # Python dependencies
└── setup.py                # Package setup
```

## Model Requirements

### SAM3 Model

For SAM3-based methods (`sam3`, `sam3-bbox`), download the checkpoint from Hugging Face and pass it with `--sam3-checkpoint`.

Download link: https://huggingface.co/facebook/sam3

### LaMa Model

For `--repair-strategy lama`, place the Big-LaMa model at:
```
models/big-lama/
├── config.yaml
└── models/best.ckpt
```

Download link: https://github.com/advimman/lama

### AutoMM / timm (classify commands)

Install the `classify` extras — AutoMM will download backbone weights automatically on first use.

Supported timm backbones include:
- `convnextv2_femto` (default, lightweight)
- `convnextv2_tiny`, `convnextv2_small`, `convnextv2_base`
- `resnet18`, `resnet50`, `resnet101`
- `efficientnet_b0` through `efficientnet_b7`
- `vit_small_patch16_224`, `vit_base_patch16_224`
- And many more from [timm models](https://huggingface.co/timm)

## Usage

Recommended workflow command order:

1. `extract-frames`
2. `segment`
3. `synthesize`
4. `clean`
5. `augment`
6. `split-csv`
7. `classify`

### Segment Command

Segment insects from images using multiple methods (SAM3, Otsu, GrabCut). Optionally generates annotations in COCO, VOC, or YOLO format.

#### Basic Usage

```bash
# SAM3 with alpha channel (transparent background)
entomokit segment \
    --input-dir images/clean_insects/ \
    --out-dir outputs/insects_clean/ \
    --sam3-checkpoint models/sam3.pt \
    --segmentation-method sam3 \
    --device auto

# With COCO annotations
entomokit segment \
    --input-dir images/clean_insects/ \
    --out-dir outputs/insects_clean/ \
    --sam3-checkpoint models/sam3.pt \
    --segmentation-method sam3 \
    --annotation-format coco

# With YOLO annotations and xyxy bbox format
entomokit segment \
    --input-dir images/ --out-dir outputs/ \
    --segmentation-method otsu \
    --annotation-format yolo \
    --coco-bbox-format xyxy

# SAM3-bbox mode (crops to bounding box)
entomokit segment \
    --input-dir images/ --out-dir outputs/ \
    --sam3-checkpoint models/sam3.pt \
    --segmentation-method sam3-bbox \
    --padding-ratio 0.1

# With LaMa repair for filling holes
entomokit segment \
    --input-dir images/ --out-dir outputs/ \
    --sam3-checkpoint models/sam3.pt \
    --repair-strategy lama \
    --lama-model models/big-lama/
```

#### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input-dir` | Input directory | Required |
| `--out-dir` | Output directory | Required |
| `--segmentation-method` | `sam3`, `sam3-bbox`, `otsu`, `grabcut` | `sam3` |
| `--sam3-checkpoint` | SAM3 checkpoint path | Required for sam3/sam3-bbox |
| `--hint` | Text prompt for SAM3 grounding | `insect` |
| `--device` | `auto`, `cpu`, `cuda`, `mps` | `auto` |
| `--confidence-threshold` | Minimum confidence score for masks | `0.0` |
| `--padding-ratio` | Padding ratio for bounding box | `0.0` |
| `--repair-strategy` | `opencv`, `sam3-fill`, `black-mask`, `lama` | None |
| `--lama-model` | LaMa model directory | None |
| `--annotation-format` | `coco`, `voc`, `yolo` | None |
| `--coco-bbox-format` | `xywh`, `xyxy` | `xywh` |
| `--threads` | Parallel workers | 8 |

**Output structure (COCO example):**
```
output_dir/
├── annotations.coco.json     # COCO annotations
├── cleaned_images/           # Segmented images
│   ├── image_01.png
│   └── ...
└── repaired_images/          # (if repair-strategy enabled)
```

**YOLO/VOC layout:**
```
output_dir/
├── images/
├── labels/                   # YOLO: .txt per image + data.yaml
└── Annotations/              # VOC: .xml per image + ImageSets/Main/
```

---

### Extract Frames Command

Extract frames from video files. Accepts a directory or a single video file path.

```bash
# Extract from directory every 1 second
entomokit extract-frames --input-dir videos/ --out-dir frames/

# Extract from single video, time range 5s–30s
entomokit extract-frames --input-dir video.mp4 --out-dir frames/ \
    --start-time 5.0 --end-time 30.0

# Custom interval and format
entomokit extract-frames --input-dir videos/ --out-dir frames/ \
    --interval 500 --out-image-format png

# Limit frames per video
entomokit extract-frames --input-dir videos/ --out-dir frames/ \
    --max-frames 100
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input-dir` | Video directory or single video file | Required |
| `--out-dir` | Output directory | Required |
| `--interval` | Interval in milliseconds | 1000 |
| `--start-time` | Start time in seconds | 0 |
| `--end-time` | End time in seconds | video end |
| `--out-image-format` | jpg/png/tif | jpg |
| `--threads` | Parallel threads | 8 |
| `--max-frames` | Max frames per video | All |

**Supported video formats**: mp4, mov, avi, mkv, webm, flv, m4v, mpeg, mpg, wmv, 3gp, ts

---

### Clean Command

Clean and deduplicate images with consistent naming.

```bash
# Basic (MD5 dedup)
entomokit clean --input-dir images/raw/ --out-dir images/cleaned/

# Recursive scan + perceptual hash
entomokit clean --input-dir images/ --out-dir cleaned/ \
    --recursive --dedup-mode phash --phash-threshold 5

# Resize to shorter side 512px
entomokit clean --input-dir images/raw/ --out-dir cleaned/ \
    --out-short-size 512 --out-image-format png

# Keep original size and EXIF data
entomokit clean --input-dir images/raw/ --out-dir cleaned/ \
    --out-short-size -1 --keep-exif
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input-dir` | Input directory | Required |
| `--out-dir` | Output directory | Required |
| `--recursive` | Scan subdirectories | No |
| `--out-short-size` | Shorter side size (-1 = original) | 512 |
| `--dedup-mode` | `none`, `md5`, `phash` | md5 |
| `--phash-threshold` | Phash similarity threshold | 5 |
| `--out-image-format` | jpg/png/tif | jpg |
| `--keep-exif` | Preserve EXIF metadata | No |
| `--threads` | Parallel threads | 12 |

---

### Augment Command

Augment images with albumentations presets or a custom policy file.

```bash
# Light preset (default), one output per input image
entomokit augment --input-dir images/cleaned/ --out-dir images/augmented/

# Heavy preset and 3 copies per image
entomokit augment --input-dir images/cleaned/ --out-dir images/augmented/ \
    --preset heavy --multiply 3 --seed 123

# Custom policy JSON
entomokit augment --input-dir images/cleaned/ --out-dir images/augmented/ \
    --policy configs/augment_policy.json
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input-dir` | Input image directory | Required |
| `--out-dir` | Output directory | Required |
| `--preset` | `light`, `medium`, `heavy`, `safe-for-small-dataset` | `light` |
| `--policy` | Custom policy JSON path (exclusive with `--preset`) | None |
| `--seed` | Random seed for reproducibility | 42 |
| `--multiply` | Augmented copies per input image | 1 |

**Output:**
```
output_dir/
├── images/
└── augment_manifest.json
```

---

### Split-CSV Command

Split a labelled CSV into train / val / test files.

```bash
# Ratio split (80/10/10)
entomokit split-csv --raw-image-csv data/images.csv \
    --known-test-classes-ratio 0.1 --val-ratio 0.1 --out-dir datasets/

# Count split with image copy
entomokit split-csv --raw-image-csv data/images.csv --mode count \
    --known-test-classes-count 100 --val-count 50 \
    --copy-images --images-dir images/ --out-dir datasets/

# With unknown class test split (for open-set evaluation)
entomokit split-csv --raw-image-csv data/images.csv \
    --unknown-test-classes-ratio 0.1 \
    --known-test-classes-ratio 0.1 \
    --out-dir datasets/

# Filter classes with too few samples
entomokit split-csv --raw-image-csv data/images.csv \
    --min-count-per-class 10 \
    --out-dir datasets/
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--raw-image-csv` | Input CSV (image, label columns) | Required |
| `--out-dir` | Output directory | Required |
| `--mode` | `ratio` or `count` | ratio |
| `--val-ratio` / `--val-count` | Validation split | None |
| `--known-test-classes-ratio` | Known-class test ratio | 0.1 |
| `--unknown-test-classes-ratio` | Unknown-class test ratio | 0 |
| `--min-count-per-class` | Drop classes with fewer images | 0 |
| `--max-count-per-class` | Cap images per class | None |
| `--copy-images` | Copy images into split subdirs | No |
| `--images-dir` | Source images dir (for copy) | None |
| `--seed` | Random seed | 42 |

**Output:**
```
output_dir/
├── train.csv
├── val.csv          # if --val-ratio / --val-count specified
├── test.known.csv
├── test.unknown.csv # if unknown classes configured
├── class_count/     # per-split class counts
│   ├── class.train.count
│   ├── class.val.count
│   └── ...
└── images/          # if --copy-images
    ├── train/
    ├── val/
    └── test_known/
```

---

### Synthesize Command

Composite target objects onto background images with rotation, color matching, and intelligent positioning.

```bash
# Basic synthesis
entomokit synthesize \
    --target-dir images/targets/ \
    --background-dir images/backgrounds/ \
    --out-dir outputs/synthesized/ \
    --num-syntheses 10

# With COCO annotations and rotation
entomokit synthesize \
    --target-dir images/targets/ \
    --background-dir images/backgrounds/ \
    --out-dir outputs/synthesized/ \
    --num-syntheses 10 \
    --annotation-output-format coco \
    --rotate 30

# With YOLO annotations
entomokit synthesize \
    --target-dir images/targets/ \
    --background-dir images/backgrounds/ \
    --out-dir outputs/synthesized/ \
    --annotation-output-format yolo \
    --coco-bbox-format xyxy

# Avoid black regions in backgrounds
entomokit synthesize \
    --target-dir images/targets/ \
    --background-dir images/backgrounds/ \
    --out-dir outputs/synthesized/ \
    --avoid-black-regions \
    --color-match-strength 0.7
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--target-dir` | Target images (with alpha channel) | Required |
| `--background-dir` | Background images | Required |
| `--out-dir` | Output directory | Required |
| `--num-syntheses` | Syntheses per target | 10 |
| `--annotation-output-format` | `coco`, `voc`, `yolo` | `coco` |
| `--coco-bbox-format` | `xywh`, `xyxy` | `xywh` |
| `--rotate` | Max rotation degrees | 0 |
| `--avoid-black-regions` | Skip dark background areas | No |
| `--color-match-strength` | 0–1 color matching | 0.5 |
| `--area-ratio-min` | Min target/background area ratio | 0.05 |
| `--area-ratio-max` | Max target/background area ratio | 0.20 |
| `--threads` | Parallel workers | 4 |

**Output (COCO):**
```
output_dir/
├── images/
│   ├── target_01.png
│   └── ...
└── annotations.coco.json
```

**Output (YOLO):**
```
output_dir/
├── images/
├── labels/
└── data.yaml
```

---

### Classify Commands

All classification commands require the `classify` extras:

```bash
pip install -e ".[classify]"
```

#### `classify train`

Train an image classifier using AutoGluon MultiModalPredictor.

```bash
entomokit classify train \
    --train-csv data/train.csv \
    --images-dir data/images/ \
    --out-dir runs/exp1/ \
    --base-model convnextv2_femto \
    --augment medium \
    --max-epochs 50 \
    --learning-rate 3e-4 \
    --device auto
```

**Resume training** (extend epoch limit from 50 to 100):

```bash
entomokit classify train \
    --train-csv data/train.csv \
    --images-dir data/images/ \
    --out-dir runs/exp1/ \
    --base-model convnextv2_femto \
    --max-epochs 100 \
    --resume
```

**Custom augmentation**:

```bash
# Using preset
entomokit classify train ... --augment heavy

# Using custom transforms (JSON array)
entomokit classify train ... --augment '["random_resize_crop","color_jitter","randaug"]'
```

**With focal loss** (for imbalanced classes):

```bash
entomokit classify train \
    --train-csv data/train.csv \
    --images-dir data/images/ \
    --out-dir runs/exp1/ \
    --focal-loss \
    --focal-loss-gamma 2.0
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--train-csv` | CSV with `image` and `label` columns | Required |
| `--images-dir` | Training images directory | Required |
| `--out-dir` | Output directory | Required |
| `--base-model` | timm backbone name | `convnextv2_femto` |
| `--augment` | Preset or JSON array | `medium` |
| `--max-epochs` | Max training epochs | 50 |
| `--time-limit` | Time limit in hours | 1.0 |
| `--resume` | Continue from checkpoint | No |
| `--learning-rate` | AutoGluon `optim.lr` | `1e-4` |
| `--weight-decay` | AutoGluon `optim.weight_decay` | `1e-3` |
| `--warmup-steps` | AutoGluon `optim.warmup_steps` | `0.1` |
| `--patience` | Early-stopping patience | 10 |
| `--top-k` | Checkpoint averaging count | 3 |
| `--focal-loss` | Enable focal loss | No |
| `--device` | `auto/cpu/cuda/mps` | `auto` |
| `--batch-size` | Batch size | 32 |
| `--num-workers` | DataLoader workers | 4 |

**Augmentation presets**:
| Preset | Transforms |
|--------|-----------|
| `none` | resize_shorter_side, center_crop |
| `light` | none + random_horizontal_flip |
| `medium` | light + color_jitter + trivial_augment |
| `heavy` | random_resize_crop, random_horizontal_flip, random_vertical_flip, color_jitter, trivial_augment, randaug |

---

#### `classify predict`

Run inference on images using AutoGluon or ONNX model.

```bash
# AutoGluon model
entomokit classify predict \
    --images-dir data/test/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --out-dir runs/predict/

# ONNX model
entomokit classify predict \
    --input-csv test.csv \
    --onnx-model runs/onnx/model.onnx \
    --out-dir runs/predict/

# CSV image names + image root directory
entomokit classify predict \
    --input-csv out/split/test.known.csv \
    --images-dir data/Epidorcus/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --out-dir runs/predict/
```

**Input resolution rules**:
- Provide at least one of `--input-csv` or `--images-dir`
- If CSV `image` values are already readable paths, CSV is used directly
- If CSV `image` values are names/relative paths, also provide `--images-dir`
- If only `--images-dir` is given, all images in that directory are predicted

**ONNX requirements**:
```bash
pip install onnxruntime
# or
pip install 'entomokit[classify]'
```

**ONNX output**:
- `prediction` is class name when `label_classes.json` exists next to the ONNX file
- `prediction_index` always stores the numeric class index

---

#### `classify evaluate`

Evaluate model performance on a test set.

```bash
entomokit classify evaluate \
    --test-csv data/test.csv \
    --images-dir data/images/ \
    --onnx-model runs/onnx/model.onnx \
    --out-dir runs/eval/
```

**Output metrics** (saved to `evaluations.csv`):
- Accuracy, Balanced Accuracy
- Precision/Recall/F1 (macro, micro, weighted)
- Matthews Correlation Coefficient (MCC)
- Quadratic Kappa
- ROC-AUC (OVO, OVR)

---

#### `classify embed`

Extract embeddings and compute quality metrics.

```bash
# Pretrained timm backbone (no training required)
entomokit classify embed \
    --images-dir data/images/ \
    --base-model convnextv2_femto \
    --label-csv data/labels.csv \
    --visualize \
    --out-dir runs/embed/

# Fine-tuned AutoGluon backbone
entomokit classify embed \
    --images-dir data/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --label-csv data/labels.csv \
    --out-dir runs/embed/
```

**Outputs**:
- `embeddings.csv` — Feature vectors (feat_0, feat_1, ...)
- `metrics.csv` — Quality metrics
- `umap.pdf` — UMAP visualization (with `--visualize`)

**Quality metrics**:
| Metric | Description |
|--------|-------------|
| NMI | Normalized Mutual Information |
| ARI | Adjusted Rand Index |
| Recall@1/5/10 | Retrieval recall at K |
| kNN_Acc_k1/5/20 | k-NN classification accuracy |
| Linear_Probing_Acc | Linear classifier accuracy |
| mAP@R | Mean Average Precision at R |
| Purity | Cluster purity |
| Silhouette_Score | Clustering quality |

---

#### `classify cam`

Generate GradCAM heatmaps for model interpretability.

```bash
entomokit classify cam \
    --images-dir data/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --cam-method gradcam \
    --out-dir runs/cam/ \
    --save-npy
```

**With ground-truth labels**:
```bash
entomokit classify cam \
    --label-csv data/test.csv \
    --images-dir data/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --out-dir runs/cam/
```

**CAM methods**: `gradcam`, `gradcampp`, `layercam`, `scorecam`, `eigencam`, `ablationcam`

**Architecture auto-detection**: Automatically detects CNN vs ViT architecture.

**Outputs**:
- `figures/` — CAM overlay images
- `cam_summary.csv` — Metadata
- `arrays/` — Raw CAM arrays (with `--save-npy`)

**Find target layer**:
```bash
entomokit classify cam \
    --images-dir data/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --dump-model-structure \
    --out-dir runs/cam/
# Then check runs/cam/model_layers.txt
```

**Note**: ONNX models not supported (requires PyTorch hooks).

---

#### `classify export-onnx`

Export AutoGluon model to ONNX format for deployment.

```bash
entomokit classify export-onnx \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --out-dir runs/onnx/ \
    --opset 17
```

**With sample image for tracing**:
```bash
entomokit classify export-onnx \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --out-dir runs/onnx/ \
    --sample-image data/sample.jpg
```

**Outputs**:
- `model.onnx` — ONNX model file
- `label_classes.json` — Class label mapping

---

### Doctor Command

Diagnose environment and dependency readiness.

```bash
entomokit doctor
```

The report includes:
- Python and available devices (`cpu`, `cuda`, `mps`)
- Key package versions and status (ok/missing/outdated)
- Install/upgrade recommendations (including `autogluon.multimodal>=1.4.0`)

---

## Common Behaviours

### Logging

All commands save `log.txt` to the output directory containing:
- Full command line
- Timestamp
- All parameter values
- Runtime output

Use `--verbose` for debug-level output.

### Graceful Shutdown

Press `Ctrl+C` — the current image finishes before exiting; partial results are saved.

### Device Selection

`--device auto` chooses automatically:
1. CUDA (if available)
2. MPS / Apple Silicon (if available)
3. CPU (fallback)

### Shell Completion

Install shell completion for entomokit:

```bash
entomokit --install-completion
```

Supported shells: bash, zsh, fish

### Version

Show installed version:

```bash
entomokit --version
entomokit -v
```

---

## AI Assistant Integration (Skills)

EntomoKit includes a skill for AI assistants (OpenCode, Claude Code, Codex, etc.) that provides guided workflow orchestration for users unfamiliar with command-line tools.

### What is `entomokit-workflow` Skill?

The `entomokit-workflow` skill enables AI assistants to:
- Guide users through the complete dataset preparation pipeline
- Validate parameters and CSV files before execution
- Provide step-by-step assistance for each command
- Handle errors and suggest fixes
- Resume workflows after interruption

### Installation

**OpenCode:**

```bash
mkdir -p ~/.config/opencode/skills
cp -r skills/entomokit-workflow ~/.config/opencode/skills/
```

**Claude Code:**

```bash
mkdir -p ~/.claude/skills
cp -r skills/entomokit-workflow ~/.claude/skills/
```

**Codex:**

```bash
mkdir -p ~/.codex/skills
cp -r skills/entomokit-workflow ~/.codex/skills/
```

**Other CLI tools:** Copy the `skills/entomokit-workflow` directory to your tool's skills directory.

### Usage

Once installed, start a conversation with your AI assistant:

**Example 1 - Data cleaning and classification:**
```
I need to use entomokit-workflow skill to clean images in data/Epidorcus and train a classification model.
```

**Example 2 - Complete pipeline:**
```
Use entomokit-workflow skill to process data/my_insects: clean images, split dataset, and train a convnextv2_femto classifier.
```

**Example 3 - Resume interrupted workflow:**
```
I was running entomokit classify train yesterday. Help me continue with evaluation and ONNX export.
```

The AI will guide you through each phase, confirm parameters, and summarize results.

### Skill Features

| Feature | Description |
|---------|-------------|
| Parameter Validation | Validates all parameters against CLI schema before execution |
| CSV Teaching | Helps generate and validate `image,label` CSV files |
| Progress Tracking | Maintains session state for resumable workflows |
| Error Recovery | Maps errors to repair actions |
| Demo Mode | Optional teaching flows with repository sample data |

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- Email: `xtmtd.zf@gmail.com`

## Citation

If you use EntomoKit in your research, please cite:

```bibtex
@software{entomokit2026,
  author = {Zhang, Feng},
  title = {EntomoKit: A Python Toolkit for Insect Image Dataset Construction and Classification},
  year = {2026},
  url = {https://github.com/xtmtd/entomokit}
}
```
