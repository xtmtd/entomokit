# entomokit segment
[English](segment.md) | [中文](segment.cn.md)

## Purpose

Segment insects from images and write bounding-box or segmentation annotations
(COCO by default). Six segmentation methods are available: SAM3, Otsu, GrabCut, each with
a bbox-crop variant. Annotation granularity follows the method: methods ending in
`-bbox` emit bounding-box-only annotations, the others emit bounding boxes and
segmentation data.

## Usage

Minimal Otsu run with the default COCO annotations:

```bash
entomokit segment --input-dir images/ --out-dir out/ --segmentation-method otsu
```

SAM3 requires a checkpoint:

```bash
entomokit segment --input-dir images/ --out-dir out/ \
    --segmentation-method sam3 --sam3-checkpoint checkpoints/sam3.pt \
    --annotation-format coco
```

## Parameters

### `--input-dir`, `-i`

Required. No default. Input images directory, scanned recursively.

### `--out-dir`, `-o`

Required. No default. Output directory. Segmented images are flattened into
`images/` under this directory; annotation files go to the per-format directories
described in Outputs.

### `--segmentation-method`

Optional. Default `sam3`. Choices: `sam3`, `sam3-bbox`, `otsu`, `otsu-bbox`,
`grabcut`, `grabcut-bbox`. Methods ending in `-bbox` output bounding-box-only
annotations; the other methods output both bounding-box and segmentation
annotations.

### `--sam3-checkpoint`, `-c`

Optional. No default. Path to the SAM3 checkpoint file; required when
`--segmentation-method` is `sam3` or `sam3-bbox`.

### `--hint`, `-t`

Optional. Default `insect`. Text prompt used for SAM3 grounding.

### `--device`, `-d`

Optional. Default `auto`. Choices: `auto`, `cpu`, `cuda`, `mps`. Device used for
inference; `auto` selects the best available device.

### `--confidence-threshold`

Optional. Default `0.0`. Minimum confidence score for accepted masks; `0.0`
disables filtering.

### `--padding-ratio`

Optional. Default `0.0`. Bounding-box padding ratio; `0.0` adds no padding.

### `--repair-strategy`, `-r`

Optional. No default; no repair is performed. Choices: `opencv`, `sam3-fill`,
`black-mask`, `lama`. Applied to the source image: the union of the detected
foreground masks is inpainted (`opencv`, `sam3-fill`, `lama`) or filled with black
(`black-mask`), and the result is written to `repaired_images/` under the same encoded
sample ID and output format as the segmented image. The produced masks and annotation
files are not modified. The option help
describes this as "filling holes", which does not match the implemented behavior.

### `--lama-model`

Optional. No default. Path to the LaMa model checkpoint directory; required when
`--repair-strategy lama` is used.

### `--lama-mask-dilate`

Optional. Default `0`. Number of dilation iterations applied to the LaMa repair
mask.

### `--out-image-format`, `-f`

Optional. Default `png`. Choices: `png`, `jpg`. Output image format.

### `--threads`, `-n`

Optional. Default `8`. Concurrent image workers used by the Otsu and GrabCut
methods; SAM3 and SAM3-bbox keep a single stateful predictor and remain serial on
every device.

### `--annotation-format`

Optional. No default value, but the CLI substitutes `coco`, so omitting this option
still writes COCO annotations; there is no way to disable annotation output with this
flag. Choices: `coco`, `voc`, `yolo`. Controls the annotation file encoding only;
annotation granularity follows `--segmentation-method` (VOC segmentation is stored as
mask PNG files). The option help says `None = no annotations`, which does not match
this behavior.

### `--coco-output-mode`

Optional. Default `unified`. Choices: `unified`, `separate`. COCO layout:
`unified` writes one `annotations.coco.json`, `separate` writes one JSON file per
image.

### `--coco-bbox-format`

Optional. Default `xywh`. Choices: `xywh`, `xyxy`. COCO bounding-box coordinate
convention; only used when `--annotation-format coco`.

### `--resume`

Optional. Default off (boolean flag without a value). Continue a previous run:
inputs whose exact single-mask output already exists are skipped, while multi-mask
inputs (`{sample_id}_01.png`, `_02`, ...) are always re-processed so a partially
written set is never treated as complete. With a unified COCO output the previous
`annotations.coco.json` is merged, so skipped samples keep their annotations.

### `--overwrite`

Optional. Default off (boolean flag without a value). Delete the contents of
`--out-dir` and start fresh.

### `--verbose`, `-v`

Optional. Default off (boolean flag without a value). Enable verbose logging.

## Inputs

- `--input-dir` is scanned recursively. Images for which the method returns no mask
  are logged with their full source path and listed in `out-dir/no_mask_images.txt`.
- `sam3` and `sam3-bbox` require `--sam3-checkpoint`; SAM3 also uses `--hint` as its
  grounding prompt. The checkpoint is not downloaded automatically: fetch it from
  https://huggingface.co/facebook/sam3 and pass the file path.
- `lama` repair requires `--lama-model` pointing at a Big-LaMa checkpoint directory
  with the weights placed under `models/`:

  ```text
  models/big-lama/
  ├── config.yaml
  └── models/best.ckpt
  ```

  Download: https://github.com/advimman/lama. `opencv`, `sam3-fill` and `black-mask`
  do not need a model file.
- SAM3 runs on `--device`; the Otsu and GrabCut methods run on CPU workers.

## Outputs

Segmented images are always flattened into `images/`; the input-relative path
becomes a unique, filesystem-safe sample ID (readable stem plus a short path
digest, for example `a__4cabcf2b3682`). That ID names the image file, VOC XML,
YOLO TXT, SegmentationClass mask, COCO file name and `ImageSets/Main/default.txt`
entry, so same-named images in different subdirectories stay distinct.

```text
out_dir/
├── images/                    # segmented images
├── repaired_images/           # only when a repair strategy is enabled
├── annotations.coco.json      # --annotation-format coco, --coco-output-mode unified
├── annotations/               # --annotation-format coco, --coco-output-mode separate
├── Annotations/               # --annotation-format voc: .xml per image
├── SegmentationClass/         # voc without -bbox: mask PNG, foreground=255, background=0
├── ImageSets/Main/default.txt # voc: sample list
├── labels/                    # --annotation-format yolo: .txt per image
├── data.yaml                  # --annotation-format yolo: class list (output root)
└── no_mask_images.txt         # images with no mask from the chosen method
```

Standard dataset layouts (for example Pascal VOC `JPEGImages/`) are produced by a
later conversion or split step, not by `segment`.

### Annotation field semantics

For methods without the `-bbox` suffix, `area` is the mask pixel area
(`np.sum(mask > 0)`) and `segmentation` holds polygon coordinates
(`[x1,y1,x2,y2,...]`). For `-bbox` methods, `area` is the bounding-box area
(`w × h`) and `segmentation` is empty.

## Examples

GrabCut bbox crops with COCO annotations in `xyxy` convention:

```bash
entomokit segment --input-dir images/ --out-dir out/ \
    --segmentation-method grabcut-bbox --annotation-format coco \
    --coco-bbox-format xyxy
```

SAM3 segmentation with LaMa repair and VOC annotations:

```bash
entomokit segment --input-dir images/ --out-dir out/ \
    --segmentation-method sam3 --sam3-checkpoint checkpoints/sam3.pt \
    --repair-strategy lama --lama-model models/lama \
    --annotation-format voc
```

Continue a previous run and keep its annotations:

```bash
entomokit segment --input-dir images/ --out-dir out/ --resume
```

## Notes

- Directory scanning, the flat sample-ID exception and output safety are shared
  rules: [directory policy](../../README.md#directory-policy). Logging, device
  selection and version display are shared too:
  [common behaviours](../../README.md#common-behaviours).
- Interruption: the shutdown flag is checked between images. The serial path stops
  after the current image; the parallel Otsu/GrabCut path submits all of its
  computation tasks up front, so images already queued still finish before the run
  stops.
- `--resume` trusts only an exact single-mask output as completion; multi-mask
  inputs are always re-processed because the mask count is not recorded.
- A non-empty `--out-dir` stops with an error unless `--resume` (continue) or
  `--overwrite` (fresh start) is passed; `--overwrite` deletes the existing
  contents before the run.
- `--annotation-format` selects the encoding; whether bbox and segmentation data
  are written is decided by `--segmentation-method`.

## Version Notes

- `0.7.0`: `--input-dir` is scanned recursively, images are flattened into
  `images/` with encoded sample IDs instead of mirrored paths, and `--resume`
  merges the unified COCO annotations of the previous run.
