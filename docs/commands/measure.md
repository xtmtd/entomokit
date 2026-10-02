# entomokit measure
[English](measure.md) | [中文](measure.cn.md)

## Purpose

Measure morphology metrics from segmentation masks in bulk and export CSV reports.
Metric definitions align with scikit-image `regionprops` so results are reproducible
and comparable across tools.

## Usage

Minimal run: produce masks with `segment`, then measure them.

```bash
entomokit segment --input-dir images/ --out-dir segmented/ \
    --segmentation-method otsu --annotation-format voc
entomokit measure --mask-dir segmented/SegmentationClass --out-dir runs/measure
```

With a calibrated scale:

```bash
entomokit measure --mask-dir segmented/SegmentationClass --out-dir runs/measure \
    --pixel-size-um 2.5
```

## Parameters

### `--mask-dir`, `-i`

Required. No default. Directory of binary mask images, scanned recursively.
Each image is read as a binary mask: a 3D image contributes its **first channel**
only, binarised at `> 0`, and an alpha channel is ignored.

### `--out-dir`, `-o`

Required. No default. Output directory for the CSV reports.

### `--pixel-size-um`

Optional. No default; measurements stay in pixels and the physical-unit columns are
not derived. Pixel size in micrometers per pixel (`um/px`).

### `--resume`

Optional. Default off (boolean flag without a value). Append measurements for new
masks and skip masks already present in `metrics.csv`. The output-affecting
parameters (currently `--pixel-size-um`) are recorded in
`out-dir/.entomokit/measure_params.json` and must match the previous run; a mismatch
exits with an error instead of mixing measurements taken at different scales.

### `--overwrite`

Optional. Default off (boolean flag without a value). Delete the contents of
`--out-dir` and start fresh.

### `--verbose`, `-v`

Optional. Default off (boolean flag without a value). Enable verbose logging.

## Inputs

- `--mask-dir` is scanned recursively; every mask image found there is measured.
- The directory must contain binary masks: either the `SegmentationClass/*.png`
  files written by `segment --annotation-format voc` (foreground 255, background 0)
  or your own mask images. Do **not** point `--mask-dir` at `segment`'s `images/`
  directory: those are RGB/RGBA crops, and `measure` binarises only their first
  colour channel, so a dark specimen with a correct alpha channel can measure as an
  empty mask.
- Only the largest connected component of each mask is measured; every other
  component is discarded before metrics are computed (connectivity 2). Fragmented
  specimens and masks holding several objects are therefore not measured as their
  combined foreground — split or merge such masks first if that matters.
- Supplying `--pixel-size-um` enables the physical-unit metrics; without it the
  reports stay in pixel units.

## Outputs

```text
out_dir/
├── metrics.csv              # per-image metrics and warning reasons
├── metrics_summary.csv      # aggregated statistics plus warning counts
└── metric_definitions.csv   # metric definitions (zh/en) with units and formulas
```

`file_name` in `metrics.csv` is the mask path relative to `--mask-dir` (for
example `beetles/a.png`), which keeps nested same-named masks distinct and is the
key used by `--resume`.

## Examples

Measure masks with a calibrated scale and verbose logging:

```bash
entomokit measure --mask-dir segmented/SegmentationClass --out-dir runs/measure \
    --pixel-size-um 2.5 --verbose
```

Add measurements for masks that arrived after a previous run, keeping the original
scale:

```bash
entomokit measure --mask-dir segmented/SegmentationClass --out-dir runs/measure \
    --pixel-size-um 2.5 --resume
```

## Notes

- Recursive scanning, the mirrored output layout and output-directory safety are
  shared rules: [directory policy](../../README.md#directory-policy). Logging,
  interruption and version display are shared too:
  [common behaviours](../../README.md#common-behaviours).
- Caution on body length and width:
  - `body_length_*` and `body_width_*` are geometry-based estimates from binary
    masks, not direct anatomical measurements.
  - They can be biased when masks include appendages (antennae, legs), are clipped
    by image borders, or contain merged or fragmented body regions.
  - Check `quality_flag` and `warn_reason` (for example `touching_border`,
    `too_many_branches`) before downstream analysis.
- A non-empty `--out-dir` stops with an error unless `--resume` (append to the
  existing reports) or `--overwrite` (fresh start) is passed.
