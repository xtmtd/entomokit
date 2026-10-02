# entomokit synthesize
[English](synthesize.md) | [中文](synthesize.cn.md)

## Purpose

Composite RGBA target cutouts onto background images with rotation, colour
matching and area-ratio placement, and optionally write bounding-box annotations
for the composited objects.

## Usage

Minimal run with ten syntheses per target:

```bash
entomokit synthesize --target-dir images/targets/ --background-dir images/backgrounds/ \
    --out-dir outputs/synthesized/ --num-syntheses 10
```

## Parameters

### `--target-dir`, `-t`

Required. No default. Directory of target object images; they must be RGBA cutouts
with an alpha channel.

### `--background-dir`, `-b`

Required. No default. Directory of background images.

### `--out-dir`, `-o`

Required. No default. Output directory.

### `--num-syntheses`, `-n`

Optional. Default `1`. A positive integer is the number of syntheses per target,
with backgrounds sampled with replacement and no background-count cap. A fraction
between `0` and `1` selects that share of targets and produces one synthesis each.

### `--seed`

Optional. Default `42`. Base random seed for target sampling and task-local
synthesis randomness; a fixed seed makes fractional selection deterministic.

### `--area-ratio-min`, `-a`

Optional. Default `0.05`. Minimum area ratio (target area / background area);
accepted range `0.01`-`0.50`.

### `--area-ratio-max`, `-x`

Optional. Default `0.2`. Maximum area ratio (target area / background area);
accepted range `0.01`-`0.50`.

### `--color-match-strength`, `-c`

Optional. Default `0.5`. Strength of colour matching between target and
background, from `0` to `1`.

### `--avoid-black-regions`, `-A`

Optional. Default off (boolean flag without a value). Avoid compositing onto pure
black regions of the background.

### `--rotate`, `-r`

Optional. Default `0.0`. Maximum random rotation in degrees; `0` disables
rotation.

### `--out-image-format`, `-f`

Optional. Default `png`. Choices: `png`, `jpg`. Output image format.

### `--annotation-output-format`

Optional. Default `coco`. Choices: `coco`, `voc`, `yolo`. Annotation output
format for the composited objects.

### `--coco-output-mode`

Optional. Default `unified`. Choices: `unified`, `separate`. Accepted for CLI
compatibility, but the current implementation always accumulates the unified layout
and writes a single `annotations.coco.json`; `separate` does **not** yet produce
per-image JSON files.

### `--coco-bbox-format`

Optional. Default `xywh`. Choices: `xywh`, `xyxy`. COCO bounding-box coordinate
convention; only used with `--annotation-output-format coco`.

### `--threads`, `-d`

Optional. Default `4`. Number of parallel workers.

### `--resume`

Optional. Default off (boolean flag without a value). Skip a target only when its
existing output indices already equal the current `--num-syntheses` set; the only
additional check is `--coco-bbox-format` against the recorded COCO file. No `--seed`,
rotation or colour parameter is compared, so a target whose index set matches is
skipped even when those changed. An incomplete set or a changed `--num-syntheses`
re-synthesises the target, and for targets that produced a new result the leftover
copies and their annotations from a larger previous run are deleted. With unified COCO
output the previous `annotations.coco.json` is merged, so skipped targets keep their
annotations.

### `--overwrite`

Optional. Default off (boolean flag without a value). Delete the contents of
`--out-dir` and start fresh.

### `--verbose`, `-v`

Optional. Default off (boolean flag without a value). Enable verbose logging.

## Inputs

- `--target-dir` must contain RGBA cutouts, for example mask-mode `segment`
  output (`--segmentation-method sam3`, `otsu` or `grabcut`). Bbox-mode crops
  (`sam3-bbox`, `otsu-bbox`, `grabcut-bbox`), repaired images and raw photos are
  RGB and are rejected; when every target fails the error lists the observed modes
  (for example `18 RGB`).
- Both directories are scanned recursively.
- Backgrounds are sampled with replacement per synthesis task, so the number of
  syntheses per target is not capped by the number of backgrounds.

## Outputs

```text
out_dir/
├── images/                  # mirrors each target's subdirectory path
│   └── <subdir>/target_01.png
├── annotations.coco.json    # --annotation-output-format coco (always unified today)
├── Annotations/             # --annotation-output-format voc: .xml per image
├── labels/                  # --annotation-output-format yolo: .txt per image
└── data.yaml                # --annotation-output-format yolo: class list
```

- Output files are named `{target_stem}_{NN}`. Two targets in the same directory
  that differ only by extension (for example `a.png` and `a.tif`) receive a short
  path digest in the stem so neither overwrites the other.
- Annotations mirror the same target-relative paths as the images.
- Known discrepancy: `--coco-output-mode separate` is accepted and documented in the
  CLI help, but the COCO dispatch accumulates every image into the unified writer
  (`_accumulate_coco_single`) regardless of the mode, so only `annotations.coco.json`
  is produced today. The option is kept for forward compatibility; changing the CLI
  help wording needs its own review.

## Examples

COCO annotations with up to 30 degrees of rotation:

```bash
entomokit synthesize --target-dir images/targets/ --background-dir images/backgrounds/ \
    --out-dir outputs/synthesized/ --num-syntheses 10 \
    --annotation-output-format coco --rotate 30
```

YOLO annotations with stronger colour matching and black-region avoidance:

```bash
entomokit synthesize --target-dir images/targets/ --background-dir images/backgrounds/ \
    --out-dir outputs/synthesized/ --annotation-output-format yolo \
    --avoid-black-regions --color-match-strength 0.7
```

## Notes

- Recursive discovery, the mirrored layout and output-directory safety are shared
  rules: [directory policy](../../README.md#directory-policy). Logging and version
  display are shared:
  [common behaviours](../../README.md#common-behaviours).
- Interruption: the shutdown flag is checked while tasks are prepared, not inside the
  synthesis loop, so the tasks already prepared run to completion after `Ctrl+C`.
- A non-empty `--out-dir` stops with an error unless `--resume` or `--overwrite`
  is passed.
- A fractional `--num-syntheses` with a fixed `--seed` reproduces the same target
  selection across runs.

## Version Notes

- `0.7.0`: `--num-syntheses` gained the fractional count semantics and `--seed`
  was added for reproducible target sampling.
