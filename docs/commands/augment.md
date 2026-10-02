# entomokit augment
[English](augment.md) | [中文](augment.cn.md)

## Purpose

Augment images with an albumentations preset or a custom policy file. Input
directories are scanned recursively and every output mirrors the input's relative
path.

## Usage

Light preset with one output per input image:

```bash
entomokit augment --input-dir images/cleaned/ --out-dir images/augmented/
```

## Parameters

### `--input-dir`

Required. No default. Input directory containing images. Accepted formats are
`jpg`/`jpeg`, `png`, `bmp`, `tif`/`tiff` and `webp`; the output format matches the
input format with no conversion. Use `entomokit clean --out-image-format` to
convert formats.

### `--out-dir`

Required. No default. Output directory for the augmented images and the manifest.

### `--preset`

Optional. No default; when omitted with no `--policy`, the `light` preset is used.
Named preset: `light`, `medium`, `heavy` or `safe-for-small-dataset`.

### `--policy`

Optional. No default. Path to a custom augmentation policy JSON file, mutually
exclusive with `--preset`. The file is read with `json.loads`, so it must be a JSON
**object** with a `transforms` array; each entry is an object whose `name` is an
albumentations class and whose remaining keys are that class's constructor
arguments, for example
`{"transforms": [{"name": "HorizontalFlip", "p": 0.5}]}`.

### `--seed`

Optional. Default `42`. Random seed for reproducible augmentation.

### `--multiply`

Optional. Default `1`. Number of augmented copies created per input image.

### `--resume`

Optional. Default off (boolean flag without a value). Skip a source image only when
its existing `_aug<digits>` file set exactly equals the current `--multiply` set. An
incomplete or differently sized set is regenerated, and leftover copies from a
previous larger `--multiply` are deleted. There is no `--seed` or `--policy`
consistency check, so an exact file-set match is skipped even when those changed.

### `--overwrite`

Optional. Default off (boolean flag without a value). Delete the contents of
`--out-dir` and start fresh.

## Inputs

- `--input-dir` is scanned recursively.
- `--preset` and `--policy` are mutually exclusive; pass at most one.
- A custom policy file must be a JSON object shaped like
  `{"transforms": [{"name": "<albumentations-class>", ...}]}`. A bare JSON array is
  rejected by the loader; unknown transform names fail the run before any image is
  written. Arguments are passed through to the albumentations class, so they must
  match the installed version's signature (for example `RandomResizedCrop` takes
  `size`, not `height`/`width`, in albumentations 2.x).

## Outputs

```text
out_dir/
├── images/                 # mirrors each input's subdirectory path
│   └── <subdir>/source_aug1.png
└── augment_manifest.json
```

- Every output keeps an `_augN` suffix, even with `--multiply 1`, so an original
  basename is never overwritten. The index is zero-padded to the width of
  `--multiply`: `1`–`9` give `_aug1`…`_aug9`, `10`–`99` give `_aug01`…`_aug99`, and
  `100` gives `_aug001`.
- `augment_manifest.json` records `preset`, `multiply`, `seed`, `images_processed` and
  `augmented_images_created`. Per-image `original`/`augmented` path records are built in
  memory but are **not** serialised into the file today, so do not rely on the manifest
  as a path mapping.
- The output format always matches the input format; `augment` never converts.

## Examples

Heavy preset with three copies per image and a fixed seed:

```bash
entomokit augment --input-dir images/cleaned/ --out-dir images/augmented/ \
    --preset heavy --multiply 3 --seed 123
```

Custom policy file `configs/augment_policy.json`:

```json
{"transforms": [
  {"name": "RandomResizedCrop", "size": [512, 512]},
  {"name": "HorizontalFlip", "p": 0.5}
]}
```

```bash
entomokit augment --input-dir images/cleaned/ --out-dir images/augmented/ \
    --policy configs/augment_policy.json
```

## Notes

- Recursive discovery, the mirrored layout and output-directory safety are shared
  rules: [directory policy](../../README.md#directory-policy). Logging,
  interruption and version display are shared:
  [common behaviours](../../README.md#common-behaviours).
- A non-empty `--out-dir` stops with an error unless `--resume` or `--overwrite`
  is passed.
- `safe-for-small-dataset` keeps augmentations conservative for tiny training sets.
- Requires the `augment` extra; on some platforms the binary `stringzilla` wheel
  must be installed first, as described in the README installation section.

## Version Notes

- `0.7.0`: `augment` scans `--input-dir` recursively and mirrors each input's
  relative path under `images/` instead of flattening the outputs.
