# entomokit clean
[English](clean.md) | [中文](clean.cn.md)

## Purpose

Clean, resize and deduplicate images with consistent naming. Directory input is
always scanned recursively and outputs mirror each input's parent directory, with the
file name rewritten (normalised stem, unique `_N` suffix, `--out-image-format`
extension).

## Usage

Minimal MD5-deduplication run:

```bash
entomokit clean --input-dir images/raw/ --out-dir images/cleaned/
```

## Parameters

### `--input-dir`

Required. No default. Input images directory.

### `--out-dir`

Required. No default. Output directory; cleaned images are written under
`cleaned_images/`.

### `--out-short-size`

Optional. Default `512`. Target size of the shorter edge; use `-1` to keep the
original dimensions.

### `--out-image-format`

Optional. Default `jpg`. Choices: `jpg`, `png`, `tif`. Output image format for
cleaned files.

### `--dedup-mode`

Optional. Default `md5`. Choices: `none`, `md5`, `phash`, `md5+phash`.
Deduplication strategy: `none` keeps everything, `md5` removes exact duplicates,
`phash` removes perceptually similar images, and `md5+phash` runs `md5` first and
then `phash` on the survivors.

### `--phash-threshold`

Optional. Default `5`. Maximum perceptual-hash distance at which two images are
treated as duplicates.

### `--pad-color`

Optional. Default `none`. Choices: `none`, `median`, `black`, `white`. Pads
non-square images to a square with this fill colour; `none` keeps the resized
dimensions. `median` uses the median RGB of the border pixels.

### `--keep-exif`

Optional. Default off (boolean flag without a value). Preserve EXIF metadata in the
output images.

### `--threads`

Optional. Default `12`. Number of worker threads used for image processing.

### `--resume`

Optional. Default off (boolean flag without a value). Continue into a non-empty
output directory without error.

### `--overwrite`

Optional. Default off (boolean flag without a value). Delete the contents of
`--out-dir` and start fresh.

### `--verbose`, `-v`

Optional. Default off (boolean flag without a value). Enable verbose progress
output.

## Inputs

- `--input-dir` is always scanned recursively; there is no `--recursive` or
  `--flatten` flag.
- Accepted input formats follow the imaging library: `jpg`, `jpeg`, `png`, `bmp`,
  `tif`, `tiff`, `webp`.

## Outputs

- Cleaned images are written under `out-dir/cleaned_images/`, mirroring each input's
  **parent directory**; the file name itself is rewritten: the stem is normalised
  (invalid characters and whitespace become `_`, leading and trailing `.`/`_` are
  stripped, an empty stem becomes `untitled`), a case-insensitive `_1`, `_2`, ... suffix
  is added when that name is already used in the same directory, and the extension
  follows `--out-image-format`. For example `in/beetles/a.png` becomes
  `out/cleaned_images/beetles/a.jpg` with the default format, and a second
  `in/beetles/a.tif` becomes `beetles/a_1.jpg`; do not key labels by the original file
  name.
- Resizing happens first, padding second; `--pad-color none` skips the padding
  step.
- Deduplicated files are omitted from the output; no report file is written.

## Examples

Perceptual-hash dedup with a distance threshold of five:

```bash
entomokit clean --input-dir images/ --out-dir cleaned/ \
    --dedup-mode phash --phash-threshold 5
```

Resize to a 512-pixel shorter edge, convert to PNG and pad to squares with the
median border colour:

```bash
entomokit clean --input-dir images/raw/ --out-dir cleaned/ \
    --out-short-size 512 --out-image-format png --pad-color median
```

Keep original dimensions and EXIF data:

```bash
entomokit clean --input-dir images/raw/ --out-dir cleaned/ \
    --out-short-size -1 --keep-exif
```

## Notes

- Recursive discovery, the mirrored layout and output-directory safety are shared
  rules: [directory policy](../../README.md#directory-policy). Logging and version
  display are shared:
  [common behaviours](../../README.md#common-behaviours).
- Interruption: a SIGINT handler is installed, but the cleaning loop never reads the
  shutdown flag, so the first `Ctrl+C` only sets it and prints a notice and a second
  `Ctrl+C` exits; no partial-result guarantee is documented.
- A non-empty `--out-dir` stops with an error unless `--resume` or `--overwrite`
  is passed.
- `--dedup-mode md5+phash` is the strictest option: perceptual duplicates are only
  searched among images that survived the exact-hash pass.

## Version Notes

- `0.7.0`: `clean` scans recursively by default and mirrors outputs under
  `cleaned_images/` (the `--recursive` and `--flatten` flags were removed), and
  `--pad-color` was added.
