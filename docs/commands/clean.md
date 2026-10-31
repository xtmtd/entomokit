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
Deduplication strategy: `none` keeps everything, `md5` removes exact pixel-content
duplicates, `phash` removes perceptually similar images using connected groups,
and `md5+phash` runs `md5` first and then `phash` over the exact-hash group
representatives. Groups are transitive: `A~B` and `B~C` form one group even when
`A` and `C` are above the threshold.

`md5` is the MD5 of decoded pixel bytes, not the file-byte MD5; do not compare it
with `md5sum`.

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

Optional. Default `12`. Number of worker threads used for hashing inputs and
writing retained images.

### `--resume`

Optional. Default off (boolean flag without a value). Continue into a non-empty
output directory without error. Matching uses the cumulative
`clean_manifest.json["index"]` first (source hashes and input/output paths), then
falls back to hashing all existing output files for inputs the index did not
match. Inputs that match an existing output are reported as `existing` and are
not rewritten. A subset resume keeps prior index rows, so later inputs still
match. `dedup_mode="none"` only matches the index's `input_path`; without a
manifest it does not guess from mirrored output names. Without a usable manifest,
MD5 output-hash matching can miss resized/converted outputs; a warning is logged.
`--resume` does not re-encode existing outputs: changing `--out-short-size` or
`--out-image-format` requires `--overwrite`. On an already-split tree, an existing
train output can keep a duplicate out of test; use `--overwrite` or clean before
splitting to rebalance. Resume matching is per input: an input that is only
transitively similar to an existing output through another new input is retained,
because every current input is matched by its own index/output hash.

### `--overwrite`

Optional. Default off (boolean flag without a value). Delete the contents of
`--out-dir` and start fresh.

### `--dry-run`

Optional. Default off (boolean flag without a value). Write the decision reports
and `clean_manifest.json` without creating `cleaned_images/` or writing any image.
It deletes nothing and is mutually exclusive with `--overwrite`. The manifest
copies the previous valid cumulative `index` unchanged and adds no would-be outputs. A
dry run needs an empty `--out-dir` or `--resume`; because the reports and log make
the directory non-empty, a later real run into the same directory also needs
`--resume`.

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
- Deduplicated files are omitted from the output; input files are never deleted.
- Four report artifacts are written to `--out-dir` (not inside `cleaned_images/`):
  - `filter_decisions.csv`: exactly one row per input, sorted by input-relative
    path. Columns: `input_path,status,reason,kept_path,kept_scope,matched_path,matched_scope,distance,group_id,group_size,md5,phash,output_path`.
  - `retained_images.csv`: rows with status `retained` or `existing`, columns
    `input_path,status,output_path`. It is input-centric: multiple inputs may map
    to one existing output. Compute per-class retention from `output_path` (the
    actual retained image), not from `input_path`; `n_existing` counts input
    matches, not unique outputs.
  - `dropped_images.csv`: rows with status `dropped`, columns
    `input_path,reason,matched_path,matched_scope,distance,group_id`.
  - `clean_manifest.json`: run metadata, counts, report paths, a `files` array
    identical to the `filter_decisions.csv` rows, and a cumulative `index` array.
    `files` is current-run only and is not the resume index. Each `index` row has
    `input_path`, `output_path`, `status` (`retained`/`existing`/`dropped`),
    `md5`, and `phash`; dropped rows map to the surviving output and keep their own
    source hashes. `--dry-run` copies the previous valid `index` unchanged and adds no
    would-be outputs.
- All input paths are relative to `--input-dir`; output paths and existing-output
  references are relative to `--out-dir`. `kept_scope`/`matched_scope` say whether
  a path is an input path or an existing output.
- Grouping is deterministic. A group is a connected component; its representative
  is the member with the best held-out priority (`test` > `val` > `train`), then
  the lexicographically smallest input-relative path. Groups without a recognized
  split segment use path order only. `md5+phash` groups by MD5 first, then by
  perceptual hash over the MD5-group representatives.
- `md5` in the reports is the decoded-pixel MD5, not the file-byte MD5.
- The final per-file `log.txt` lines are emitted from the same finalized decision
  rows after writing, so they cannot disagree with the reports.
- Reports are written atomically; `log.txt` is appended across runs.

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
- `clean` never deletes input images. `drop` means "not written to
  `cleaned_images/`"; only `--overwrite` removes output-directory contents.
- The report table is the structured source of truth; `log.txt` keeps readable
  per-file lines and the summary but is not the only record.
- `--dry-run` is a report-only preview. After it, use `--resume` for a real run
  into the same `--out-dir`.

## Version Notes

- `0.7.2`: deduplication now uses deterministic connected groups and writes
  `filter_decisions.csv`, `retained_images.csv`, `dropped_images.csv`, and
  `clean_manifest.json` with a cumulative resume `index`; `--dry-run` was added;
  per-file paths and hashes are recorded; `--resume` matches the cumulative index
  first and reports `existing` rows. Transitive groups can merge more images than
  the previous pairwise behavior.
- `0.7.0`: `clean` scans recursively by default and mirrors outputs under
  `cleaned_images/` (the `--recursive` and `--flatten` flags were removed), and
  `--pad-color` was added.
