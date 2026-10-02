# entomokit split-csv
[English](split-csv.md) | [中文](split-csv.cn.md)

## Purpose

Split a labelled CSV into train, validation and test files, optionally reserving an
unknown-class test split and copying the images into per-split directories.

## Usage

Ratio split with a validation and a known-class test split:

```bash
entomokit split-csv --raw-image-csv data/images.csv \
    --known-test-sample-ratio 0.1 --val-ratio 0.1 --out-dir datasets/
```

## Parameters

### `--raw-image-csv`

Required. No default. Input CSV with `image` and `label` columns.

### `--mode`

Optional. Default `ratio`. Choices: `ratio`, `count`. Split strategy: sample ratios
or explicit sample counts.

### `--known-test-sample-ratio`

Optional. Default `0.1`. Fraction of known samples moved to the test split in
`ratio` mode.

### `--unknown-test-sample-ratio`

Optional. Default `0.0`. Target fraction of samples reserved for the unknown test
split in `ratio` mode; `0` writes no unknown split. Classes are shuffled and whole
classes are moved into the unknown split until the accumulated sample count reaches
this target, so the actual split can exceed it and is class-disjoint from the known
data.

### `--known-test-sample-count`

Optional. Default `0`. Target number of known samples moved to the test split in
`count` mode.

### `--unknown-test-sample-count`

Optional. Default `0`. Target number of samples reserved for the unknown test split
in `count` mode. As in `ratio` mode, whole classes are moved until the accumulated
count reaches this target, so the split can exceed it and never splits a class.

### `--val-ratio`

Optional. Default `0.0`. Validation split ratio taken from the training split; `0`
writes no validation split.

### `--val-count`

Optional. Default `0`. Validation split sample count taken from the training split;
`0` writes no validation split.

### `--min-count-per-class`

Optional. Default `0`. In `--mode count` only, drop classes whose remaining samples
are fewer than this number.

### `--max-count-per-class`

Optional. No default; no per-class cap is applied. In `--mode count` only, keep at
most this many samples per class from the remaining training data.

### `--seed`

Optional. Default `42`. Random seed for reproducible splits.

### `--out-dir`

Optional. Default `datasets`. Directory for the split CSV files and any copied
images.

### `--images-dir`

Optional. No default. Source image directory used only by `--copy-images`; required
when that flag is set. It never resolves or rewrites the CSV `image` values.

### `--copy-images`

Optional. Default off (boolean flag without a value). Copy the images into
`out_dir/images/{split}/` subdirectories. Only the file name is kept — the source
parent directory is dropped — and there is no collision check, so a `beetles/a.jpg`
and a `flies/a.jpg` that land in the same split both target
`images/<split>/a.jpg` and the later copy overwrites the earlier one. Basenames must
therefore be unique within each split; renaming or checking collisions would be a
code change and needs separate authorization.

### `--overwrite`

Optional. Default off (boolean flag without a value). Delete the contents of
`--out-dir` and regenerate all splits.

### `--verbose`, `-v`

Optional. Default off (boolean flag without a value). Enable verbose output during
splitting.

## Inputs

- `--raw-image-csv` must contain `image` and `label` columns. The `image` values are
  copied into the split CSVs exactly as written; `--images-dir` is only the copy
  source for `--copy-images` and never resolves or rewrites those values.
- `--mode ratio` uses the `*-ratio` options; `--mode count` uses the `*-count`
  options.
- `--copy-images` requires `--images-dir`.
- Keep `--val-ratio` and `--val-count` at `0` when the output feeds
  `entomokit classify train`; AutoGluon performs its own internal train/validation
  split.

## Outputs

```text
out_dir/
├── train.csv
├── val.csv               # only when --val-ratio or --val-count is above 0
├── test.known.csv
├── test.unknown.csv      # only when an unknown split is configured
├── class_count/          # per-split class counts
│   ├── class.train.count
│   ├── class.val.count
│   └── ...
└── images/               # only with --copy-images
    ├── train/
    ├── val/
    ├── test_known/
    └── test_unknown/
```

## Examples

Count split with copied images:

```bash
entomokit split-csv --raw-image-csv data/images.csv --mode count \
    --known-test-sample-count 100 --val-count 50 \
    --copy-images --images-dir images/ --out-dir datasets/
```

Unknown-class test split for open-set evaluation:

```bash
entomokit split-csv --raw-image-csv data/images.csv \
    --unknown-test-sample-ratio 0.1 --known-test-sample-ratio 0.1 \
    --out-dir datasets/
```

Drop classes with too few samples (count mode):

```bash
entomokit split-csv --raw-image-csv data/images.csv --mode count \
    --min-count-per-class 10 --out-dir datasets/
```

## Notes

- `split-csv` is CSV-driven and therefore outside the README's directory
  input/output policy: [directory policy](../../README.md#directory-policy).
  Logging and version display are shared:
  [common behaviours](../../README.md#common-behaviours).
- `--min-count-per-class` and `--max-count-per-class` are only applied by
  `--mode count`; the default `ratio` branch ignores them.
- With `--copy-images` the copies are flattened to the file name (see the parameter
  note): two inputs whose basenames match inside one split silently overwrite each
  other.
- The copied tree is not a self-contained relocatable dataset: the split CSVs keep
  the original `image` values, so `beetles/a.jpg` copied to `images/train/a.jpg` is
  still recorded as `beetles/a.jpg`. Pointing a later command's image root at
  `out_dir/images/` looks for `images/train/beetles/a.jpg` and fails; rewrite the CSV
  or keep using the original `--images-dir`.
- A fixed `--seed` reproduces the same split for the same input CSV.
- A non-empty `--out-dir` stops with an error unless `--overwrite` is passed;
  `split-csv` has no `--resume` and never reuses an existing output directory.
- Interruption: a SIGINT handler is installed, but the splitter never reads the
  shutdown flag, so the first `Ctrl+C` only sets it and prints a notice and a second
  `Ctrl+C` exits.
