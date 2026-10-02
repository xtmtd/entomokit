# entomokit classify train
[English](classify-train.md) | [中文](classify-train.cn.md)

## Purpose

Train an image classifier with AutoGluon MultiModalPredictor on a CSV-labelled
image dataset, using a timm backbone. Requires the `classify` extras.

## Usage

```bash
entomokit classify train --train-csv data/train.csv --images-dir data/images/ \
    --out-dir runs/exp1/
```

## Parameters

### `--train-csv`

Required. No default. CSV with `image` and `label` columns.

### `--images-dir`

Required. No default. Directory containing the training images.

### `--out-dir`

Required. No default. Output directory; the AutoGluon predictor is written to
`out-dir/AutogluonModels/<base-model>`.

### `--base-model`

Optional. Default `convnextv2_femto`. timm backbone name.

### `--augment`

Optional. Default `medium`. Data augmentation preset or a JSON array of AutoGluon
transform names. Presets: `none` = `resize_shorter_side`, `center_crop`; `light` =
`none` plus `random_horizontal_flip`; `medium` = `light` plus `color_jitter`,
`trivial_augment`; `heavy` = `random_resize_crop`, `random_horizontal_flip`,
`random_vertical_flip`, `color_jitter`, `trivial_augment`, `randaug`. Custom
example: `'["random_resize_crop","color_jitter","randaug"]'`.

### `--max-epochs`

Optional. Default `50`. Maximum number of training epochs.

### `--time-limit`

Optional. Default `1.0`. Training time limit in hours.

### `--resume`

Optional. Default off (boolean flag without a value). Resume an existing AutoGluon
run from the checkpoint in `--out-dir/AutogluonModels/<base-model>`; combined with a
larger `--max-epochs` it continues training to the new epoch limit.

### `--overwrite`

Optional. Default off (boolean flag without a value). Delete the contents of
`--out-dir` and train from scratch.

### `--learning-rate`

Optional. Default `0.0001`. Optimisation learning rate (`optim.lr`).

### `--weight-decay`

Optional. Default `0.001`. Optimiser weight decay (`optim.weight_decay`).

### `--warmup-steps`

Optional. Default `0.1`. Learning-rate warmup proportion or steps
(`optim.warmup_steps`).

### `--patience`

Optional. Default `10`. Early-stopping patience checks (`optim.patience`).

### `--top-k`

Optional. Default `3`. Number of checkpoints used for model averaging
(`optim.top_k`).

### `--focal-loss`

Optional. Default off (boolean flag without a value). Use focal loss to emphasise
hard examples, which helps with imbalanced classes.

### `--focal-loss-gamma`

Optional. Default `1.0`. Gamma value for focal-loss weighting.

### `--device`

Optional. Default `auto`. Choices: `auto`, `cpu`, `cuda`, `mps`. Compute device for
training.

### `--batch-size`

Optional. Default `32`. Mini-batch size used during training.

### `--num-workers`

Optional. Default `4`. Number of dataloader worker processes.

### `--num-threads`

Optional. Default `0`. CPU threads for PyTorch; `0` selects automatically.

### `--seed`

Optional. Default `0`. Random seed passed to AutoMM for reproducible training.

## Inputs

- `--train-csv` needs `image` and `label` columns; image paths are resolved against
  `--images-dir`.
- The `classify` extras must be installed (`pip install -e ".[classify]"`).
- Model preparation: AutoMM downloads the backbone weights automatically on first
  use, so no manual checkpoint step is needed. `--base-model` takes a timm backbone
  name; `convnextv2_femto` is the default and other common choices are
  `convnextv2_tiny`, `convnextv2_small`, `convnextv2_base`, `resnet18`, `resnet50`,
  `resnet101`, `efficientnet_b0`-`efficientnet_b7`, `vit_small_patch16_224` and
  `vit_base_patch16_224`; any other timm name works too
  (https://huggingface.co/timm).
- Training reads the labelled training split only; when the data comes from
  `entomokit split-csv`, keep `--val-ratio`/`--val-count` at `0` because AutoGluon
  performs its own internal train/validation split.

## Outputs

- The trained predictor is written to `out-dir/AutogluonModels/<base-model>`, which
  is the `--model-dir` consumed by `classify predict`, `classify evaluate`,
  `classify embed`, `classify cam` and `classify export-onnx`.
- Run logs are written to `out-dir/logs/log.txt`, and the resolved training table is
  saved to `out-dir/train.processed.csv`.

## Examples

Train with a larger epoch budget and a lower learning rate:

```bash
entomokit classify train --train-csv data/train.csv --images-dir data/images/ \
    --out-dir runs/exp1/ --max-epochs 100 --learning-rate 3e-4 --device auto
```

Extend an existing run from 50 to 100 epochs:

```bash
entomokit classify train --train-csv data/train.csv --images-dir data/images/ \
    --out-dir runs/exp1/ --base-model convnextv2_femto --max-epochs 100 --resume
```

Custom augmentation and focal loss for imbalanced classes:

```bash
entomokit classify train --train-csv data/train.csv --images-dir data/images/ \
    --out-dir runs/exp1/ --augment '["random_resize_crop","color_jitter","randaug"]' \
    --focal-loss --focal-loss-gamma 2.0
```

## Notes

- `classify train` is CSV-driven and therefore outside the README's directory
  input/output policy: [directory policy](../../README.md#directory-policy).
  Logging, device selection and version display are shared:
  [common behaviours](../../README.md#common-behaviours); classification commands
  install no SIGINT handler.
- `--resume` only makes sense with the same `--base-model`; changing the backbone
  starts a different checkpoint directory.
- `--time-limit` and `--max-epochs` both bound training; whichever is reached first
  stops the run.
- A fixed `--seed` makes the run reproducible for the same data and settings.

## Version Notes

- `0.7.0`: `--seed` was added for reproducible AutoMM training.
