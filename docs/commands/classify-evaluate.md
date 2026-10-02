# entomokit classify evaluate
[English](classify-evaluate.md) | [中文](classify-evaluate.cn.md)

## Purpose

Evaluate a trained classifier on a labelled test CSV and export overall metrics plus
per-class and confusion-matrix diagnostics.

## Usage

```bash
entomokit classify evaluate --test-csv data/test.csv --images-dir data/images/ \
    --onnx-model runs/onnx/model.onnx --out-dir runs/eval/
```

## Parameters

### `--test-csv`

Required. No default. CSV with `image` and `label` columns for evaluation.

### `--images-dir`

Required. No default. Directory containing the evaluation images.

### `--model-dir`

Optional. No default. AutoGluon predictor directory for evaluation.

### `--onnx-model`

Optional. No default. ONNX model file path; requires `onnxruntime`. Mutually
exclusive with `--model-dir`.

### `--out-dir`

Required. No default. Directory to write evaluation logs and metrics.

### `--batch-size`

Optional. Default `32`. Batch size for evaluation inference.

### `--num-workers`

Optional. Default `4`. Number of dataloader worker processes.

### `--num-threads`

Optional. Default `0`. CPU threads for PyTorch or the ONNX runtime; `0` selects
automatically.

### `--overwrite`

Optional. Default off (boolean flag without a value). Delete the contents of
`--out-dir` and re-evaluate.

### `--device`

Optional. Default `auto`. Choices: `auto`, `cpu`, `cuda`, `mps`. Compute device for
evaluation.

## Inputs

- `--test-csv` must contain `image` and `label` columns; image paths are resolved
  against `--images-dir`.
- One of `--model-dir` or `--onnx-model` is required.
- Evaluation labels are compared against the model's own class list, so the test CSV
  should use the same label names that were used for training.

## Outputs

- `evaluations.csv` with overall metrics: accuracy, balanced accuracy,
  precision/recall/F1 (macro, micro, weighted), Matthews correlation coefficient,
  quadratic kappa, and ROC-AUC (OVO, OVR).
- `confusion_matrix.csv` — raw counts, true-label rows against predicted-label
  columns.
- `confusion_matrix_normalized.csv` — row-normalised matrix for per-class recall
  diagnosis.
- `per_class_metrics.csv` — per-class precision, recall, F1 and support.
- `confusion_matrix.pdf` — heatmap PDF, written when the class count stays
  readable.

## Examples

Evaluate the AutoGluon predictor produced by training:

```bash
entomokit classify evaluate --test-csv data/test.csv --images-dir data/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto --out-dir runs/eval/
```

## Notes

- `evaluate` is driven by a labelled CSV and is therefore outside the README's
  directory input/output policy for that CSV:
  [directory policy](../../README.md#directory-policy). Logging, device selection
  and version display are shared:
  [common behaviours](../../README.md#common-behaviours).
- A non-empty `--out-dir` requires `--overwrite`; there is no `--resume` flag.
- Read the normalised confusion matrix together with `per_class_metrics.csv` before
  trusting a single overall number on imbalanced data.

## Version Notes

- `0.7.0`: evaluation writes the overall metrics plus the per-class and
  confusion-matrix diagnostics listed above.
