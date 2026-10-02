# entomokit classify predict
[English](classify-predict.md) | [中文](classify-predict.cn.md)

## Purpose

Run inference on images with either an AutoGluon predictor or an ONNX model and
write per-image predictions with class probabilities.

## Usage

```bash
entomokit classify predict --images-dir data/test/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto --out-dir runs/predict/
```

## Parameters

### `--input-csv`

Optional. No default. CSV with an `image` column. Provide at least one of
`--input-csv` or `--images-dir`.

### `--images-dir`

Optional. No default. Directory to scan for images, or the root used to resolve
relative `image` values from `--input-csv`. Provide at least one of
`--input-csv` or `--images-dir`.

### `--model-dir`

Optional. No default. AutoGluon predictor directory, typically
`out-dir/AutogluonModels/<base-model>` from `classify train`.

### `--onnx-model`

Optional. No default. ONNX model file path; requires `onnxruntime`. Mutually
exclusive with `--model-dir`.

### `--out-dir`

Required. No default. Directory to write prediction outputs.

### `--batch-size`

Optional. Default `32`. Batch size for model inference.

### `--num-workers`

Optional. Default `4`. Number of dataloader worker processes.

### `--num-threads`

Optional. Default `0`. CPU threads for PyTorch or the ONNX runtime; `0` selects
automatically.

### `--overwrite`

Optional. Default off (boolean flag without a value). Delete the contents of
`--out-dir` and re-predict all inputs.

### `--device`

Optional. Default `auto`. Choices: `auto`, `cpu`, `cuda`, `mps`. Compute device for
inference.

## Inputs

- Provide at least one of `--input-csv` or `--images-dir`.
- If the CSV `image` values are already readable paths, the CSV is used directly.
- If the CSV `image` values are names or relative paths, also provide `--images-dir`.
- If only `--images-dir` is given, every image in that directory is predicted.
- One of `--model-dir` or `--onnx-model` is required; ONNX needs `onnxruntime`
  (`pip install onnxruntime`, or the `classify` extras).
- `--images-dir` discovery is recursive. Discovered images are recorded as paths
  relative to `--images-dir` (for example `beetles/a.jpg`), so nested same-named
  images stay distinct. Explicit CSV paths passed through `--input-csv` are used
  unchanged.

## Outputs

- `out-dir/predictions/predictions.csv` with the same columns as AutoGluon output:
  `image`, `prediction`, `proba_<class_name>`, and so on. When inputs cannot be
  resolved, the missing-image list is written to `out-dir/logs/missing_images.txt`.
- With an ONNX model, `prediction` is the class name when `label_classes.json`
  exists next to the ONNX file, otherwise the numeric class index.

## Examples

Predict with an ONNX model:

```bash
entomokit classify predict --input-csv test.csv --onnx-model runs/onnx/model.onnx \
    --out-dir runs/predict/
```

Resolve relative CSV paths against an image root:

```bash
entomokit classify predict --input-csv out/split/test.known.csv \
    --images-dir data/Epidorcus/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto --out-dir runs/predict/
```

## Notes

- `predict` accepts CSV input and therefore is not governed by the README's
  directory input/output policy for the CSV path form:
  [directory policy](../../README.md#directory-policy). Logging, device selection
  and version display are shared:
  [common behaviours](../../README.md#common-behaviours).
- A non-empty `--out-dir` requires `--overwrite`; there is no `--resume` flag.
- Pass `--overwrite` to delete `--out-dir` contents and re-predict all inputs.

## Version Notes

- `0.7.0`: recursive `--images-dir` discovery records image paths relative to
  `--images-dir`, replacing basename-only records.
