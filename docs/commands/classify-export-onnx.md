# entomokit classify export-onnx
[English](classify-export-onnx.md) | [中文](classify-export-onnx.cn.md)

## Purpose

Export a trained AutoGluon image classifier to ONNX for deployment, together with
the class-label mapping used by `classify predict` and `classify evaluate`.

## Usage

```bash
entomokit classify export-onnx \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --out-dir runs/onnx/ --opset 17
```

## Parameters

### `--model-dir`

Required. No default. AutoGluon predictor directory to export.

### `--out-dir`

Required. No default. Output directory for `model.onnx`.

### `--opset`

Optional. Default `17`. ONNX opset version.

### `--sample-image`

Optional. No default; an auto-generated temporary image is used for tracing. Image
path used as the ONNX trace input.

### `--overwrite`

Optional. Default off (boolean flag without a value). Delete the contents of
`--out-dir` and re-export the ONNX model.

## Inputs

- `--model-dir` must be an AutoGluon predictor directory produced by
  `classify train`.
- `onnx` and `onnxruntime` are required; they ship with the `classify` extras.
- `--sample-image` should match the input size and channel layout the model was
  trained with; without it an auto-generated temporary image is traced.

## Outputs

- `model.onnx` — the exported ONNX model.
- `label_classes.json` — class-label mapping written next to the model; consuming
  prediction code uses it to report class names instead of numeric indices.

## Examples

Export with the default opset:

```bash
entomokit classify export-onnx \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto --out-dir runs/onnx/
```

Export and trace with a sample image:

```bash
entomokit classify export-onnx \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --out-dir runs/onnx/ --sample-image data/sample.jpg
```

## Notes

- The export reads a predictor directory and is CSV-independent; shared directory
  rules live in the README's [directory policy](../../README.md#directory-policy).
  Logging, device selection and version display are shared:
  [common behaviours](../../README.md#common-behaviours).
- A non-empty `--out-dir` requires `--overwrite`; there is no `--resume` flag.
- Keep `label_classes.json` next to `model.onnx`: `classify predict` reads it to map
  numeric outputs back to class names.

## Version Notes

- `0.7.0`: the export writes `label_classes.json` alongside `model.onnx` so
  ONNX predictions report class names.
