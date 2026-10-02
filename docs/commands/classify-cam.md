# entomokit classify cam
[English](classify-cam.md) | [中文](classify-cam.cn.md)

## Purpose

Generate GradCAM-family saliency heatmaps for a trained classifier so predictions
can be inspected visually. Requires PyTorch model hooks, so ONNX models are not
supported.

## Usage

```bash
entomokit classify cam --images-dir data/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --cam-method gradcam --out-dir runs/cam/ --save-npy raw
```

## Parameters

### `--images-dir`

Required. No default. Directory containing images; also the root for the images
referenced by `--label-csv` when that is provided.

### `--out-dir`

Required. No default. Directory to write CAM visualisations and artifacts.

### `--label-csv`

Optional. No default; when omitted, every image in `--images-dir` is used. CSV with
`image` and `label` columns.

### `--model-dir`

Optional. No default. AutoGluon predictor directory.

### `--base-model`

Optional. No default. timm backbone name.

### `--checkpoint-path`

Optional. No default. Custom `.pth` weights for a timm backbone.

### `--num-classes`

Optional. No default. Class count used when loading a custom timm checkpoint.

### `--no-pretrained`

Optional. Default off (boolean flag without a value). Disable pretrained timm
weights when using `--base-model`.

### `--cam-method`

Optional. Default `gradcam`. Choices: `gradcam`, `gradcampp`, `layercam`,
`scorecam`, `eigencam`, `ablationcam`. CAM algorithm used to generate the saliency
maps.

### `--arch`

Optional. No default; the architecture is auto-detected. Choices: `cnn`, `vit`.
Forces the architecture type.

### `--target-layer-name`

Optional. No default; the layer is auto-selected. Specific model layer used for CAM.

### `--image-weight`

Optional. Default `0.5`. Blend weight of the original image in the CAM overlay,
from `0` to `1`.

### `--fig-format`

Optional. Default `png`. Choices: `png`, `jpg`, `pdf`. Output format for CAM
figures.

### `--save-npy`

Optional. Default `none`. Choices: `none`, `raw`, `normalized`. Saves CAM arrays to
`arrays/*.npy`: `none` writes nothing, `raw` keeps the unnormalised positive CAM
with magnitude preserved, `normalized` writes the per-image min-max `[0, 1]` mask.

### `--dump-model-structure`

Optional. Default off (boolean flag without a value). Write the model layer names
to `out-dir/model_layers.txt` for use with `--target-layer-name`.

### `--max-images`

Optional. No default; all images are processed. Maximum number of images to
process.

### `--cam-batch-size`

Optional. Default `32`. Batch size for CAM inference.

### `--eval-transform`

Optional. Default `center-crop`. Choices: `center-crop`, `whole-specimen-pad`.
Evaluation preprocessing: the model's saved deterministic centre crop, or
aspect-preserving square padding for full-specimen coverage.

### `--num-threads`

Optional. Default `0`. CPU threads for PyTorch operations; `0` selects
automatically.

### `--overwrite`

Optional. Default off (boolean flag without a value). Delete the contents of
`--out-dir` and regenerate the CAM visualisations.

### `--device`

Optional. Default `auto`. Choices: `auto`, `cpu`, `cuda`, `mps`. Compute device for
CAM generation.

## Inputs

- Either `--model-dir` or `--base-model` (with optional `--checkpoint-path` and
  `--no-pretrained`) selects the model; ONNX models are rejected because CAM needs
  PyTorch hooks.
- Architecture is auto-detected: Swin backbones are treated as ViT-style models with
  the final stage block as the default CAM target, and ConvNeXt backbones use the
  final stage block instead of the pointwise `mlp.fc2` layer to avoid degenerate
  maps.
- Run with `--dump-model-structure` first when a specific target layer is needed.

## Outputs

- `figures/` — CAM overlay images in the format selected by `--fig-format`.
- `cam_summary.csv` — metadata for the generated maps.
- `arrays/` — CAM arrays, only with `--save-npy raw` or `--save-npy normalized`.
- `model_layers.txt` — layer names, only with `--dump-model-structure`.

## Examples

CAM with ground-truth labels and a normalised array dump:

```bash
entomokit classify cam --label-csv data/test.csv --images-dir data/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --out-dir runs/cam/ --save-npy normalized
```

Full-specimen preprocessing with a different CAM method:

```bash
entomokit classify cam --images-dir data/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --cam-method gradcampp --eval-transform whole-specimen-pad --out-dir runs/cam/
```

Find a target layer:

```bash
entomokit classify cam --images-dir data/images/ \
    --model-dir runs/exp1/AutogluonModels/convnextv2_femto \
    --dump-model-structure --out-dir runs/cam/
```

## Notes

- CAM reads a directory; shared recursive discovery rules are the README's
  [directory policy](../../README.md#directory-policy). Logging, device selection
  and version display are shared:
  [common behaviours](../../README.md#common-behaviours).
- A non-empty `--out-dir` requires `--overwrite`; there is no `--resume` flag.
- Raw versus normalised arrays: `normalized` writes the same per-image min-max copy
  the overlay uses and preserves the previous normalised semantics. `raw` keeps the
  CAM magnitude and is only comparable within one model, target layer and
  preprocessing configuration; `eigencam` raw values are exported for completeness
  but are not suitable for response-magnitude statistics. The overlay keeps the same
  display semantics in both modes, and pixel-identical output is not guaranteed.
- `--eval-transform center-crop` (default) limits the heatmap to the model's
  centre-crop field of view; `whole-specimen-pad` pads the image to a square with its
  edge-median background colour before resizing, so the heatmap covers the complete
  specimen without aspect-ratio distortion.

### CAM contract

- With `--model-dir`, CAM wraps the AutoGluon `backbone -> classification head` pipeline
  and explains the final trained classes; `pred_class` in `cam_summary.csv` is the
  predictor's real class label, not a backbone feature index.
- The predictor's saved `ImageProcessor.val_processor` is reused, so the saved
  `image_size` is the only input-size source (224, 384 or a custom size); when no
  processor was saved the pipeline is rebuilt from the saved `val_transforms`.
- `center-crop` keeps that validation preprocessing and maps the heatmap back onto the
  full original image: pixels outside the model's field of view are darkened and
  desaturated rather than stretched, so the crop is never blown up to the whole frame.
  `whole-specimen-pad` pads to a square with the per-channel median of the image border
  and then resizes to the saved input size, covering the whole specimen.
- With `--base-model` and no saved processor, the timm data configuration is used and
  the heatmap mapping keeps the actual resize and crop sizes (for example
  `Resize(256) -> CenterCrop(224)`).
- Saved arrays are float32 in the **model input space**. `raw` keeps the unnormalised
  positive CAM magnitude (the library's internal `scale_cam_image()` is skipped while
  ReLU and resizing to the model input size are kept); the overlay always uses a
  separate per-image min-max copy. `normalized` matches the previous normalised
  model-space mask within float32 tolerance on the current fixtures, but it is neither
  bit-identical nor a cross-model/cross-input guarantee, and overlay PNG pixel
  differences are not a compatibility contract.
- `raw` magnitudes are comparable only within one model, target layer and
  preprocessing configuration, and `eigencam` raw values come from an SVD projection
  with arbitrary sign, so they are not usable for response-magnitude statistics.

## Version Notes

- `0.7.0`: `--save-npy` takes an explicit `none`/`raw`/`normalized` value, exporting
  unnormalised CAM arrays separately from the normalised ones.
